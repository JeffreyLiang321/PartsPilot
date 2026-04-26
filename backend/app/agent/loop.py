"""
loop.py
-------
The agent execution loop for PartsPilot.

This is the core of the agent — it orchestrates the conversation between
Claude and the tools, logging every step to the traces table.

How the loop works:
1. Build the initial message (system prompt + customer email)
2. Send to Claude with tool definitions
3. If Claude returns tool calls, execute them and feed results back
4. Repeat until Claude stops calling tools (produces final text response)
5. Log the full trace to Postgres

Why we implement this ourselves rather than using LangChain/LlamaIndex:
- We learn more about how agentic loops actually work
- We have full control over what gets logged
- No framework magic that's hard to explain in an interview
- The implementation is ~100 lines, not worth a heavy dependency
"""

import json
import uuid
import time
import logging
import psycopg2
from psycopg2.extras import Json
from anthropic import Anthropic
from typing import Optional

from .prompts import SYSTEM_PROMPT
from .tools import TOOL_DEFINITIONS, ToolExecutor

log = logging.getLogger(__name__)

# Maximum tool call iterations before giving up
# Prevents infinite loops if the agent gets confused
MAX_ITERATIONS = 10

MODEL = "claude-sonnet-4-5"


def run_agent(
    email_body: str,
    email_id: Optional[str],
    customer_id: Optional[str],
    conn,           # psycopg2 connection
    retriever,      # PartRetriever instance
    anthropic_client: Anthropic,
) -> dict:
    """
    Run the agent on a single customer email.

    Returns:
        {
            "trace_id":      str,
            "quote":         dict | None,
            "final_message": str,
            "tool_calls":    list of {tool, input, output, latency_ms},
            "total_latency_ms": int,
            "iterations":    int,
            "error":         str | None,
        }
    """
    trace_id  = str(uuid.uuid4())
    t_start   = time.time()
    tool_calls_log = []
    executor  = ToolExecutor(conn, retriever)
    final_quote = None
    error = None

    # Build the conversation — starts with just the customer email
    customer_context = f"\nCustomer ID: {customer_id}" if customer_id else ""
    messages = [
        {
            "role":    "user",
            "content": f"Please process this customer email and produce a quote.{customer_context}\n\n{email_body}"
        }
    ]

    log.info("Agent run started | trace=%s | email=%s | customer=%s",
             trace_id, email_id, customer_id)

    try:
        for iteration in range(MAX_ITERATIONS):
            log.info("  Iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            # Call Claude with tools available
            response = anthropic_client.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            # Add assistant's response to conversation history
            messages.append({
                "role":    "assistant",
                "content": response.content,
            })

            # Check if Claude wants to use tools
            tool_use_blocks = [
                block for block in response.content
                if block.type == "tool_use"
            ]

            # No tool calls — Claude is done
            if not tool_use_blocks:
                log.info("  Agent complete after %d iterations", iteration + 1)
                break

            # Execute each tool call Claude requested
            tool_results = []
            for block in tool_use_blocks:
                tool_name  = block.name
                tool_input = block.input

                log.info("    Tool call: %s(%s)", tool_name,
                         json.dumps(tool_input)[:100])

                # Run the tool
                result = executor.execute(tool_name, tool_input)
                latency = result.pop("_latency_ms", 0)

                # Log this tool call for the traces table
                tool_calls_log.append({
                    "tool":        tool_name,
                    "input":       tool_input,
                    "output":      result,
                    "latency_ms":  latency,
                    "iteration":   iteration + 1,
                })

                # Capture the quote if this was draft_quote
                if tool_name == "draft_quote" and result.get("status") == "complete":
                    final_quote = result

                # Build the tool result block for the next API call
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(result),
                })

                log.info("    Result: %s... (%dms)",
                         json.dumps(result)[:80], latency)

            # Add tool results back to the conversation
            messages.append({
                "role":    "user",
                "content": tool_results,
            })

            # If we just called draft_quote, one more Claude call
            # to get the final email response text, then stop
            if any(tc["tool"] == "draft_quote" for tc in tool_calls_log
                   if tc["iteration"] == iteration + 1):
                # One more call to get the final email response
                final_response = anthropic_client.messages.create(
                    model=MODEL,
                    max_tokens=600,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
                messages.append({
                    "role":    "assistant",
                    "content": final_response.content,
                })
                break

        else:
            # Reached MAX_ITERATIONS without finishing
            log.warning("Agent hit max iterations (%d)", MAX_ITERATIONS)
            error = f"Reached maximum iterations ({MAX_ITERATIONS})"

    except Exception as e:
        log.error("Agent error: %s", e)
        error = str(e)

    # Extract final text response from last assistant message
    final_message = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            for block in (msg["content"] if isinstance(msg["content"], list)
                          else [msg["content"]]):
                if hasattr(block, "type") and block.type == "text":
                    final_message = block.text
                    break
                elif isinstance(block, dict) and block.get("type") == "text":
                    final_message = block.get("text", "")
                    break
            if final_message:
                break

    total_latency_ms = round((time.time() - t_start) * 1000)

    # Write trace to database
    _save_trace(
        conn=conn,
        trace_id=trace_id,
        email_id=email_id,
        customer_id=customer_id,
        tool_calls=tool_calls_log,
        final_quote=final_quote,
        total_latency_ms=total_latency_ms,
        model=MODEL,
    )

    log.info("Agent run complete | trace=%s | %d tool calls | %dms",
             trace_id, len(tool_calls_log), total_latency_ms)

    return {
        "trace_id":          trace_id,
        "quote":             final_quote,
        "final_message":     final_message,
        "tool_calls":        tool_calls_log,
        "total_latency_ms":  total_latency_ms,
        "iterations":        len([tc for tc in tool_calls_log]) if tool_calls_log else 0,
        "error":             error,
    }


def _save_trace(conn, trace_id: str, email_id: Optional[str],
                customer_id: Optional[str], tool_calls: list,
                final_quote: Optional[dict], total_latency_ms: int,
                model: str):
    """
    Persist the agent run to the traces table.
    This is what the frontend trajectory panel reads.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO traces
                    (trace_id, email_id, customer_id, tool_calls,
                     final_quote, total_latency_ms, model)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trace_id) DO NOTHING
            """, (
                trace_id,
                email_id,
                customer_id,
                Json(tool_calls),
                Json(final_quote),
                total_latency_ms,
                model,
            ))
        conn.commit()
        log.info("  Trace saved: %s", trace_id)
    except Exception as e:
        log.error("  Failed to save trace: %s", e)
