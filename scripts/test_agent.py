#!/usr/bin/env python3
"""
test_agent.py
-------------
Run the agent on a single email directly — no server needed.
Use this to test the agent before wiring up the full FastAPI backend.

Usage:
    python scripts/test_agent.py

Requirements: all previous dependencies + anthropic
"""

import os
import json
import sys
import logging
from dotenv import load_dotenv
import psycopg2
from anthropic import Anthropic

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.retriever import PartRetriever
from backend.app.agent.loop import run_agent

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB", "partspilot"),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

# ---------------------------------------------------------------------------
# Test emails — try these in order of complexity
# ---------------------------------------------------------------------------

TEST_EMAILS = [
    {
        "name":        "Simple in-stock quote",
        "customer_id": "CUST-001",
        "body": """Hi,

Please quote 100x RC0402JR-070RL. Need them by end of month.

Thanks,
Mike"""
    },
    {
        "name":        "Multi-part quote",
        "customer_id": "CUST-003",
        "body": """Hello,

We need pricing for the following:
- 500x RC0402JR-070RL (0 ohm jumper resistors)
- 200x CL05B104KP5NNNC (100nF capacitors)
- 50x NTCG103JF103FT1 (NTC thermistors)

Please advise on availability and lead times.

Best,
Sarah"""
    },
    {
        "name":        "Out of stock scenario",
        "customer_id": "CUST-002",
        "body": """Hey,

Do you have PIC16F684-I/P in stock? Need about 25 units for a small production run.

- Tom"""
    },
    {
        "name":        "Semantic search query",
        "customer_id": "CUST-005",
        "body": """Hi team,

Looking for a temperature sensor in 0402 package, ideally NTC type,
for an automotive application. Need around 1000 units.

Can you quote whatever you have in stock?

Thanks"""
    },
]


def print_result(result: dict, test_name: str):
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")
    print(f"Trace ID:    {result['trace_id']}")
    print(f"Latency:     {result['total_latency_ms']}ms")
    print(f"Iterations:  {result['iterations']}")
    print(f"Error:       {result['error']}")

    print(f"\n--- Tool calls ({len(result['tool_calls'])}) ---")
    for i, tc in enumerate(result["tool_calls"], 1):
        print(f"  {i}. {tc['tool']}({json.dumps(tc['input'])[:60]}...)")
        print(f"     → {json.dumps(tc['output'])[:80]}... ({tc['latency_ms']}ms)")

    if result["quote"]:
        print(f"\n--- Quote ---")
        q = result["quote"]
        for item in q.get("line_items", []):
            print(f"  {item['mpn']} x{item.get('requested_qty', '?')} "
                  f"@ ${item['unit_price_usd']} "
                  f"[{item['status']}]")
        print(f"  Subtotal: ${q.get('subtotal_usd', 0):.2f}")
        if q.get("notes"):
            print(f"  Notes: {q['notes'][:100]}")

    print(f"\n--- Final message (first 600 chars) ---")
    print(result["final_message"][:600])


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌  ANTHROPIC_API_KEY not set in .env\n")
        return

    print("Initializing retriever (loads BM25 + sentence transformer)...")
    retriever = PartRetriever()
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    conn = psycopg2.connect(**DB_CONFIG)

    # Run just the first test by default, or pass an index as argv
    test_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test = TEST_EMAILS[test_index]

    print(f"\nRunning test {test_index}: '{test['name']}'")
    print(f"Email:\n{test['body']}\n")

    result = run_agent(
        email_body=test["body"],
        email_id=f"TEST-{test_index:03d}",
        customer_id=test.get("customer_id"),
        conn=conn,
        retriever=retriever,
        anthropic_client=anthropic_client,
    )

    print_result(result, test["name"])

    conn.close()
    retriever.close()

    print(f"\n✅  Done. Trace saved to DB: {result['trace_id']}")
    print(f"   View in TablePlus: SELECT * FROM traces WHERE trace_id = '{result['trace_id']}'")


if __name__ == "__main__":
    main()
