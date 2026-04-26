"""
main.py
-------
FastAPI application entry point for PartsPilot backend.

Exposes:
  POST /agent/run     — run the agent on an email
  GET  /emails        — list all seed emails (for the frontend inbox)
  GET  /emails/{id}   — get a single email
  GET  /traces/{id}   — get a trace (agent trajectory)
  GET  /health        — health check

Usage:
    uvicorn backend.app.main:app --reload --port 8000

Requirements:
    pip install fastapi uvicorn anthropic psycopg2-binary python-dotenv
"""

import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import psycopg2
import psycopg2.extras
from anthropic import Anthropic

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import our modules
# Adjust the import path depending on how you run the server
# ---------------------------------------------------------------------------

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.retriever import PartRetriever
from backend.app.agent.loop import run_agent

# ---------------------------------------------------------------------------
# DB config
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB", "partspilot"),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

# ---------------------------------------------------------------------------
# App state — initialized at startup, shared across requests
# ---------------------------------------------------------------------------

class AppState:
    retriever: PartRetriever = None
    anthropic: Anthropic = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    We initialize expensive objects once (retriever loads BM25 index,
    sentence transformer model) rather than per-request.
    """
    log.info("Starting PartsPilot backend...")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    state.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    state.retriever = PartRetriever()  # loads BM25 index + sentence transformer

    log.info("Backend ready.")
    yield

    # Cleanup on shutdown
    if state.retriever:
        state.retriever.close()
    log.info("Backend shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PartsPilot API",
    description="AI sales agent for electronics parts distribution",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the Next.js frontend (localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class AgentRunRequest(BaseModel):
    email_body:  str
    email_id:    Optional[str] = None
    customer_id: Optional[str] = None


class AgentRunResponse(BaseModel):
    trace_id:         str
    quote:            Optional[dict]
    final_message:    str
    tool_calls:       list
    total_latency_ms: int
    iterations:       int
    error:            Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check — confirms the server and DB are reachable."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM parts")
            count = cur.fetchone()[0]
        conn.close()
        return {"status": "ok", "parts_in_catalog": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/run", response_model=AgentRunResponse)
def agent_run(req: AgentRunRequest):
    """
    Run the agent on a customer email.

    The agent will:
    1. Look up customer history (if customer_id provided)
    2. Search/check inventory for all mentioned parts
    3. Find substitutes for OOS parts
    4. Draft a structured quote
    5. Log the full trace

    Returns the quote, final email response, and complete tool call log.
    """
    if not req.email_body.strip():
        raise HTTPException(status_code=400, detail="email_body cannot be empty")

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        result = run_agent(
            email_body=req.email_body,
            email_id=req.email_id,
            customer_id=req.customer_id,
            conn=conn,
            retriever=state.retriever,
            anthropic_client=state.anthropic,
        )
    finally:
        conn.close()

    return AgentRunResponse(**result)


@app.get("/emails")
def list_emails():
    """
    List all seed emails for the frontend inbox view.
    Returns emails ordered by difficulty so easy ones come first in the demo.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT e.email_id, e.subject, e.body, e.email_type,
                       e.difficulty, e.customer_id,
                       c.company_name, c.contact_name
                FROM emails e
                LEFT JOIN customers c ON e.customer_id = c.customer_id
                ORDER BY
                    CASE e.difficulty
                        WHEN 'easy'   THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'hard'   THEN 3
                        ELSE 4
                    END,
                    e.email_id
            """)
            emails = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return {"emails": emails, "count": len(emails)}


@app.get("/emails/{email_id}")
def get_email(email_id: str):
    """Get a single email by ID."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT e.*, c.company_name, c.contact_name, c.price_tier
                FROM emails e
                LEFT JOIN customers c ON e.customer_id = c.customer_id
                WHERE e.email_id = %s
            """, (email_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Email {email_id} not found")

    return dict(row)


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str):
    """
    Get a trace by ID — the full agent trajectory.
    Used by the frontend trajectory panel to show what the agent did.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM traces WHERE trace_id = %s
            """, (trace_id,))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    return dict(row)


@app.get("/traces")
def list_traces():
    """List recent traces — for the eval dashboard."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT trace_id, email_id, customer_id, total_latency_ms,
                       model, created_at,
                       jsonb_array_length(tool_calls) as tool_call_count
                FROM traces
                ORDER BY created_at DESC
                LIMIT 50
            """)
            traces = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return {"traces": traces, "count": len(traces)}
