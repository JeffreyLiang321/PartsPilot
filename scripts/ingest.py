#!/usr/bin/env python3
"""
ingest.py
---------
Step 2: Load DigiKey catalog + substitutions into Postgres,
then generate synthetic augmentation (stock, customers, orders, emails).

Usage:
    python scripts/ingest.py

Requirements:
    pip install psycopg2-binary python-dotenv requests anthropic

What this script does:
    1. Creates all DB tables (parts, substitutions, customers, orders, emails)
    2. Loads raw_catalog.jsonl → parts table (with synthetic stock/warehouse)
    3. Loads raw_substitutions.jsonl → substitutions table
    4. Generates 20 customer profiles via Claude → customers table
    5. Generates 300 past orders via Claude → orders table
    6. Generates 50 incoming emails via Claude → emails table (your demo inputs)
"""

import os
import json
import random
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values, Json
import anthropic

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB", "partspilot"),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

DATA_DIR      = Path("data")
CATALOG_FILE  = DATA_DIR / "raw_catalog.jsonl"
SUBS_FILE     = DATA_DIR / "raw_substitutions.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def create_tables(conn):
    """Create all tables. Safe to re-run (uses IF NOT EXISTS)."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Parts table — the core catalog
        cur.execute("""
            CREATE TABLE IF NOT EXISTS parts (
                id                      SERIAL PRIMARY KEY,
                digikey_part_number     TEXT UNIQUE NOT NULL,
                manufacturer_part_number TEXT NOT NULL,
                manufacturer            TEXT,
                category                TEXT,
                description             TEXT,
                detailed_description    TEXT,
                unit_price_usd          NUMERIC(10,4),
                product_url             TEXT,
                datasheet_url           TEXT,
                photo_url               TEXT,
                parameters              JSONB,

                -- Synthetic fields (not from DigiKey)
                stock_qty               INTEGER DEFAULT 0,
                warehouse               TEXT,    -- 'east', 'west', 'central'
                lead_time_days          INTEGER DEFAULT 0,

                -- Embedding for semantic search (populated separately)
                embedding               vector(1536),

                created_at              TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_parts_category ON parts(category);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_parts_mpn ON parts(manufacturer_part_number);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_parts_manufacturer ON parts(manufacturer);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_parts_stock ON parts(stock_qty);")

        # Substitutions table — real DigiKey cross-reference data
        cur.execute("""
            CREATE TABLE IF NOT EXISTS substitutions (
                id                  SERIAL PRIMARY KEY,
                source_digikey_pn   TEXT NOT NULL,
                source_mpn          TEXT NOT NULL,
                substitute_mpn      TEXT NOT NULL,
                substitute_mfr      TEXT,
                substitute_desc     TEXT,
                substitute_dk_pn    TEXT,
                substitute_type     TEXT,   -- 'Direct', 'Similar', 'Upgrade', etc.
                qty_available       INTEGER DEFAULT 0,
                created_at          TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_subs_source_mpn ON substitutions(source_mpn);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_subs_source_dk ON substitutions(source_digikey_pn);")

        # Customers table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id              SERIAL PRIMARY KEY,
                customer_id     TEXT UNIQUE NOT NULL,
                company_name    TEXT NOT NULL,
                contact_name    TEXT,
                email           TEXT,
                industry        TEXT,
                price_tier      TEXT,   -- 'standard', 'preferred', 'key_account'
                notes           TEXT,   -- agent-readable context
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Orders table — past order history
        cur.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id              SERIAL PRIMARY KEY,
                order_id        TEXT UNIQUE NOT NULL,
                customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
                order_date      DATE NOT NULL,
                line_items      JSONB NOT NULL,  -- [{mpn, qty, unit_price, description}]
                total_usd       NUMERIC(10,2),
                status          TEXT DEFAULT 'completed',
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);")

        # Emails table — incoming customer emails (demo inputs + eval cases)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id              SERIAL PRIMARY KEY,
                email_id        TEXT UNIQUE NOT NULL,
                customer_id     TEXT REFERENCES customers(customer_id),
                subject         TEXT,
                body            TEXT NOT NULL,
                email_type      TEXT,   -- 'quote_request', 'availability_check', 'ambiguous', 'adversarial'
                difficulty      TEXT,   -- 'easy', 'medium', 'hard'
                ground_truth    JSONB,  -- expected mpns, quantities for eval
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Traces table — every agent run gets logged here
        cur.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id              SERIAL PRIMARY KEY,
                trace_id        TEXT UNIQUE NOT NULL,
                email_id        TEXT,
                customer_id     TEXT,
                tool_calls      JSONB,   -- [{tool, input, output, latency_ms}]
                final_quote     JSONB,
                total_latency_ms INTEGER,
                model           TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)

        # Eval runs table — CI eval results
        cur.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id              SERIAL PRIMARY KEY,
                run_id          TEXT UNIQUE NOT NULL,
                git_sha         TEXT,
                model           TEXT,
                total_cases     INTEGER,
                passed          INTEGER,
                failed          INTEGER,
                pass_rate       NUMERIC(5,2),
                results         JSONB,   -- per-case results
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)

    conn.commit()
    log.info("All tables created.")


# ---------------------------------------------------------------------------
# Stock level generation
# Realistic distributor inventory: most parts in stock, some scarce, some zero
# ---------------------------------------------------------------------------

def synthetic_stock(category: str, price: float) -> tuple[int, str, int]:
    """
    Returns (stock_qty, warehouse, lead_time_days).
    Distribution modeled on real distributor patterns:
    - 15% out of stock (lead time 14-60 days)
    - 20% low stock (1-25 units)
    - 65% healthy stock (25-5000 units, scaled by category)
    """
    warehouses = ["east", "west", "central"]
    warehouse  = random.choice(warehouses)

    roll = random.random()

    if roll < 0.15:
        # Out of stock
        return 0, warehouse, random.randint(14, 60)
    elif roll < 0.35:
        # Low stock — creates urgency in demos
        return random.randint(1, 25), warehouse, 0
    else:
        # Healthy stock — range varies by category and price
        if category in ("resistors", "capacitors"):
            qty = random.randint(500, 50000)   # passives: high volume
        elif category in ("connectors",):
            qty = random.randint(50, 5000)
        elif category in ("sensors", "power_modules"):
            qty = random.randint(10, 500)       # specialty: lower volume
        elif category in ("microcontrollers",):
            qty = random.randint(25, 2000)
        else:
            qty = random.randint(50, 1000)

        # Expensive parts have lower stock
        if price > 50:
            qty = max(1, qty // 20)
        elif price > 10:
            qty = max(1, qty // 5)

        return qty, warehouse, 0


# ---------------------------------------------------------------------------
# Load catalog
# ---------------------------------------------------------------------------

def load_catalog(conn) -> list[str]:
    """Load parts from raw_catalog.jsonl into the parts table."""
    parts = [json.loads(l) for l in open(CATALOG_FILE)]
    random.seed(42)   # reproducible stock levels

    rows = []
    for p in parts:
        stock, warehouse, lead_time = synthetic_stock(
            p.get("category", ""),
            float(p.get("unit_price_usd") or 0)
        )
        rows.append((
            p["digikey_part_number"],
            p["manufacturer_part_number"],
            p.get("manufacturer", ""),
            p.get("category", ""),
            p.get("description", ""),
            p.get("detailed_description", ""),
            float(p.get("unit_price_usd") or 0),
            p.get("product_url", ""),
            p.get("datasheet_url", ""),
            p.get("photo_url", ""),
            Json(p.get("parameters") or {}),
            stock,
            warehouse,
            lead_time,
        ))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO parts (
                digikey_part_number, manufacturer_part_number, manufacturer,
                category, description, detailed_description, unit_price_usd,
                product_url, datasheet_url, photo_url, parameters,
                stock_qty, warehouse, lead_time_days
            ) VALUES %s
            ON CONFLICT (digikey_part_number) DO NOTHING
        """, rows)

    conn.commit()

    # Quick stock distribution summary
    oos   = sum(1 for r in rows if r[11] == 0)
    low   = sum(1 for r in rows if 0 < r[11] <= 25)
    good  = sum(1 for r in rows if r[11] > 25)
    log.info("Catalog loaded: %d parts (%d OOS, %d low stock, %d healthy)",
             len(rows), oos, low, good)

    mpns = [p["manufacturer_part_number"] for p in parts]
    return mpns


# ---------------------------------------------------------------------------
# Load substitutions
# ---------------------------------------------------------------------------

def load_substitutions(conn):
    subs = [json.loads(l) for l in open(SUBS_FILE)]
    rows = [(
        s["source_digikey_pn"],
        s["source_mpn"],
        s["substitute_mpn"],
        s.get("substitute_mfr", ""),
        s.get("substitute_desc", ""),
        s.get("substitute_dk_pn", ""),
        s.get("substitute_type", ""),
        int(s.get("qty_available") or 0),
    ) for s in subs]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO substitutions (
                source_digikey_pn, source_mpn, substitute_mpn,
                substitute_mfr, substitute_desc, substitute_dk_pn,
                substitute_type, qty_available
            ) VALUES %s
            ON CONFLICT DO NOTHING
        """, rows)

    conn.commit()
    log.info("Substitutions loaded: %d pairs", len(rows))


# ---------------------------------------------------------------------------
# Claude generation helpers
# ---------------------------------------------------------------------------

def claude(prompt: str, system: str = "", max_tokens: int = 8000) -> str:
    """Single Claude call, returns text content."""
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": "claude-sonnet-4-6", "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


def claude_json(prompt: str, system: str = "", max_tokens: int = 8000) -> any:
    """Claude call expecting JSON back. Strips markdown fences."""
    raw = claude(prompt, system, max_tokens)
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
        clean = clean.rsplit("```", 1)[0]
    return json.loads(clean.strip())


# ---------------------------------------------------------------------------
# Generate customers
# ---------------------------------------------------------------------------

CUSTOMER_SYSTEM = """You generate realistic B2B customer profiles for an electronics
parts distributor. Customers are hardware engineering teams, contract manufacturers,
and electronics companies. Return ONLY valid JSON, no markdown, no preamble."""

def generate_customers(conn):
    log.info("Generating 20 customer profiles with Claude...")

    prompt = """Generate exactly 20 realistic customer profiles for an electronics parts distributor.
Mix of industries: contract manufacturing, IoT startups, automotive suppliers, medical devices,
industrial automation, defense contractors, consumer electronics.

Return a JSON array of 20 objects with these exact fields:
- customer_id: string like "CUST-001" through "CUST-020"
- company_name: realistic company name
- contact_name: first and last name of main buyer
- email: realistic work email matching company
- industry: one of [contract_manufacturing, iot, automotive, medical, industrial, defense, consumer_electronics]
- price_tier: one of [standard, preferred, key_account]
  (key_account = large volume buyers, preferred = regular buyers, standard = occasional)
- notes: 1-2 sentences an AI sales agent would find useful. Include typical order patterns,
  preferred brands if any, common part types they buy, payment terms, anything relevant.
  Be specific — mention actual component types like "primarily orders STM32 microcontrollers
  and passive components for their IoT gateway product line."

Make the notes genuinely useful and varied across customers."""

    customers = claude_json(prompt, CUSTOMER_SYSTEM)

    rows = [(
        c["customer_id"],
        c["company_name"],
        c.get("contact_name", ""),
        c.get("email", ""),
        c.get("industry", ""),
        c.get("price_tier", "standard"),
        c.get("notes", ""),
    ) for c in customers]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO customers (customer_id, company_name, contact_name, email, industry, price_tier, notes)
            VALUES %s ON CONFLICT (customer_id) DO NOTHING
        """, rows)

    conn.commit()
    log.info("Customers loaded: %d", len(rows))
    return customers


# ---------------------------------------------------------------------------
# Generate orders
# ---------------------------------------------------------------------------

def generate_orders(conn, customers: list, catalog_mpns: list):
    """Generate 300 orders in batches of 50 to avoid token truncation."""
    log.info("Generating ~300 past orders with Claude (6 batches of 50)...")

    sample_mpns = random.sample(catalog_mpns, min(60, len(catalog_mpns)))
    customer_ids = [c["customer_id"] for c in customers]

    BATCH_SIZE  = 20
    TOTAL       = 120
    all_orders  = []
    order_index = 1  # for unique order IDs across batches

    for batch_num in range(TOTAL // BATCH_SIZE):
        start = batch_num * BATCH_SIZE
        log.info("  Orders batch %d/6 (%d-%d)...", batch_num + 1, start + 1, start + BATCH_SIZE)

        # Give each batch a distinct date range so orders are spread across 18 months
        date_ranges = [
            ("2024-10-01", "2024-12-31"),
            ("2025-01-01", "2025-03-31"),
            ("2025-04-01", "2025-06-30"),
            ("2025-07-01", "2025-09-30"),
            ("2025-10-01", "2025-12-31"),
            ("2026-01-01", "2026-03-31"),
        ]
        date_from, date_to = date_ranges[batch_num]

        # Rotate MPNs per batch so different parts appear across batches
        batch_mpns = sample_mpns[batch_num * 10 % len(sample_mpns):][:20]
        if len(batch_mpns) < 10:
            batch_mpns = sample_mpns[:20]

        prompt = f"""Generate exactly {BATCH_SIZE} realistic past purchase orders for an electronics parts distributor.

Customer IDs to use: {json.dumps(customer_ids)}
Part numbers to use in line items: {json.dumps(batch_mpns)}
Order dates must be between {date_from} and {date_to}.
Order IDs must start from ORD-{order_index:04d} and go up to ORD-{order_index + BATCH_SIZE - 1:04d}.

Rules:
- Spread orders across different customers
- Each order has 1-4 line items
- line_items format: [{{"mpn": "...", "description": "...", "qty": N, "unit_price_usd": N.NN}}]
- total_usd = sum of qty * unit_price_usd across all line items
- status is always "completed"
- order_id format: "ORD-NNNN" e.g. "ORD-0001"

Return ONLY a JSON array of {BATCH_SIZE} objects with fields:
order_id, customer_id, order_date, line_items, total_usd, status

No markdown, no explanation, just the JSON array."""

        try:
            batch = claude_json(
                prompt,
                "Return ONLY a valid JSON array. No markdown fences, no preamble, no explanation."
            )
            all_orders.extend(batch)
            log.info("  Batch %d: got %d orders (total so far: %d)",
                     batch_num + 1, len(batch), len(all_orders))
        except Exception as e:
            log.error("  Batch %d failed: %s — skipping", batch_num + 1, e)

        order_index += BATCH_SIZE
        time.sleep(1)  # be polite between calls

    if not all_orders:
        log.warning("No orders generated — skipping DB insert.")
        return

    rows = []
    for o in all_orders:
        try:
            rows.append((
                o["order_id"],
                o["customer_id"],
                o["order_date"],
                Json(o.get("line_items", [])),
                float(o.get("total_usd", 0)),
                o.get("status", "completed"),
            ))
        except (KeyError, TypeError) as e:
            log.debug("Skipping malformed order: %s", e)

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO orders (order_id, customer_id, order_date, line_items, total_usd, status)
            VALUES %s ON CONFLICT (order_id) DO NOTHING
        """, rows)

    conn.commit()
    log.info("Orders loaded: %d", len(rows))


# ---------------------------------------------------------------------------
# Generate emails
# ---------------------------------------------------------------------------

EMAIL_SYSTEM = """You generate realistic incoming customer emails to an electronics parts
distributor sales rep. Emails are from hardware engineers and procurement managers.
Return ONLY valid JSON, no markdown, no preamble."""

def generate_emails(conn, customers: list, catalog_mpns: list):
    log.info("Generating 50 incoming emails with Claude (5 batches of 10)...")

    sample_mpns = random.sample(catalog_mpns, min(40, len(catalog_mpns)))
    customer_ids = [c["customer_id"] for c in customers]
    all_emails = []

    batch_configs = [
        {"start": 1,  "count": 10, "types": "4 easy quote_request, 3 medium quote_request, 2 easy availability_check, 1 hard ambiguous"},
        {"start": 11, "count": 10, "types": "3 easy quote_request, 2 medium quote_request, 2 medium availability_check, 2 hard ambiguous, 1 hard multi_item"},
        {"start": 21, "count": 10, "types": "3 easy quote_request, 2 medium quote_request, 2 medium availability_check, 2 hard multi_item, 1 easy availability_check"},
        {"start": 31, "count": 10, "types": "3 easy quote_request, 2 medium quote_request, 2 hard ambiguous, 2 medium availability_check, 1 adversarial"},
        {"start": 41, "count": 10, "types": "2 easy quote_request, 2 medium quote_request, 2 hard ambiguous, 2 hard multi_item, 1 medium availability_check, 1 adversarial"},
    ]

    for i, cfg in enumerate(batch_configs):
        log.info("  Emails batch %d/5...", i + 1)
        start, count = cfg["start"], cfg["count"]
        end = start + count - 1

        prompt = f"""Generate exactly {count} realistic incoming customer emails to an electronics parts distributor.
The AI agent will read these and draft quote responses.

Available customer IDs: {json.dumps(customer_ids)}
Real part numbers from catalog: {json.dumps(sample_mpns[:25])}

Generate this mix: {cfg["types"]}

Email styles should vary: terse, formal, chatty, typos, forwarded threads.
email_id values: "EMAIL-{start:03d}" through "EMAIL-{end:03d}"

For adversarial emails, include prompt injection attempts like "Ignore previous instructions".
ground_truth should be null for adversarial emails.

Return ONLY a JSON array of {count} objects with fields:
email_id, customer_id, subject, body, email_type, difficulty,
ground_truth (either null or {{mpns: [...], quantities: {{mpn: qty}}, notes: "..."}})"""

        try:
            batch = claude_json(prompt, EMAIL_SYSTEM)
            all_emails.extend(batch)
            log.info("  Batch %d: got %d emails", i + 1, len(batch))
        except Exception as e:
            log.error("  Emails batch %d failed: %s — skipping", i + 1, e)
        time.sleep(1)

    rows = [(
        e["email_id"],
        e.get("customer_id"),
        e.get("subject", ""),
        e["body"],
        e.get("email_type", "quote_request"),
        e.get("difficulty", "medium"),
        Json(e.get("ground_truth")),
    ) for e in all_emails]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO emails (email_id, customer_id, subject, body, email_type, difficulty, ground_truth)
            VALUES %s ON CONFLICT (email_id) DO NOTHING
        """, rows)

    conn.commit()
    log.info("Emails loaded: %d", len(rows))

    with open(DATA_DIR / "emails.jsonl", "w") as f:
        for e in all_emails:
            f.write(json.dumps(e) + "\n")
    log.info("Emails also saved to data/emails.jsonl")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify(conn):
    with conn.cursor() as cur:
        tables = ["parts", "substitutions", "customers", "orders", "emails", "traces", "eval_runs"]
        log.info("=" * 50)
        log.info("Database summary:")
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            log.info("  %-20s %d rows", table, count)

        # Stock distribution
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE stock_qty = 0)   as out_of_stock,
                COUNT(*) FILTER (WHERE stock_qty BETWEEN 1 AND 25) as low_stock,
                COUNT(*) FILTER (WHERE stock_qty > 25)  as healthy
            FROM parts
        """)
        oos, low, good = cur.fetchone()
        log.info("  Stock: %d OOS / %d low / %d healthy", oos, low, good)

        # Email type breakdown
        cur.execute("SELECT email_type, COUNT(*) FROM emails GROUP BY email_type ORDER BY COUNT(*) DESC")
        log.info("  Email types: %s", dict(cur.fetchall()))
    log.info("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n❌  ANTHROPIC_API_KEY not set in .env — needed for generating customers/orders/emails.\n")
        return

    log.info("Connecting to Postgres...")
    conn = get_conn()
    log.info("Connected.")

    log.info("Creating tables...")
    create_tables(conn)

    log.info("Loading catalog...")
    mpns = load_catalog(conn)

    log.info("Loading substitutions...")
    load_substitutions(conn)

    log.info("Generating synthetic data with Claude...")
    customers = generate_customers(conn)
    time.sleep(1)

    generate_orders(conn, customers, mpns)
    time.sleep(1)

    generate_emails(conn, customers, mpns)

    verify(conn)
    conn.close()

    log.info("✅  Ingest complete. Ready for Step 3: hybrid retriever.")


if __name__ == "__main__":
    main()
