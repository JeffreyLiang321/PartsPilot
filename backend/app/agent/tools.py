"""
tools.py
--------
Tool definitions and implementations for the PartsPilot agent.

Each tool has two parts:
1. A JSON Schema definition (sent to Claude so it knows what tools exist
   and what parameters they accept)
2. A Python implementation (the actual function that runs when Claude
   calls the tool)

Why separate these?
The JSON Schema is part of the API call to Claude — it's what lets Claude
"see" the tools. The Python implementation runs server-side and never
goes to Claude directly. Keeping them together in one file makes it easy
to ensure they stay in sync.

Tool design principles:
- Tools should do ONE thing and do it well
- Return structured dicts, not strings — the agent needs to reason over results
- Include enough context in returns that the agent can make decisions
  (e.g., don't just return stock_qty, also return lead_time_days)
- Fail gracefully — return empty results rather than raising exceptions
"""

import json
import time
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas — sent to Claude via the API
# These follow the Anthropic tool use format exactly.
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_catalog",
        "description": (
            "Search the parts catalog using natural language or part number queries. "
            "Use this when you have a description, specs, or manufacturer name but not "
            "an exact part number. Returns up to 10 matching parts with stock info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query or partial part number. "
                                   "Examples: '10k resistor 0402', 'Microchip 8-bit MCU', "
                                   "'temperature sensor automotive grade'"
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter. One of: resistors, capacitors, "
                                   "connectors, sensors, microcontrollers, power_modules",
                    "enum": ["resistors", "capacitors", "connectors", "sensors",
                             "microcontrollers", "power_modules"]
                },
                "in_stock_only": {
                    "type": "boolean",
                    "description": "If true, only return parts with stock_qty > 0. "
                                   "Default false."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "check_inventory",
        "description": (
            "Get exact stock, price, and lead time for a specific part by manufacturer "
            "part number (MPN). Use this when you have an exact MPN from the customer's "
            "email or from search_catalog results. Much faster than search_catalog."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mpn": {
                    "type": "string",
                    "description": "Manufacturer part number exactly as it appears. "
                                   "Example: 'RC0402JR-070RL'"
                }
            },
            "required": ["mpn"]
        }
    },
    {
        "name": "find_substitution",
        "description": (
            "Find in-stock substitute parts for an out-of-stock or unavailable part. "
            "Returns real DigiKey-curated substitutions ordered by type: Direct replacements "
            "first (drop-in compatible), then Similar, then Upgrade. "
            "Always call this when check_inventory returns stock_qty = 0."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mpn": {
                    "type": "string",
                    "description": "Manufacturer part number of the out-of-stock part"
                },
                "in_stock_only": {
                    "type": "boolean",
                    "description": "If true, only return substitutes with qty_available > 0. "
                                   "Default true."
                }
            },
            "required": ["mpn"]
        }
    },
    {
        "name": "get_customer_history",
        "description": (
            "Retrieve a customer's profile and past order history. Use this at the start "
            "of every interaction when a customer_id is available. Returns the customer's "
            "industry, price tier (affects discount), and recent orders — useful for "
            "understanding their typical parts and quantities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID in format CUST-XXX. Example: 'CUST-001'"
                }
            },
            "required": ["customer_id"]
        }
    },
    {
        "name": "draft_quote",
        "description": (
            "Produce the final structured quote once you have looked up all parts "
            "and their inventory status. This is the LAST tool you call — it signals "
            "you are done researching and ready to respond to the customer. "
            "Do not call this until you have checked inventory for every part in the request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "line_items": {
                    "type": "array",
                    "description": "Array of quote line items",
                    "items": {
                        "type": "object",
                        "properties": {
                            "mpn":              {"type": "string"},
                            "manufacturer":     {"type": "string"},
                            "description":      {"type": "string"},
                            "requested_qty":    {"type": "integer"},
                            "available_qty":    {"type": "integer"},
                            "unit_price_usd":   {"type": "number"},
                            "total_price_usd":  {"type": "number"},
                            "status":           {
                                "type": "string",
                                "enum": ["in_stock", "low_stock", "out_of_stock",
                                         "substituted", "lead_time"]
                            },
                            "substitute_for":   {"type": ["string", "null"]},
                            "lead_time_days":   {"type": "integer"},
                            "warehouse":        {"type": "string"},
                            "notes":            {"type": "string"}
                        },
                        "required": ["mpn", "manufacturer", "description",
                                     "requested_qty", "unit_price_usd", "status"]
                    }
                },
                "subtotal_usd":         {"type": "number"},
                "currency":             {"type": "string"},
                "valid_days":           {"type": "integer"},
                "notes":                {"type": "string"},
                "clarification_needed": {"type": ["string", "null"]}
            },
            "required": ["line_items", "subtotal_usd"]
        }
    }
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class ToolExecutor:
    """
    Runs tool calls from the agent.
    Initialized once per agent run with a DB connection and retriever.
    """

    def __init__(self, conn, retriever):
        """
        conn:      psycopg2 connection
        retriever: PartRetriever instance (has .search() method)
        """
        self.conn      = conn
        self.retriever = retriever

    def execute(self, tool_name: str, tool_input: dict) -> dict:
        """
        Dispatch a tool call by name. Returns the result dict.
        Logs execution time for the traces table.
        """
        t0 = time.time()
        try:
            if tool_name == "search_catalog":
                result = self._search_catalog(**tool_input)
            elif tool_name == "check_inventory":
                result = self._check_inventory(**tool_input)
            elif tool_name == "find_substitution":
                result = self._find_substitution(**tool_input)
            elif tool_name == "get_customer_history":
                result = self._get_customer_history(**tool_input)
            elif tool_name == "draft_quote":
                result = self._draft_quote(**tool_input)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            log.error("Tool %s failed: %s", tool_name, e)
            result = {"error": str(e)}

        result["_latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    def _search_catalog(self, query: str, category: str = None,
                        in_stock_only: bool = False) -> dict:
        """
        Hybrid BM25 + vector search over parts catalog.
        Returns top 10 results with key fields for agent reasoning.
        """
        results = self.retriever.search(
            query=query,
            top_k=10,
            category=category,
            in_stock_only=in_stock_only,
        )

        # Return only the fields the agent needs — don't flood the context
        # with raw parameters JSONB or internal scoring fields
        parts = []
        for r in results:
            parts.append({
                "mpn":                  r["manufacturer_part_number"],
                "manufacturer":         r["manufacturer"],
                "category":             r["category"],
                "description":          r["description"],
                "detailed_description": r.get("detailed_description", ""),
                "unit_price_usd":       float(r["unit_price_usd"] or 0),
                "stock_qty":            r["stock_qty"],
                "warehouse":            r["warehouse"],
                "lead_time_days":       r["lead_time_days"],
                "product_url":          r.get("product_url", ""),
            })

        return {
            "query":        query,
            "results_count": len(parts),
            "parts":        parts,
        }

    def _check_inventory(self, mpn: str) -> dict:
        """
        Exact lookup by manufacturer part number.
        Returns full inventory status for one part.
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT manufacturer_part_number, manufacturer, category,
                       description, detailed_description, unit_price_usd,
                       stock_qty, warehouse, lead_time_days, product_url,
                       datasheet_url, parameters
                FROM parts
                WHERE LOWER(manufacturer_part_number) = LOWER(%s)
                LIMIT 1
            """, (mpn,))
            row = cur.fetchone()

        if not row:
            return {
                "found":   False,
                "mpn":     mpn,
                "message": f"Part '{mpn}' not found in catalog. Try search_catalog instead."
            }

        r = dict(row)

        # Determine stock status for the agent
        qty = r["stock_qty"]
        if qty == 0:
            status = "out_of_stock"
        elif qty <= 25:
            status = "low_stock"
        else:
            status = "in_stock"

        return {
            "found":                True,
            "mpn":                  r["manufacturer_part_number"],
            "manufacturer":         r["manufacturer"],
            "category":             r["category"],
            "description":          r["description"],
            "detailed_description": r["detailed_description"],
            "unit_price_usd":       float(r["unit_price_usd"] or 0),
            "stock_qty":            qty,
            "status":               status,
            "warehouse":            r["warehouse"],
            "lead_time_days":       r["lead_time_days"],
            "product_url":          r["product_url"],
            "datasheet_url":        r["datasheet_url"],
            "parameters":           dict(r["parameters"]) if r["parameters"] else {}, 
        }

    def _find_substitution(self, mpn: str, in_stock_only: bool = True) -> dict:
        """
        Find substitute parts from the DigiKey cross-reference table.
        Returns up to 5 alternatives, Direct substitutes first.
        """
        stock_clause = "AND qty_available > 0" if in_stock_only else ""

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT substitute_mpn, substitute_mfr, substitute_desc,
                       substitute_type, qty_available, substitute_dk_pn
                FROM substitutions
                WHERE LOWER(source_mpn) = LOWER(%s)
                {stock_clause}
                ORDER BY
                    CASE WHEN LOWER(substitute_mpn) IN (
                        SELECT LOWER(manufacturer_part_number) FROM parts WHERE stock_qty > 0
                    ) THEN 0 ELSE 1 END,
                    CASE substitute_type
                        WHEN 'Direct'               THEN 1
                        WHEN 'MFR Recommended'      THEN 2
                        WHEN 'Parametric Equivalent' THEN 3
                        WHEN 'Similar'              THEN 4
                        WHEN 'Upgrade'              THEN 5
                        ELSE 6
                    END,
                    qty_available DESC
                LIMIT 5
            """, (mpn,))
            rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            return {
                "source_mpn":   mpn,
                "found":        False,
                "substitutes":  [],
                "message":      f"No {'in-stock ' if in_stock_only else ''}substitutes found for {mpn}."
            }

        return {
            "source_mpn":   mpn,
            "found":        True,
            "substitutes":  [
                {
                    "mpn":           r["substitute_mpn"],
                    "manufacturer":  r["substitute_mfr"],
                    "description":   r["substitute_desc"],
                    "type":          r["substitute_type"],
                    "qty_available": r["qty_available"],
                }
                for r in rows
            ]
        }

    def _get_customer_history(self, customer_id: str) -> dict:
        """
        Fetch customer profile + last 10 orders.
        Gives the agent context for pricing and preference decisions.
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Customer profile
            cur.execute("""
                SELECT customer_id, company_name, contact_name, email,
                       industry, price_tier, notes
                FROM customers
                WHERE LOWER(customer_id) = LOWER(%s)
            """, (customer_id,))
            customer = cur.fetchone()

            if not customer:
                return {
                    "found":   False,
                    "message": f"Customer '{customer_id}' not found."
                }

            # Last 10 orders, most recent first
            cur.execute("""
                SELECT order_id, order_date, total_usd, line_items
                FROM orders
                WHERE customer_id = %s
                ORDER BY order_date DESC
                LIMIT 10
            """, (customer["customer_id"],))
            orders = [dict(r) for r in cur.fetchall()]

        # Summarize order history into a readable format for the agent
        # This avoids flooding the context window with raw JSONB
        order_summary = []
        for o in orders:
            items = o["line_items"] if isinstance(o["line_items"], list) else []
            mpns = [item.get("mpn", "") for item in items]
            order_summary.append({
                "order_id":   o["order_id"],
                "date":       str(o["order_date"]),
                "total_usd":  float(o["total_usd"]),
                "parts":      mpns,
            })

        return {
            "found":          True,
            "customer_id":    customer["customer_id"],
            "company_name":   customer["company_name"],
            "contact_name":   customer["contact_name"],
            "industry":       customer["industry"],
            "price_tier":     customer["price_tier"],
            "notes":          customer["notes"],
            "order_count":    len(orders),
            "recent_orders":  order_summary,
        }

    def _draft_quote(self, line_items: list, subtotal_usd: float,
                     currency: str = "USD", valid_days: int = 30,
                     notes: str = "", clarification_needed: str = None) -> dict:
        """
        Finalizes and returns the structured quote.
        This is a passthrough — the agent builds the quote, we just
        validate and return it so it gets logged in the trace.
        """
        # Apply discount logic based on what the agent will have told us
        # via customer history (price_tier comes from get_customer_history)
        return {
            "status":               "complete",
            "line_items":           line_items,
            "subtotal_usd":         round(subtotal_usd, 2),
            "currency":             currency,
            "valid_days":           valid_days,
            "notes":                notes,
            "clarification_needed": clarification_needed,
        }
