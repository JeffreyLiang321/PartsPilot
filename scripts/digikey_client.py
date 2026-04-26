#!/usr/bin/env python3
"""
digikey_client.py
-----------------
Pulls ~3,000 real electronic component records from the DigiKey
Product Information V4 API and saves them to data/raw_catalog.jsonl

Usage:
    python digikey_client.py

Requirements:
    pip install requests python-dotenv

Output:
    data/raw_catalog.jsonl   — one JSON record per line, ~3k parts
    data/raw_substitutions.jsonl — substitution/cross-ref pairs
"""

import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

CLIENT_ID     = os.getenv("DIGIKEY_CLIENT_ID")
CLIENT_SECRET = os.getenv("DIGIKEY_CLIENT_SECRET")

TOKEN_URL   = "https://api.digikey.com/v1/oauth2/token"
SEARCH_URL  = "https://api.digikey.com/products/v4/search/keyword"
DETAIL_URL  = "https://api.digikey.com/products/v4/search/{part_number}/productdetails"
SUB_URL     = "https://api.digikey.com/products/v4/search/{part_number}/substitutions"

# How many parts to pull per category (50 is the API max per request)
LIMIT_PER_REQUEST = 50
# How many pages to pull per category (50 parts x 6 pages = 300 per category)
PAGES_PER_CATEGORY = 6

OUTPUT_DIR = Path("data")
CATALOG_FILE = OUTPUT_DIR / "raw_catalog.jsonl"
SUBS_FILE    = OUTPUT_DIR / "raw_substitutions.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Categories to pull
# DigiKey category IDs for our 6 target categories.
# These IDs are stable — confirmed from the Reference APIs /categories endpoint.
# Format: (category_id, human_label, keyword_to_seed_search)
# ---------------------------------------------------------------------------

CATEGORIES = [
    (2,  "resistors",        "resistor"),
    (3,  "capacitors",       "capacitor"),
    (20, "connectors",       "connector header"),
    (25, "sensors",          "sensor"),
    (32, "microcontrollers", "microcontroller"),
    (43, "power_modules",    "power module dc dc"),
]

# ---------------------------------------------------------------------------
# OAuth: 2-legged client credentials flow
# ---------------------------------------------------------------------------

_token_cache: dict = {}

def get_access_token() -> str:
    """
    Fetches a new access token using the 2-legged client credentials flow.
    Tokens are valid for ~10 minutes; we cache and refresh automatically.
    """
    global _token_cache

    now = time.time()
    if _token_cache.get("access_token") and now < _token_cache.get("expires_at", 0) - 30:
        return _token_cache["access_token"]

    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError(
            "DIGIKEY_CLIENT_ID and DIGIKEY_CLIENT_SECRET must be set in your .env file."
        )

    log.info("Fetching new OAuth token...")
    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type":    "client_credentials",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    _token_cache = {
        "access_token": data["access_token"],
        "expires_at":   now + data.get("expires_in", 600),
    }
    log.info("Token obtained, expires in %ds", data.get("expires_in", 600))
    return _token_cache["access_token"]


def auth_headers() -> dict:
    return {
        "X-DIGIKEY-Client-Id": CLIENT_ID,
        "Authorization":       f"Bearer {get_access_token()}",
        "Content-Type":        "application/json",
        "Accept":              "application/json",
        "X-DIGIKEY-Locale-Site":     "US",
        "X-DIGIKEY-Locale-Language": "en",
        "X-DIGIKEY-Locale-Currency": "USD",
    }


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def keyword_search(keyword: str, category_id: int, offset: int = 0) -> dict:
    """
    POST /products/v4/search/keyword
    Returns up to LIMIT_PER_REQUEST products matching the keyword,
    filtered to a specific category.
    """
    payload = {
        "Keywords":    keyword,
        "Limit":       LIMIT_PER_REQUEST,
        "Offset":      offset,
        "FilterOptionsRequest": {
            "CategoryFilter": [{"Id": str(category_id)}],
        },
        "SortOptions": {
            "Field":     "None",
            "SortOrder": "Ascending",
        },
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                SEARCH_URL,
                headers=auth_headers(),
                json=payload,
                timeout=20,
            )

            # Rate limit: back off and retry
            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)   # 4s, 8s, 16s
                log.warning("Rate limited. Waiting %ds...", wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            log.error("Request failed (attempt %d/3): %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(3)

    return {}


def get_substitutions(digikey_part_number: str) -> list[dict]:
    """
    GET /products/v4/search/{part_number}/substitutions
    Returns substitute parts for a given DigiKey part number.
    This is how we build our cross-reference table from real data.
    """
    url = SUB_URL.format(part_number=requests.utils.quote(digikey_part_number, safe=""))
    try:
        resp = requests.get(url, headers=auth_headers(), timeout=15)
        if resp.status_code in (404, 400):
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("ProductSubstitutes", []) or []
    except requests.RequestException as e:
        log.debug("Substitution lookup failed for %s: %s", digikey_part_number, e)
        return []


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_part(raw: dict, category_label: str) -> dict:
    """
    Flattens a raw DigiKey product record into a clean dict
    suitable for storing in Postgres.
    """
    # Pull unit price — ProductVariations may have multiple package types
    unit_price = raw.get("UnitPrice") or 0.0
    variations = raw.get("ProductVariations") or []
    digikey_pn = ""
    if variations:
        first = variations[0]
        digikey_pn = first.get("DigiKeyProductNumber", "")
        pricing = first.get("StandardPricing") or []
        if pricing:
            unit_price = pricing[0].get("UnitPrice", unit_price)

    desc = raw.get("Description") or {}
    mfr  = raw.get("Manufacturer") or {}

    # Parameters is a list of {ParameterId, ParameterText, ValueId, ValueText}
    # We store as a flat dict {param_name: value_text} for easy querying
    params_raw = raw.get("Parameters") or []
    parameters = {
        p["ParameterText"]: p.get("ValueText", "")
        for p in params_raw
        if p.get("ParameterText")
    }

    return {
        "digikey_part_number":    digikey_pn,
        "manufacturer_part_number": raw.get("ManufacturerProductNumber", ""),
        "manufacturer":           mfr.get("Name", ""),
        "category":               category_label,
        "description":            desc.get("ProductDescription", ""),
        "detailed_description":   desc.get("DetailedDescription", ""),
        "unit_price_usd":         unit_price,
        "product_url":            raw.get("ProductUrl", ""),
        "datasheet_url":          raw.get("DatasheetUrl", ""),
        "photo_url":              raw.get("PhotoUrl", ""),
        "parameters":             parameters,   # JSONB in postgres
        "raw":                    raw,           # keep full raw blob for reference
    }


# ---------------------------------------------------------------------------
# Main pull logic
# ---------------------------------------------------------------------------

def pull_category(
    category_id: int,
    label: str,
    keyword: str,
    seen_parts: set,
    catalog_fh,
) -> int:
    """
    Pull PAGES_PER_CATEGORY pages of results for one category.
    Writes unique parts to catalog_fh (JSONL).
    Returns count of new parts written.
    """
    written = 0
    for page in range(PAGES_PER_CATEGORY):
        offset = page * LIMIT_PER_REQUEST
        log.info("  [%s] page %d/%d (offset=%d)...", label, page + 1, PAGES_PER_CATEGORY, offset)

        data = keyword_search(keyword, category_id, offset)
        products = data.get("Products") or []

        if not products:
            log.info("  [%s] no more results at offset %d", label, offset)
            break

        for raw in products:
            # Deduplicate by manufacturer part number
            mpn = raw.get("ManufacturerProductNumber", "")
            if not mpn or mpn in seen_parts:
                continue
            seen_parts.add(mpn)

            part = extract_part(raw, label)
            catalog_fh.write(json.dumps(part) + "\n")
            written += 1

        # Be polite to the API
        time.sleep(0.5)

    return written


def pull_substitutions(catalog_path: Path, subs_fh) -> int:
    """
    After the main catalog pull, iterate over a sample of parts
    and fetch their substitutes to build the cross-reference table.
    We sample up to 200 parts (to stay within rate limits) and
    only write pairs where a real substitute was found.
    """
    written = 0
    sampled = 0
    max_sample = 200

    log.info("Pulling substitutions for up to %d parts...", max_sample)

    with open(catalog_path) as f:
        for line in f:
            if sampled >= max_sample:
                break
            part = json.loads(line)
            dkpn = part.get("digikey_part_number", "")
            if not dkpn:
                continue

            subs = get_substitutions(dkpn)
            for sub in subs:
                sub_pn = sub.get("ManufacturerProductNumber", "")
                if sub_pn:
                    record = {
                        "source_digikey_pn":  dkpn,
                        "source_mpn":         part.get("manufacturer_part_number", ""),
                        "substitute_mpn":     sub.get("ManufacturerProductNumber", ""),
                        "substitute_mfr":     (sub.get("Manufacturer") or {}).get("Name", ""),
                        "substitute_desc":    sub.get("Description", ""),
                        "substitute_dk_pn":   sub.get("DigiKeyProductNumber", ""),
                        "substitute_type":    sub.get("SubstituteType", ""),  # "Direct" or "Similar"
                        "qty_available":      sub.get("QuantityAvailable", 0),
                    }
                    subs_fh.write(json.dumps(record) + "\n")
                    written += 1

            sampled += 1
            time.sleep(0.3)   # stay well under rate limit

    return written


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("\n❌  Missing credentials. Copy .env.template → .env and fill in your keys.\n")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Wipe previous runs so we start fresh
    CATALOG_FILE.unlink(missing_ok=True)
    SUBS_FILE.unlink(missing_ok=True)

    seen_parts: set = set()
    total_parts = 0

    log.info("=" * 60)
    log.info("Starting PartsPilot catalog pull")
    log.info("Target: ~%d parts across %d categories", 
             PAGES_PER_CATEGORY * LIMIT_PER_REQUEST * len(CATEGORIES),
             len(CATEGORIES))
    log.info("=" * 60)

    with open(CATALOG_FILE, "w") as cat_fh:
        for cat_id, label, keyword in CATEGORIES:
            log.info("Pulling category: %s (id=%d)", label, cat_id)
            n = pull_category(cat_id, label, keyword, seen_parts, cat_fh)
            total_parts += n
            log.info("  → %d new parts written (total so far: %d)", n, total_parts)
            # Pause between categories
            time.sleep(1)

    log.info("-" * 60)
    log.info("Catalog pull complete: %d unique parts in %s", total_parts, CATALOG_FILE)

    # Now pull substitutions
    with open(SUBS_FILE, "w") as subs_fh:
        total_subs = pull_substitutions(CATALOG_FILE, subs_fh)

    log.info("Substitutions pull complete: %d pairs in %s", total_subs, SUBS_FILE)
    log.info("=" * 60)
    log.info("✅  Done! Next step: run ingest.py to load into Postgres.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
