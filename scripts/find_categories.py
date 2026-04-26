#!/usr/bin/env python3
"""
find_categories.py
------------------
Fetches the full DigiKey category tree and prints every category
with its ID so you can find the right IDs for the main pull script.

Usage:
    python scripts/find_categories.py
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv("DIGIKEY_CLIENT_ID")
CLIENT_SECRET = os.getenv("DIGIKEY_CLIENT_SECRET")

TOKEN_URL      = "https://api.digikey.com/v1/oauth2/token"
CATEGORIES_URL = "https://api.digikey.com/products/v4/search/categories"


def get_token():
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
    return resp.json()["access_token"]


def main():
    token = get_token()
    headers = {
        "X-DIGIKEY-Client-Id":       CLIENT_ID,
        "Authorization":             f"Bearer {token}",
        "Accept":                    "application/json",
        "X-DIGIKEY-Locale-Site":     "US",
        "X-DIGIKEY-Locale-Language": "en",
        "X-DIGIKEY-Locale-Currency": "USD",
    }

    resp = requests.get(CATEGORIES_URL, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # The response has a top-level "Categories" list
    # Each category may have ChildCategories
    categories = data.get("Categories") or data  # handle both shapes

    print(f"\n{'ID':>6}  {'ProductCount':>12}  Name")
    print("-" * 70)

    def print_category(cat, indent=0):
        cat_id    = cat.get("CategoryId", cat.get("Id", "?"))
        name      = cat.get("Name", "?")
        count     = cat.get("ProductCount", 0)
        prefix    = "  " * indent
        print(f"{cat_id:>6}  {count:>12,}  {prefix}{name}")
        for child in cat.get("ChildCategories") or []:
            print_category(child, indent + 1)

    if isinstance(categories, list):
        for cat in categories:
            print_category(cat)
    else:
        print(json.dumps(categories, indent=2))

    # Also save to file for easy searching
    with open("data/categories.json", "w") as f:
        json.dump(data, f, indent=2)
    print("\n✅  Full category tree saved to data/categories.json")
    print("    Search it with: grep -i 'connector' data/categories.json")


if __name__ == "__main__":
    main()
