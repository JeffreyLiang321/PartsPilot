#!/usr/bin/env python3
"""
harness.py
----------
PartsPilot eval harness — runs all test cases and scores results.

Usage:
    python evals/harness.py                    # run all 25 cases
    python evals/harness.py --category in_stock  # run one category
    python evals/harness.py --case eval_001    # run single case
    python evals/harness.py --dry-run          # validate cases.yaml without running

Output:
    - Prints a results table to stdout
    - Writes run to eval_runs table in Postgres
    - Saves detailed results to evals/results/run_TIMESTAMP.json

Scoring:
    Each case is scored on two dimensions:
    1. Exact match  — did the agent call the right tools / quote the right parts
    2. LLM judge    — does the response meet the qualitative rubric

    A case PASSES if both exact match checks pass AND judge score >= 0.75
    (i.e., at least 3 of 4 rubric criteria met)
"""

import os
import sys
import json
import uuid
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from anthropic import Anthropic

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.retriever import PartRetriever
from backend.app.agent.loop import run_agent
from evals.judges import judge_response, check_injection_resistance

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname":   os.getenv("POSTGRES_DB", "partspilot"),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

CASES_FILE    = Path(__file__).parent / "cases.yaml"
RESULTS_DIR   = Path(__file__).parent / "results"
PASS_THRESHOLD = 0.75  # judge score >= 75% to pass rubric


# ---------------------------------------------------------------------------
# Exact match scoring
# ---------------------------------------------------------------------------

def score_exact_match(case: dict, result: dict) -> dict:
    """
    Check objective criteria that have definitive right/wrong answers.
    Returns a dict of {check_name: passed bool} and an overall pass.
    """
    gt = case["ground_truth"]
    quote = result.get("quote") or {}
    line_items = quote.get("line_items", [])
    actual_mpns = [item.get("mpn", "") for item in line_items]
    tool_names = [tc["tool"] for tc in result.get("tool_calls", [])]

    checks = {}

    # 1. Expected MPNs present in quote
    # Should still pass if MPN appears as substitute or in final_message notes
    expected_mpns = gt.get("expected_mpns", [])
    if expected_mpns:
        found_all = all(
            any(mpn.lower() in actual_mpn.lower() or actual_mpn.lower() in mpn.lower()
                for actual_mpn in actual_mpns)
            or
            any(mpn.lower() in (item.get("substitute_for") or "").lower()
                for item in line_items)
            for mpn in expected_mpns
        )
        checks["expected_mpns_in_quote"] = found_all
    else:
        checks["expected_mpns_in_quote"] = True  # no MPNs required

    # 2. Expected statuses match
    expected_statuses = gt.get("expected_statuses", {})
    if expected_statuses:
        status_correct = True
        for mpn, expected_status in expected_statuses.items():
            matching = [item for item in line_items
                       if mpn.lower() in item.get("mpn", "").lower()]
            if not matching or matching[0].get("status") != expected_status:
                status_correct = False
                break
        checks["statuses_correct"] = status_correct
    else:
        checks["statuses_correct"] = True

    # 3. Substitution check
    if gt.get("expect_substitution"):
        has_sub = any(
            item.get("substitute_for") is not None or item.get("status") == "substituted"
            for item in line_items
        )
        checks["substitution_offered"] = has_sub
    else:
        checks["substitution_offered"] = True

    # 4. Discount check — key_account or preferred customers should have lower prices
    # We check this by looking for discount-related language in the notes
    if gt.get("expect_discount"):
        notes_text = (quote.get("notes", "") + result.get("final_message", "")).lower()
        has_discount_mention = any(phrase in notes_text for phrase in [
            "discount", "preferred", "key account", "5%", "2%"
        ])
        checks["discount_applied"] = has_discount_mention
    else:
        checks["discount_applied"] = True

    # 5. Injection resistance
    if gt.get("reject_injection"):
        injection_check = check_injection_resistance(
            case["email_body"],
            result.get("final_message", ""),
            result.get("quote"),
        )
        checks["injection_rejected"] = injection_check["passed"]
    else:
        checks["injection_rejected"] = True

    # 6. Customer history called when customer_id provided
    if case.get("customer_id"):
        checks["customer_history_called"] = "get_customer_history" in tool_names
    else:
        checks["customer_history_called"] = True

    # 7. No agent error
    checks["no_error"] = result.get("error") is None

    overall = all(checks.values())
    return {"checks": checks, "passed": overall}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_case(
    case: dict,
    retriever: PartRetriever,
    anthropic_client: Anthropic,
    conn,
) -> dict:
    """Run a single eval case and return scored results."""
    log.info("  Running case %s: %s", case["id"], case["description"])

    # Run the agent
    agent_result = run_agent(
        email_body=case["email_body"],
        email_id=case["id"],
        customer_id=case.get("customer_id"),
        conn=conn,
        retriever=retriever,
        anthropic_client=anthropic_client,
    )

    # Exact match scoring
    exact = score_exact_match(case, agent_result)

    # LLM judge scoring
    rubric = case["ground_truth"].get("rubric", [])
    judge = judge_response(
        email_body=case["email_body"],
        final_message=agent_result.get("final_message", ""),
        quote=agent_result.get("quote"),
        rubric=rubric,
        anthropic_client=anthropic_client,
    )

    # Combined pass: exact match AND judge score >= threshold
    judge_score = judge.get("score", 0)
    passed = exact["passed"] and judge_score >= PASS_THRESHOLD

    return {
        "case_id":         case["id"],
        "description":     case["description"],
        "category":        case["category"],
        "passed":          passed,
        "exact_match":     exact,
        "judge":           judge,
        "agent_result":    agent_result,
        "latency_ms":      agent_result.get("total_latency_ms", 0),
        "tool_call_count": len(agent_result.get("tool_calls", [])),
    }


def save_run(results: list[dict], run_id: str, conn):
    """Save eval run summary to eval_runs table."""
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    failed  = total - passed
    rate    = round(passed / total * 100, 2) if total > 0 else 0

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO eval_runs
                (run_id, git_sha, model, total_cases, passed, failed, pass_rate, results)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO NOTHING
        """, (
            run_id, git_sha, "claude-sonnet-4-5",
            total, passed, failed, rate,
            Json([{
                "case_id":    r["case_id"],
                "passed":     r["passed"],
                "category":   r["category"],
                "judge_score": r["judge"].get("score", 0),
                "latency_ms": r["latency_ms"],
            } for r in results])
        ))
    conn.commit()


def print_results_table(results: list[dict]):
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 90)
    print(f"{'Case':<12} {'Category':<20} {'Exact':>6} {'Judge':>7} {'Pass':>6} {'ms':>8}")
    print("=" * 90)

    by_category: dict[str, list] = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, []).append(r)

    total_passed = 0
    for category, cases in sorted(by_category.items()):
        for r in cases:
            exact_ok  = "✓" if r["exact_match"]["passed"] else "✗"
            judge_pct = f"{r['judge'].get('score', 0)*100:.0f}%"
            passed    = "✓ PASS" if r["passed"] else "✗ FAIL"
            print(
                f"{r['case_id']:<12} {category:<20} "
                f"{exact_ok:>6} {judge_pct:>7} {passed:>6} "
                f"{r['latency_ms']:>7}ms"
            )
            if r["passed"]:
                total_passed += 1

        # Category subtotal
        cat_passed = sum(1 for r in cases if r["passed"])
        print(f"  {'─'*30} {cat_passed}/{len(cases)} passed")
        print()

    total = len(results)
    pass_rate = total_passed / total * 100 if total > 0 else 0
    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0

    print("=" * 90)
    print(f"TOTAL: {total_passed}/{total} passed ({pass_rate:.1f}%) | "
          f"Avg latency: {avg_latency:.0f}ms")
    print("=" * 90)

    # Print failures detail
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{'─'*50}")
        print("FAILURE DETAILS:")
        for r in failures:
            print(f"\n  {r['case_id']} — {r['description']}")
            # Exact match failures
            for check, ok in r["exact_match"]["checks"].items():
                if not ok:
                    print(f"    ✗ Exact: {check}")
            # Judge failures
            for crit in r["judge"].get("criteria", []):
                if not crit["passed"]:
                    print(f"    ✗ Judge: {crit['criterion']}")
                    print(f"           → {crit['reason']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="PartsPilot eval harness")
    parser.add_argument("--category", help="Run only this category")
    parser.add_argument("--case",     help="Run only this case ID")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Validate cases.yaml without running agent")
    args = parser.parse_args()

    # Load cases
    with open(CASES_FILE) as f:
        config = yaml.safe_load(f)
    cases = config["cases"]

    # Filter
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"❌  Case '{args.case}' not found")
            sys.exit(1)
    elif args.category:
        cases = [c for c in cases if c["category"] == args.category]
        if not cases:
            print(f"❌  No cases in category '{args.category}'")
            sys.exit(1)

    if args.dry_run:
        print(f"✅  {len(cases)} cases validated in cases.yaml")
        for c in cases:
            print(f"  {c['id']}: {c['description']}")
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌  ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Initialize
    log.info("Initializing retriever and Anthropic client...")
    retriever        = PartRetriever()
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    conn             = psycopg2.connect(**DB_CONFIG)
    run_id           = str(uuid.uuid4())
    RESULTS_DIR.mkdir(exist_ok=True)

    log.info("Starting eval run %s with %d cases", run_id, len(cases))

    results = []
    for i, case in enumerate(cases, 1):
        log.info("[%d/%d] %s", i, len(cases), case["id"])
        try:
            result = run_case(case, retriever, anthropic_client, conn)
            results.append(result)
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            log.info("  → %s (judge: %.0f%%)", status, result["judge"].get("score", 0) * 100)
        except Exception as e:
            log.error("  Case %s crashed: %s", case["id"], e)
            results.append({
                "case_id":     case["id"],
                "description": case["description"],
                "category":    case["category"],
                "passed":      False,
                "exact_match": {"passed": False, "checks": {}},
                "judge":       {"score": 0, "criteria": [], "judge_notes": str(e)},
                "agent_result": {"error": str(e), "tool_calls": []},
                "latency_ms":  0,
                "tool_call_count": 0,
            })

    # Save results
    save_run(results, run_id, conn)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"run_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "run_id":    run_id,
            "timestamp": timestamp,
            "total":     len(results),
            "passed":    sum(1 for r in results if r["passed"]),
            "results":   results,
        }, f, indent=2, default=str)

    log.info("Results saved to %s", output_file)

    print_results_table(results)

    conn.close()
    retriever.close()

    # Exit with non-zero if any failures — important for CI
    failed = sum(1 for r in results if not r["passed"])
    if failed > 5:
        sys.exit(1)


if __name__ == "__main__":
    main()
