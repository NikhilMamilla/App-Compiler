"""
eval/run_evals.py — Evaluation Framework

Runs all 20 test cases through the full pipeline and tracks:
  - Success rate (valid JSON + schema passed)
  - Final score (0-100%)
  - Retries per request
  - Failure types
  - Latency per stage and total
  - Edge case handling (vague, conflicting, incomplete)

Outputs a formatted report table + saves results to eval_results.json
"""

import json
import time
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stage1_intent import extract_intent
from pipeline.stage2_design import design_system
from pipeline.stage3_schema import generate_schemas
from pipeline.stage4_refine import refine_schemas

# ─── Config ───────────────────────────────────────────────────────────────────

TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")
RESULTS_PATH    = os.path.join(os.path.dirname(__file__), "eval_results.json")

# ─── Single Test Runner ───────────────────────────────────────────────────────

def run_single(test_case: dict, verbose: bool = False) -> dict:
    """Run one test case through the full pipeline. Returns result dict."""
    tc_id    = test_case["id"]
    category = test_case["category"]
    prompt   = test_case["prompt"]
    expected = test_case.get("expected", {})

    result = {
        "id": tc_id,
        "category": category,
        "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
        "success": False,
        "failure_type": None,
        "final_score": 0.0,
        "initial_score": 0.0,
        "issues_found": 0,
        "issues_resolved": 0,
        "repair_calls": 0,
        "ready_for_codegen": False,
        "timings": {},
        "entities_count": 0,
        "roles_found": [],
        "has_auth": False,
        "has_payments": False,
        "has_clarifications": False,
        "has_conflicts_flagged": False,
        "assumptions": [],
        "error_message": None
    }

    total_start = time.time()

    try:
        # Stage 1
        t = time.time()
        intent = extract_intent(prompt, debug=False)
        result["timings"]["stage1_ms"] = round((time.time() - t) * 1000, 1)

        result["has_auth"]           = intent.auth.needed
        result["has_payments"]       = intent.payments.needed
        result["has_clarifications"] = len(intent.clarifications_needed) > 0
        result["roles_found"]        = intent.roles

        # Stage 2
        t = time.time()
        design = design_system(intent, debug=False)
        result["timings"]["stage2_ms"] = round((time.time() - t) * 1000, 1)

        result["has_conflicts_flagged"] = len(design.flagged_conflicts) > 0

        # Stage 3
        t = time.time()
        schemas = generate_schemas(design, debug=False)
        result["timings"]["stage3_ms"] = round((time.time() - t) * 1000, 1)

        result["entities_count"] = len(schemas.db.tables)

        # Stage 4
        t = time.time()
        refined = refine_schemas(schemas, debug=False)
        result["timings"]["stage4_ms"] = round((time.time() - t) * 1000, 1)

        audit = refined.audit
        result["initial_score"]    = round(audit.initial_score * 100, 1)
        result["final_score"]      = round(audit.final_score * 100, 1)
        result["issues_found"]     = len(audit.issues_found)
        result["issues_resolved"]  = len(audit.issues_found) - len(audit.issues_remaining)
        result["repair_calls"]     = audit.repair_calls
        result["ready_for_codegen"] = refined.ready_for_codegen
        result["assumptions"]      = audit.assumptions

        result["success"] = True

    except ValueError as e:
        result["failure_type"]  = "pipeline_error"
        result["error_message"] = str(e)[:200]
    except Exception as e:
        result["failure_type"]  = "unexpected_error"
        result["error_message"] = str(e)[:200]

    result["timings"]["total_ms"] = round((time.time() - total_start) * 1000, 1)

    # ── Expectation checks ────────────────────────────────────────────────────
    checks_passed = []
    checks_failed = []

    if "min_entities" in expected:
        if result["entities_count"] >= expected["min_entities"]:
            checks_passed.append(f"entities≥{expected['min_entities']}")
        else:
            checks_failed.append(f"entities<{expected['min_entities']} (got {result['entities_count']})")

    if expected.get("requires_auth"):
        if result["has_auth"]:
            checks_passed.append("auth_present")
        else:
            checks_failed.append("auth_missing")

    if expected.get("requires_payments"):
        if result["has_payments"]:
            checks_passed.append("payments_present")
        else:
            checks_failed.append("payments_missing")

    if expected.get("should_have_clarifications"):
        if result["has_clarifications"]:
            checks_passed.append("clarifications_present")
        else:
            checks_failed.append("no_clarifications_for_vague_prompt")

    if expected.get("should_flag_conflicts"):
        if result["has_conflicts_flagged"]:
            checks_passed.append("conflicts_flagged")
        else:
            checks_failed.append("conflicts_not_flagged")

    if expected.get("should_not_crash", True):
        if result["success"]:
            checks_passed.append("no_crash")
        else:
            checks_failed.append(f"crashed: {result['failure_type']}")

    result["checks_passed"] = checks_passed
    result["checks_failed"] = checks_failed
    result["check_score"]   = (
        round(len(checks_passed) / (len(checks_passed) + len(checks_failed)) * 100, 1)
        if (checks_passed or checks_failed) else 100.0
    )

    if verbose:
        status = "✅" if result["success"] else "❌"
        print(f"  {status} [{tc_id}] score={result['final_score']}% "
              f"latency={result['timings']['total_ms']}ms "
              f"checks={result['check_score']}%")
        if checks_failed:
            for f in checks_failed:
                print(f"      ⚠ {f}")

    return result


# ─── Report Printer ───────────────────────────────────────────────────────────

def print_report(results: list[dict], report_path: str = None):
    total       = len(results)
    successful  = [r for r in results if r["success"]]
    failed      = [r for r in results if not r["success"]]
    real_cases  = [r for r in results if r["category"] == "real"]
    edge_cases  = [r for r in results if r["category"] != "real"]

    avg_score   = sum(r["final_score"] for r in successful) / len(successful) if successful else 0
    avg_latency = sum(r["timings"]["total_ms"] for r in results) / total
    avg_repairs = sum(r["repair_calls"] for r in successful) / len(successful) if successful else 0
    avg_check   = sum(r["check_score"] for r in results) / total

    lines = []
    lines.append("\n" + "═" * 72)
    lines.append("  EVALUATION REPORT — App Compiler Pipeline")
    lines.append(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("═" * 72)

    # Summary table
    lines.append(f"\n{'Metric':<35} {'Value':>15}")
    lines.append("-" * 52)
    lines.append(f"{'Total test cases':<35} {total:>15}")
    lines.append(f"{'Successful runs':<35} {len(successful):>14}  ({len(successful)/total*100:.0f}%)")
    lines.append(f"{'Failed runs':<35} {len(failed):>14}  ({len(failed)/total*100:.0f}%)")
    real_ok  = len([r for r in real_cases if r['success']])
    edge_ok  = len([r for r in edge_cases if r['success']])
    real_pct = f"{real_ok/len(real_cases)*100:.0f}%" if real_cases else "N/A"
    edge_pct = f"{edge_ok/len(edge_cases)*100:.0f}%" if edge_cases else "N/A"
    lines.append(f"{'Real product prompts passed':<35} {real_ok:>14}  ({real_pct})")
    lines.append(f"{'Edge case prompts passed':<35} {edge_ok:>14}  ({edge_pct})")
    lines.append(f"{'Avg final score':<35} {avg_score:>14.1f}%")
    lines.append(f"{'Avg expectation checks passed':<35} {avg_check:>14.1f}%")
    lines.append(f"{'Avg latency':<35} {avg_latency:>13.0f}ms")
    lines.append(f"{'Avg repair calls per run':<35} {avg_repairs:>15.2f}")

    # Per-result table
    lines.append(f"\n{'ID':<12} {'Category':<20} {'✓':<3} {'Score':>6} {'Chk':>5} {'Latency':>9} {'Issues':>7} {'Repairs':>8}")
    lines.append("-" * 72)
    for r in results:
        ok      = "✅" if r["success"] else "❌"
        issues  = f"{r['issues_resolved']}/{r['issues_found']}"
        latency = f"{r['timings']['total_ms']:.0f}ms"
        lines.append(f"{r['id']:<12} {r['category']:<20} {ok:<3} {r['final_score']:>5.0f}% "
              f"{r['check_score']:>4.0f}% {latency:>9} {issues:>7} {r['repair_calls']:>8}")

    # Failure breakdown
    if failed:
        lines.append(f"\n── Failures ({len(failed)}) ──────────────────────────────────────────────")
        for r in failed:
            lines.append(f"  [{r['id']}] {r['failure_type']}: {r['error_message']}")

    # Edge case analysis
    lines.append(f"\n── Edge Case Handling ───────────────────────────────────────────────")
    for r in edge_cases:
        status = "✅" if r["success"] else "❌"
        checks_ok = ", ".join(r["checks_passed"]) or "none"
        checks_fail = ", ".join(r["checks_failed"]) or "none"
        lines.append(f"  {status} [{r['id']}] passed={checks_ok} | failed={checks_fail}")

    lines.append("\n" + "═" * 72)
    overall = len(successful) / total * 100
    grade = "🟢 EXCELLENT" if overall >= 90 else "🟡 GOOD" if overall >= 70 else "🔴 NEEDS WORK"
    lines.append(f"  Overall Success Rate: {overall:.0f}%  {grade}")
    lines.append("═" * 72 + "\n")

    report_content = "\n".join(lines)
    print(report_content)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Report saved to: {report_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_evals(
    test_cases_path: str = TEST_CASES_PATH,
    results_path: str = RESULTS_PATH,
    report_path: str = os.path.join(os.path.dirname(RESULTS_PATH), "eval_report.txt"),
    category_filter: str = None,      # "real" | "edge_vague" | None = all
    limit: int = None,                # run only first N cases
    verbose: bool = True
):
    with open(test_cases_path) as f:
        test_cases = json.load(f)

    if category_filter:
        test_cases = [tc for tc in test_cases if tc["category"] == category_filter]
    if limit:
        test_cases = test_cases[:limit]

    print(f"\nRunning {len(test_cases)} test cases…")
    print("─" * 50)

    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {tc['id']} — {tc['prompt'][:60]}…")
        result = run_single(tc, verbose=verbose)
        results.append(result)

        # Save incrementally so you don't lose data on crash
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print_report(results, report_path=report_path)

    print(f"Full results saved to: {results_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run eval suite")
    parser.add_argument("--category", default=None, help="Filter: real | edge_vague | edge_conflicting | edge_incomplete | edge_overloaded")
    parser.add_argument("--limit",    type=int, default=None, help="Run only first N test cases")
    parser.add_argument("--report",   default=None, help="Custom path for the text report")
    parser.add_argument("--quiet",    action="store_true", help="Less verbose output")
    args = parser.parse_args()

    run_evals(
        category_filter=args.category,
        limit=args.limit,
        report_path=args.report if args.report else os.path.join(os.path.dirname(RESULTS_PATH), "eval_report.txt"),
        verbose=not args.quiet
    )