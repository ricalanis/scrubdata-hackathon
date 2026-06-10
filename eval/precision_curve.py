"""WS1 deliverable: precision-coverage curve for the verified planner on real errors.

Sweeps the verifier threshold tau over the hospital benchmark and reports, per tau:
  precision  = repair_prec  (of the cells we changed, how many match gold)
  coverage   = repair_recall (of the real errors, how many we fixed)

GATE (publication plan): precision >= 0.70 at coverage >= 0.30. The verified planner
abstains on low-confidence merges instead of committing them — selective prediction at
the plan level, contract-preserving (dropped entries become review flags).

    uv run python -m eval.precision_curve                 # grounded heuristic planner
    uv run python -m eval.precision_curve --plan plan.json # pre-captured model plan
    uv run python -m eval.precision_curve --plan plan.json --union  # production pipeline
"""

from __future__ import annotations

import argparse
import json

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan
from scrubdata.verifier import union_plans, verify_plan

from .run_real import _ensure_data, _load
from .run_real_multi import score as _cn_score          # churn-neutral scoring

TAUS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def _repairs_only(plan: dict) -> dict:
    """Keep only the REPAIR decisions (canonicalize mappings); drop format/table ops.
    This is the Baran-comparable protocol: precision over error-repair decisions,
    not over convention standardization (dates->ISO etc., which the raw benchmark
    stores as text and would flood the denominator)."""
    import copy
    out = copy.deepcopy(plan)
    out["table_operations"] = []
    for c in out.get("columns", []):
        c["operations"] = [o for o in c.get("operations", [])
                           if o.get("op") == "canonicalize_categories"]
    out["columns"] = [c for c in out.get("columns", []) if c.get("operations")]
    return out


def curve(dirty, clean, base_plan: dict, label: str, union: bool = False) -> list[dict]:
    rows = []
    heuristic = mock_plan(dirty) if union else None
    print(f"\n=== precision-coverage: {label} (hospital, 509 real errors) ===")
    print(f"{'tau':>5}{'precision':>11}{'coverage':>10}{'changed':>9}{'fixed':>7}")
    print("-" * 44)
    for tau in TAUS:
        plan = verify_plan(dirty, base_plan, tau=tau)
        if union:                       # the production (active.py) composition
            plan = union_plans(plan, heuristic)
        plan = _repairs_only(plan)
        cleaned, _ = apply_plan(dirty, plan)
        m = _cn_score(dirty, clean, cleaned)
        rows.append({"tau": tau, "precision": m["precision"], "coverage": m["recall"],
                     "changed": m["_changed"], "fixed": m["_fixed"]})
        gate = "  <-- GATE" if m["precision"] >= 0.70 and m["recall"] >= 0.30 else ""
        print(f"{tau:>5.2f}{m['precision']:>11.3f}{m['recall']:>10.3f}"
              f"{m['_changed']:>9}{m['_fixed']:>7}{gate}")
    ok = [r for r in rows if r["precision"] >= 0.70 and r["coverage"] >= 0.30]
    best = max(ok, key=lambda r: r["coverage"]) if ok else max(rows, key=lambda r: (r["precision"] >= 0.70) * r["coverage"])
    if ok:
        print(f"\nGATE: PASS at tau={best['tau']} (precision {best['precision']:.3f}, "
              f"coverage {best['coverage']:.3f})")
    else:
        hi = max(rows, key=lambda r: r["precision"])
        print(f"\nGATE: not cleared — max precision {hi['precision']:.3f} at "
              f"coverage {hi['coverage']:.3f} (tau={hi['tau']})")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=str, default=None,
                    help="path to a captured raw plan JSON (e.g. the v6 model's)")
    ap.add_argument("--union", action="store_true",
                    help="union each verified plan with the grounded heuristic "
                         "(the shipped active.py pipeline)")
    ap.add_argument("--out", type=str, default=None, help="write curve rows to JSON")
    args = ap.parse_args()

    _ensure_data()
    dirty, clean = _load()
    if args.plan:
        base_plan = json.load(open(args.plan))
        label = f"model plan ({args.plan})" + (" + heuristic union" if args.union else "")
    else:
        base_plan = mock_plan(dirty)
        label = "grounded heuristic"
    rows = curve(dirty, clean, base_plan, label, union=args.union)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"curve written to {args.out}")


if __name__ == "__main__":
    main()
