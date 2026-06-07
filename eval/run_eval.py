"""Run the eval harness: baseline (heuristic) vs oracle (gold) on a held-out gold set.

    uv run eval/run_eval.py --n 300 --seed 4242

Plug a fine-tuned model in later by passing a planner callable to `evaluate`.
The held-out seed differs from the training seed so gold instances are unseen.
"""

from __future__ import annotations

import argparse
import random

from scrubdata.planner import mock_plan
from training.generate import make_example

from . import metrics


def _micro_f1(items, extract) -> dict:
    tp = fp = fn = 0
    for pred_plan, gold_plan in items:
        pred, gold = extract(pred_plan), extract(gold_plan)
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return metrics._prf(tp, fp, fn)


def evaluate(planner, gold) -> dict:
    """planner: (dirty_df, gold_plan) -> plan dict. gold: list of make_example dicts."""
    preds = [(planner(ex["dirty_df"], ex["plan"]), ex) for ex in gold]
    valid = sum(metrics.is_valid(p) for p, _ in preds) / len(preds)
    op = _micro_f1([(p, ex["plan"]) for p, ex in preds], metrics.op_pairs)
    canon = _micro_f1([(p, ex["plan"]) for p, ex in preds], metrics.canon_pairs)
    rec = sum(metrics.recovery(ex["clean_df"], ex["dirty_df"], p)
              for p, ex in preds) / len(preds)
    return {"json_valid": valid, "op_f1": op["f1"], "op_r": op["r"],
            "canon_f1": canon["f1"], "canon_r": canon["r"], "recovery": rec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Held-out gold: keep only oracle-solvable examples so the ceiling is a clean ~1.0.
    gold = []
    while len(gold) < args.n:
        ex = make_example(rng)
        if metrics.recovery(ex["clean_df"], ex["dirty_df"], ex["plan"]) >= 0.999:
            gold.append(ex)

    systems = {
        "ORACLE (gold plan)": lambda df, gold_plan: gold_plan,
        "HEURISTIC (baseline)": lambda df, gold_plan: mock_plan(df),
    }
    rows = {name: evaluate(fn, gold) for name, fn in systems.items()}

    cols = ["json_valid", "op_f1", "op_r", "canon_f1", "canon_r", "recovery"]
    print(f"\nEval on {args.n} held-out synthetic examples (seed {args.seed})\n")
    print(f"{'system':<22}" + "".join(f"{c:>11}" for c in cols))
    print("-" * (22 + 11 * len(cols)))
    for name, m in rows.items():
        print(f"{name:<22}" + "".join(f"{m[c]:>11.3f}" for c in cols))
    print("\nGoalpost: the fine-tuned model should approach ORACLE and clearly beat "
          "HEURISTIC — especially on canon_f1/canon_r (the fuzzy skill).")


if __name__ == "__main__":
    main()
