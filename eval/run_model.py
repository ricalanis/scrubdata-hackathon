"""Run a vanilla Ollama Cloud model through the eval harness (small batch).

Completes the eval matrix with a real LLM row alongside no-op/heuristic/oracle, so we
can see whether a fine-tune is needed and how big the gap is.

    uv run eval/run_model.py --n 12 --model glm-5.1
"""

from __future__ import annotations

import argparse
import random

from scrubdata.model_planner import make_ollama_planner
from scrubdata.planner import mock_plan
from training.generate import make_example

from . import metrics
from .run_eval import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--model", type=str, default="glm-5.1")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gold = []
    while len(gold) < args.n:
        ex = make_example(rng)
        if metrics.recovery(ex["clean_df"], ex["dirty_df"], ex["plan"]) >= 0.999:
            gold.append(ex)

    systems = {
        "ORACLE (gold)": lambda df, gp: gp,
        "HEURISTIC": lambda df, gp: mock_plan(df),
        f"VANILLA {args.model}": make_ollama_planner(args.model),
    }
    rows = {name: evaluate(fn, gold) for name, fn in systems.items()}

    cols = ["json_valid", "op_f1", "canon_f1", "canon_r", "recovery"]
    print(f"\nModel eval on {args.n} held-out examples (seed {args.seed})\n")
    print(f"{'system':<22}" + "".join(f"{c:>11}" for c in cols))
    print("-" * (22 + 11 * len(cols)))
    for name, m in rows.items():
        print(f"{name:<22}" + "".join(f"{m[c]:>11.3f}" for c in cols))
    print("\nNote: a vanilla model scores low canon_f1/recovery mostly from CONVENTION "
          "mismatch (its canonical forms/ops differ from our executor's) — which is "
          "exactly what fine-tuning aligns.")


if __name__ == "__main__":
    main()
