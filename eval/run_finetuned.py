"""Evaluate the fine-tuned model (local Ollama GGUF) on BOTH layers vs the goalposts.

    ollama pull hf.co/ricalanis/scrubdata-qwen3-4b-gguf
    uv run eval/run_finetuned.py --model hf.co/ricalanis/scrubdata-qwen3-4b-gguf --n 40

Prints the synthetic matrix (vs heuristic + oracle) and the real-data row, then checks
each goalpost (eval/README.md): recovery≥0.95, canon_f1≥0.85, op_f1≥0.95, json_valid≥0.99
(synthetic) and recovery≥0.985, repair_recall≥0.30, broken≤50 (real).
"""

from __future__ import annotations

import argparse
import random

from scrubdata.executor import apply_plan
from scrubdata.model_planner import make_local_ollama_planner
from scrubdata.planner import mock_plan
from training.generate import make_example

from . import metrics
from .run_eval import evaluate
from .run_real import _ensure_data, _load, _score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local ollama model id (the FT GGUF)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    ft = make_local_ollama_planner(args.model)

    # ---- Layer 1: synthetic held-out matrix ----
    rng = random.Random(args.seed)
    gold = []
    while len(gold) < args.n:
        ex = make_example(rng)
        if metrics.recovery(ex["clean_df"], ex["dirty_df"], ex["plan"]) >= 0.999:
            gold.append(ex)
    systems = {
        "ORACLE (gold)": lambda df, gp: gp,
        "HEURISTIC": lambda df, gp: mock_plan(df),
        f"FT {args.model.split('/')[-1]}": ft,
    }
    rows = {name: evaluate(fn, gold) for name, fn in systems.items()}
    cols = ["json_valid", "op_f1", "canon_f1", "canon_r", "recovery"]
    print(f"\n=== Layer 1: synthetic ({args.n} held-out, seed {args.seed}) ===")
    print(f"{'system':<26}" + "".join(f"{c:>11}" for c in cols))
    print("-" * (26 + 11 * len(cols)))
    for name, m in rows.items():
        print(f"{name:<26}" + "".join(f"{m[c]:>11.3f}" for c in cols))
    ftm = rows[f"FT {args.model.split('/')[-1]}"]
    gp1 = {"recovery": 0.95, "canon_f1": 0.85, "op_f1": 0.95, "json_valid": 0.99}
    print("\nGoalpost check (synthetic):")
    for k, t in gp1.items():
        ok = "✅" if ftm[k] >= t else "❌"
        print(f"  {ok} {k}: {ftm[k]:.3f} (target ≥{t})")

    # ---- Layer 2: real OOD (Raha hospital) ----
    _ensure_data()
    dirty, clean = _load()
    ft_plan = ft(dirty)
    cleaned, _ = apply_plan(dirty, ft_plan)
    noop = _score(dirty, clean, dirty)
    ftr = _score(dirty, clean, cleaned)
    print(f"\n=== Layer 2: real OOD (Raha hospital, {noop['_errors']} errors) ===")
    rcols = ["recovery", "repair_recall", "repair_prec", "broken"]
    print(f"{'system':<26}" + "".join(f"{c:>14}" for c in rcols))
    print("-" * (26 + 14 * len(rcols)))
    for name, m in [("NO-OP", noop), (f"FT {args.model.split('/')[-1]}", ftr)]:
        print(f"{name:<26}" + "".join(
            f"{m[c]:>14.3f}" if isinstance(m[c], float) else f"{m[c]:>14}" for c in rcols))
    print("\nGoalpost check (real — repair_recall is the real test; recovery is "
          "convention-sensitive, report-only):")
    for k, t in [("repair_recall", 0.30), ("repair_prec", 0.70)]:
        ok = "✅" if ftr[k] >= t else "❌"
        print(f"  {ok} {k}: {ftr[k]:.3f} (target ≥{t})")
    print(f"  (report-only) recovery: {ftr['recovery']:.3f}")


if __name__ == "__main__":
    main()
