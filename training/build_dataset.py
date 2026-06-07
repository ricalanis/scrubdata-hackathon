"""Build a verified SFT dataset of (dirty profile -> cleaning plan) pairs.

For each synthetic example we run scrubdata.executor(dirty, ground_truth_plan)
and compare to the clean reference. Only PERFECTLY recovered examples are kept —
this is the quality gate that makes the synthetic data trustworthy.

Usage:
    uv run training/build_dataset.py --n 2000 --out data/train.jsonl --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from scrubdata.executor import apply_plan
from scrubdata.prompt import build_chat_example
from scrubdata.profiler import profile_dataframe

from .generate import make_example
import random


def _cell_equal(a, b) -> bool:
    a_missing = a is None or (isinstance(a, float) and math.isnan(a)) or pd.isna(a)
    b_missing = b is None or (isinstance(b, float) and math.isnan(b)) or pd.isna(b)
    if a_missing or b_missing:
        return a_missing and b_missing
    # numeric tolerance
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return str(a) == str(b)


def verify(clean_df: pd.DataFrame, dirty_df: pd.DataFrame, plan: dict) -> bool:
    cleaned, _ = apply_plan(dirty_df, plan)
    if list(cleaned.columns) != list(clean_df.columns):
        return False
    if len(cleaned) != len(clean_df):
        return False
    for col in clean_df.columns:
        for a, b in zip(clean_df[col].tolist(), cleaned[col].tolist()):
            if not _cell_equal(a, b):
                return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="target verified examples")
    ap.add_argument("--out", type=str, default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts-factor", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept, attempts = 0, 0
    max_attempts = args.n * args.max_attempts_factor
    with out_path.open("w", encoding="utf-8") as f:
        while kept < args.n and attempts < max_attempts:
            attempts += 1
            ex = make_example(rng)
            if not verify(ex["clean_df"], ex["dirty_df"], ex["plan"]):
                continue
            record = build_chat_example(ex["profile"], ex["dirty_df"], ex["plan"])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    rate = kept / attempts if attempts else 0.0
    print(f"Wrote {kept} verified examples to {out_path} "
          f"({attempts} attempts, {rate:.0%} verified).")


if __name__ == "__main__":
    main()
