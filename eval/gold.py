"""Frozen held-out gold eval set (committed to eval/gold.jsonl).

A FIXED test set so every fine-tune iteration (and generator change) is scored on the
same examples — v1 vs v2 stay comparable. Regenerate intentionally with `build_gold`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from training.generate import make_example

from . import metrics

GOLD_PATH = Path(__file__).resolve().parent / "gold.jsonl"


def build_gold(n: int = 300, seed: int = 4242, path: Path = GOLD_PATH) -> list[dict]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        ex = make_example(rng)
        if metrics.recovery(ex["clean_df"], ex["dirty_df"], ex["plan"]) >= 0.999:
            out.append(ex)
    with Path(path).open("w", encoding="utf-8") as f:
        for ex in out:
            clean = ex["clean_df"].where(pd.notna(ex["clean_df"]), None)
            f.write(json.dumps({
                "dirty": ex["dirty_df"].to_dict("records"),
                "clean": clean.to_dict("records"),
                "dirty_cols": list(ex["dirty_df"].columns),
                "clean_cols": list(ex["clean_df"].columns),
                "plan": ex["plan"],
            }, ensure_ascii=False, default=str) + "\n")
    return out


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return build_gold(path=p)
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        dirty = (pd.DataFrame(d["dirty"])[d["dirty_cols"]] if d["dirty"]
                 else pd.DataFrame(columns=d["dirty_cols"]))
        clean = (pd.DataFrame(d["clean"])[d["clean_cols"]] if d["clean"]
                 else pd.DataFrame(columns=d["clean_cols"]))
        out.append({"dirty_df": dirty, "clean_df": clean, "plan": d["plan"]})
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    g = build_gold(args.n, args.seed)
    print(f"Wrote {len(g)} frozen gold examples to {GOLD_PATH}")
