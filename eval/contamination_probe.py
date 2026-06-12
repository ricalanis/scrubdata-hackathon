"""Memorization probe (W4.6): can a web-trained model complete benchmark rows verbatim?

Legacy-public benchmarks (hospital et al., GitHub since 2019) sit inside every base
model's training window; a HIGH verbatim-completion rate red-flags memorized gold.
A low rate does not prove absence — the contamination statement stays assumption-based.
Control: a date-stamped post-cutoff wild harvest (expected ~0).

    uv run python -m eval.contamination_probe
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N_ROWS, N_GIVEN, MODEL = 30, 5, "glm-5.1"


def probe(df: pd.DataFrame, name: str) -> dict:
    rng = random.Random(0)
    rows = rng.sample(range(len(df)), min(N_ROWS, len(df)))
    cols = list(df.columns)
    given, asked = cols[:N_GIVEN], cols[N_GIVEN:N_GIVEN + 4]
    hits = total = 0
    for r in rows:
        prompt = (f"This is a row from the well-known public dataset '{name}'. "
                  f"Complete the remaining fields EXACTLY as they appear in the dataset. "
                  f"Known fields: "
                  + "; ".join(f"{c}={df.iloc[r][c]}" for c in given)
                  + ". Respond ONLY with: " + "; ".join(f"{c}=<value>" for c in asked))
        out = subprocess.run(["oll", prompt, "--model", MODEL, "--max-tokens", "200"],
                             capture_output=True, text=True, timeout=120).stdout.lower()
        for c in asked:
            total += 1
            v = str(df.iloc[r][c]).strip().lower()
            if v and v not in ("nan", "") and v in out:
                hits += 1
    return {"table": name, "rows": len(rows), "cells_asked": total,
            "verbatim_hits": hits, "rate": round(hits / max(total, 1), 4)}


def main() -> None:
    hosp = pd.read_csv(ROOT / "data" / "real" / "hospital" / "clean.csv").astype(str)
    wild = pd.read_csv(ROOT / "data" / "wild" / "glassdoor_jobs.csv").astype(str)
    res = {"model": MODEL, "protocol": f"{N_ROWS} rows, {N_GIVEN} given cols, 4 asked cols, exact-substring match",
           "probes": [probe(hosp, "hospital (Raha benchmark)"),
                      probe(wild, "glassdoor_jobs (post-cutoff wild harvest)")]}
    json.dump(res, open(ROOT / "eval" / "results" / "contamination_probe.json", "w"), indent=1)
    print(json.dumps(res["probes"], indent=1))


if __name__ == "__main__":
    main()
