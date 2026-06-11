"""Stage-3 PAIRED BENCH — shipped-system scorecard over every paired dirty/clean set.

Walks data/real/*/ for cell-aligned pairs and scores the SHIPPED deterministic
pipeline (mock_plan — what the Space runs) with the same churn-neutral metric and
variant-class breakdown as eval/generalization.py. Sources that fed the champion's
training mix are INCLUDED but flagged (seen=True) — transparency over exclusion.

    uv run python -m eval.paired_bench
Writes eval/results/paired_bench.json and docs/PAIRED_BENCH.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan

from .generalization import TRAIN_SOURCES, variant_breakdown
from .run_real_multi import _cell_only, score

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"
N_CAP = 20000          # row cap for very large pairs (gidcl_imdb, tax100k)
SEEN = TRAIN_SOURCES["v6"] | {"fodors_zagats", "cleanml_company", "cleanml_movie"}


def pairs() -> list[Path]:
    return sorted(p for p in REAL.iterdir()
                  if (p / "dirty.csv").exists() and (p / "clean.csv").exists())


def _load(p: Path):
    kw = dict(dtype=str, keep_default_na=False, nrows=N_CAP, on_bad_lines="skip")
    d = pd.read_csv(p / "dirty.csv", encoding_errors="replace", **kw)
    c = pd.read_csv(p / "clean.csv", encoding_errors="replace", **kw)
    n = min(len(d), len(c))
    return d.head(n).reset_index(drop=True), c.head(n).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="eval/results/paired_bench.json")
    args = ap.parse_args()
    rows = []
    for p in pairs():
        name = p.name
        if args.only and name != args.only:
            continue
        try:
            dirty, clean = _load(p)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: LOAD FAILED {type(e).__name__}")
            continue
        t0 = time.perf_counter()
        cleaned, _ = apply_plan(dirty, _cell_only(mock_plan(dirty)))
        m = score(dirty, clean, cleaned)
        b = variant_breakdown(dirty, clean, cleaned)
        vr = b["variant_fixed"] / b["variant_errors"] if b["variant_errors"] else None
        row = {"name": name, "seen_in_training": name in SEEN,
               "rows": len(dirty), "cols": dirty.shape[1],
               "errors": m["_errors"], "variant_errors": b["variant_errors"],
               "f1": round(m["f1"], 3), "precision": round(m["precision"], 3),
               "recall": round(m["recall"], 3), "damage": round(m["damage"], 4),
               "variant_recall": round(vr, 3) if vr is not None else None,
               "sec": round(time.perf_counter() - t0, 1)}
        rows.append(row)
        print(f"  {name:<42} F1={row['f1']:<6} VR={row['variant_recall']} "
              f"dmg={row['damage']} err={row['errors']} "
              f"{'[SEEN]' if row['seen_in_training'] else ''}", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    L = ["# Paired Bench — shipped system on every cell-aligned pair", "",
         "Churn-neutral repairs metric + variant-class recall; `seen` = source fed",
         "the champion's training mix (flagged, not hidden).", "",
         "| dataset | seen | rows×cols | errors | variant | F1 | precision | recall | VR | damage |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["f1"]):
        L.append(f"| {r['name']} | {'✓' if r['seen_in_training'] else ''} | "
                 f"{r['rows']}×{r['cols']} | {r['errors']} | {r['variant_errors']} | "
                 f"{r['f1']} | {r['precision']} | {r['recall']} | "
                 f"{r['variant_recall']} | {r['damage']} |")
    (ROOT / "docs" / "PAIRED_BENCH.md").write_text("\n".join(L) + "\n")
    print(f"{len(rows)} pairs -> {args.out} + docs/PAIRED_BENCH.md")


if __name__ == "__main__":
    main()
