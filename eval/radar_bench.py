"""RADAR mini-board: shipped pipeline vs RADAR's perturbed tables, by artifact type.

RADAR (kenqgu/RADAR, CC-BY-4.0; table-QA under data artifacts) ships, per example,
the perturbed table AND the gold recovery transform (overwrite_cells + drop_rows) —
so dirty/clean pairs are derivable exactly. We score the shipped deterministic
pipeline per table with the churn-neutral metric and aggregate by artifact_type:
which artifact classes the system repairs, where it abstains, what it damages.

    uv run python -m eval.radar_bench --n 150
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan

from .run_real_multi import _cell_only, score

ROOT = Path(__file__).resolve().parent.parent


def derive_pair(ex):
    t = ex["table"]
    rows = [list(r) for r in t["rows"]]              # numpy object-array -> lists
    dirty = pd.DataFrame(rows, columns=list(t["headers"])).astype(str)
    spec = ex.get("recovered_tables_transform_spec") or {}
    clean = dirty.copy()
    oc = spec.get("overwrite_cells")
    groups = list(oc) if oc is not None else []
    cells = list(groups[0]) if len(groups) else []   # first consistent recovery group
    for cell in cells:
        r, c, v = int(cell["row"]), cell["col"], cell["new_value"]
        if c in clean.columns and 0 <= r < len(clean):
            clean.iat[r, clean.columns.get_loc(c)] = str(v)
    dr = spec.get("drop_rows")
    drops = sorted({int(r) for grp in (list(dr) if dr is not None else [])
                    for r in list(grp)})
    if drops:
        keep = [i for i in range(len(clean)) if i not in drops]
        dirty = dirty.iloc[keep].reset_index(drop=True)   # row-drop class: align both
        clean = clean.iloc[keep].reset_index(drop=True)
    return dirty, clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    ds = pd.read_parquet("hf://datasets/kenqgu/RADAR/radar/test-00000-of-00001.parquet")
    by_type = collections.defaultdict(list)
    seen = collections.Counter()
    for _, ex in ds.iterrows():
        at = ex["artifact_type"]
        if seen[at] >= max(8, args.n // ds["artifact_type"].nunique()):
            continue
        seen[at] += 1
        try:
            dirty, clean = derive_pair(ex)
            if dirty.empty or dirty.shape != clean.shape:
                continue
            cleaned, _ = apply_plan(dirty, _cell_only(mock_plan(dirty)))
            m = score(dirty, clean, cleaned)
            if m["_errors"] > 0:
                by_type[at].append((m["f1"], m["damage"]))
        except Exception:  # noqa: BLE001
            continue
        if sum(seen.values()) >= args.n:
            break
    rows = []
    for at, ms in sorted(by_type.items()):
        f1 = sum(f for f, _ in ms) / len(ms)
        dmg = sum(d for _, d in ms) / len(ms)
        rows.append({"artifact_type": at, "tables": len(ms),
                     "macro_f1": round(f1, 3), "macro_damage": round(dmg, 4)})
        print(f"  {at:<28} n={len(ms):<4} F1={f1:.3f} dmg={dmg:.4f}")
    json.dump(rows, open(ROOT / "eval" / "results" / "radar_bench.json", "w"), indent=1)
    print(f"-> eval/results/radar_bench.json ({sum(seen.values())} examples scanned)")


if __name__ == "__main__":
    main()
