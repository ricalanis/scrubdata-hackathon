"""Layer-2 eval: our pipeline on a REAL dirty/clean dataset (out-of-distribution).

Uses Raha's `hospital` (1000×20, ~2.5% cells are char-substitution typos), row-aligned
dirty/clean. Reports the Raha-style repair protocol — the right metric when the data is
already mostly correct — plus overall recovery.

    uv run eval/run_real.py

Metrics (per system, vs the clean reference):
  recovery      fraction of cells matching clean (tolerant of pure type-coercion)
  repair_recall corrected errors / total errors        (did we FIX the errors?)
  repair_prec   corrected errors / cells we changed     (did we avoid BREAKING good cells?)
  broken        good cells we changed to wrong          (lower is better)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan

from .metrics import _cell_equal

BASE = Path(__file__).resolve().parent.parent / "data" / "real" / "hospital"
URLS = {
    "dirty.csv": "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/hospital/dirty.csv",
    "clean.csv": "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/hospital/clean.csv",
}


def _ensure_data() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for fn, url in URLS.items():
        p = BASE / fn
        if not p.exists():
            urllib.request.urlretrieve(url, p)


def _load():
    d = pd.read_csv(BASE / "dirty.csv", dtype=str, keep_default_na=False)
    c = pd.read_csv(BASE / "clean.csv", dtype=str, keep_default_na=False)
    return d, c


def _score(dirty: pd.DataFrame, clean: pd.DataFrame, out: pd.DataFrame) -> dict:
    """Compare a system output `out` to `clean`, by position, vs the `dirty` input."""
    cols = [c for c in dirty.columns if c in out.columns]
    nrows = min(len(dirty), len(out), len(clean))
    total = errors = fixed = changed = broken = recovered = 0
    for j, col in enumerate(dirty.columns):
        present = col in out.columns
        for i in range(nrows):
            total += 1
            dv, cv = dirty.iat[i, j], clean.iat[i, j]
            ov = out.iloc[i][col] if present else None
            is_err = not _cell_equal(dv, cv)
            is_changed = present and not _cell_equal(ov, dv)
            ok = present and _cell_equal(ov, cv)
            if ok:
                recovered += 1
            if is_err:
                errors += 1
                if ok:
                    fixed += 1
            if is_changed:
                changed += 1
                if not is_err and not ok:   # we changed a good cell into a wrong one
                    broken += 1
    return {
        "recovery": recovered / total,
        "repair_recall": fixed / errors if errors else 0.0,
        "repair_prec": fixed / changed if changed else 0.0,
        "broken": broken,
        "_errors": errors, "_changed": changed, "_fixed": fixed,
    }


def main() -> None:
    _ensure_data()
    dirty, clean = _load()
    noop = _score(dirty, clean, dirty)
    h_plan = mock_plan(dirty)
    cleaned, _ = apply_plan(dirty, h_plan)
    heur = _score(dirty, clean, cleaned)

    print(f"\nLayer-2 real-data eval — Raha hospital ({dirty.shape[0]}×{dirty.shape[1]}, "
          f"{noop['_errors']} error cells)\n")
    cols = ["recovery", "repair_recall", "repair_prec", "broken"]
    print(f"{'system':<22}" + "".join(f"{c:>14}" for c in cols))
    print("-" * (22 + 14 * len(cols)))
    for name, m in [("NO-OP (dirty as-is)", noop), ("HEURISTIC (baseline)", heur)]:
        print(f"{name:<22}" + "".join(f"{m[c]:>14.3f}" if isinstance(m[c], float)
                                       else f"{m[c]:>14}" for c in cols))
    print(f"\nHeuristic changed {heur['_changed']} cells, fixed {heur['_fixed']} errors, "
          f"broke {heur['broken']}.")
    print("Errors here are char-substitution typos — fixable by cluster-canonicalization "
          "(model's job), not by the rule heuristic. The model run plugs in the same way.")


if __name__ == "__main__":
    main()
