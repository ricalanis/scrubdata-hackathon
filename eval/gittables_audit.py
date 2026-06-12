"""N=250 GitTables audit — the at-scale trust + repair board.

250 real GitHub tables (LUH-DBS Matelda GitTables-subsets, Apache-2.0; injected
typos on real heterogeneous tables) scored end-to-end with the shipped pipeline:
schema validity, SILENT-EDIT attribution (the trust contract at scale), and the
churn-neutral repair metric. No inject-recovery here (these pairs carry their own
errors). Summary feeds docs/GITTABLES_AUDIT.md.

    uv run python -m eval.gittables_audit
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan

from .metrics import is_valid
from .run_real_multi import _cell_only, score
from .wild_bench import behavioral

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "gittables250"
N_CAP = 3000


def _load(p: Path):
    kw = dict(dtype=str, keep_default_na=False, nrows=N_CAP, on_bad_lines="skip")
    try:
        return pd.read_csv(p, encoding_errors="replace", **kw)
    except Exception:  # noqa: BLE001
        return pd.read_csv(p, engine="python", **kw)


def main() -> None:
    slugs = sorted({p.name.split("_")[0] for p in DIR.glob("t*_dirty.csv")})
    rows, failures = [], []
    t0 = time.perf_counter()
    for slug in slugs:
        try:
            dirty = _load(DIR / f"{slug}_dirty.csv")
            clean = _load(DIR / f"{slug}_clean.csv")
            n = min(len(dirty), len(clean))
            if n < 3 or dirty.shape[1] < 2:
                continue
            dirty, clean = dirty.head(n), clean.head(n)
            b = behavioral(dirty)
            plan = _cell_only(mock_plan(dirty))
            cleaned, _ = apply_plan(dirty, plan)
            m = score(dirty, clean, cleaned)
            rows.append({"table": slug, "rows": n, "cols": dirty.shape[1],
                         "plan_valid": b["plan_valid"],
                         "silent_edit_columns": len(b["silent_edit_columns"]),
                         "errors": m["_errors"], "f1": round(m["f1"], 3),
                         "damage": round(m["damage"], 4)})
        except Exception as e:  # noqa: BLE001
            failures.append(f"{slug}: {type(e).__name__}")
    dt = time.perf_counter() - t0

    n = len(rows)
    valid = sum(r["plan_valid"] for r in rows)
    silent = sum(1 for r in rows if r["silent_edit_columns"])
    scored = [r for r in rows if r["errors"] > 0]
    f1s = [r["f1"] for r in scored]
    dmgs = [r["damage"] for r in rows]
    summary = {
        "tables_audited": n, "pipeline_failures": len(failures),
        "plan_valid": valid, "tables_with_silent_edits": silent,
        "tables_with_errors": len(scored),
        "macro_f1_on_errored": round(sum(f1s) / len(f1s), 3) if f1s else None,
        "macro_damage": round(sum(dmgs) / len(dmgs), 4),
        "zero_damage_tables": sum(1 for d in dmgs if d == 0),
        "seconds": round(dt, 1),
    }
    json.dump({"summary": summary, "rows": rows, "failures": failures},
              open(ROOT / "eval" / "results" / "gittables_audit.json", "w"), indent=1)
    L = ["# GitTables N=250 audit — trust contract at scale", "",
         f"Shipped pipeline over {n} real GitHub tables (Matelda GitTables-subsets,",
         "Apache-2.0; injected typos on real heterogeneous tables).", "",
         "| metric | value |", "|---|---|"]
    for k, v in summary.items():
        L.append(f"| {k} | {v} |")
    (ROOT / "docs" / "GITTABLES_AUDIT.md").write_text("\n".join(L) + "\n")
    print(json.dumps(summary, indent=1))
    if failures:
        print("failures:", failures[:8])


if __name__ == "__main__":
    main()
