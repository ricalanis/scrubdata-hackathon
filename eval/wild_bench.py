"""Stage-3 WILD BENCH — "can this system clean real-world tables?" scoreboard.

Every registered wild dataset (no gold required) gets a row built from the SHIPPED
deterministic pipeline (the same `mock_plan` the Space runs):

  * BEHAVIORAL audit — what the product does on the raw table: changes applied,
    columns touched, review flags raised (abstentions), PII columns flagged,
    plan schema validity, silent-edit check (every diff cell attributable to a
    logged op), runtime per 1k rows.
  * INJECT-RECOVERY — seeded errors (typo/ocr/case/whitespace, eval/inject.py)
    injected into the table's OWN content, then cleaned and scored churn-neutral:
    in-domain robustness. (Caveat, disclosed: the raw table plays "clean", so
    pre-existing errors slightly deflate scores — uniform across systems.)

Sources: training/unpaired_sources.json cache + data/wild/ extras (stage-3 hunts).

    uv run python -m eval.wild_bench                    # full scoreboard
    uv run python -m eval.wild_bench --only spotify     # one dataset
Writes eval/results/wild_bench.json and docs/WILD_BENCH.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from scrubdata.executor import apply_plan
from scrubdata.planner import mock_plan

from .inject import inject
from .metrics import is_valid
from .run_real_multi import _cell_only, score

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "real" / "cache"
WILD = ROOT / "data" / "wild"
INJECT_TYPES = ("typo", "ocr", "case", "whitespace")
N_ROWS = 800


def registry() -> list[dict]:
    """All benchmark targets: cached portal tables + stage-3 wild extras."""
    out = []
    src = json.load(open(ROOT / "training" / "unpaired_sources.json"))
    for s in src:
        p = CACHE / f"{s['name']}.csv"
        if p.exists():
            out.append({"name": s["name"], "domain": s["domain"], "path": p})
    if WILD.exists():
        manifest = WILD / "manifest.json"
        extras = json.load(open(manifest)) if manifest.exists() else []
        for s in extras:
            p = WILD / f"{s['name']}.csv"
            if p.exists():
                out.append({"name": s["name"], "domain": s.get("domain", "wild"),
                            "path": p})
    return out


def _load(path: Path) -> pd.DataFrame:
    kw = dict(dtype=str, keep_default_na=False, nrows=N_ROWS, on_bad_lines="skip")
    try:
        df = pd.read_csv(path, encoding_errors="replace", **kw)
    except pd.errors.ParserError:           # ragged quoting etc. — slow tolerant path
        df = pd.read_csv(path, engine="python", **kw)
    return df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]


def behavioral(df: pd.DataFrame) -> dict:
    t0 = time.perf_counter()
    plan = mock_plan(df)
    cleaned, log = apply_plan(df, plan)
    dt = time.perf_counter() - t0
    flags = plan.get("flags", [])
    cells_changed = sum(e.get("cells_changed", 0) for e in log
                        if isinstance(e.get("cells_changed"), int))
    ops_logged = {e.get("op") for e in log}
    pii_flagged = sum(1 for e in log if e.get("op") == "flag_pii") + \
        sum(1 for c in plan.get("columns", []) for o in c.get("operations", [])
            if o.get("op", "").endswith("_pii") and o.get("op") != "flag_pii")
    # silent-edit check: every changed CELL must be attributable to a logged op.
    # apply_plan resets the index after row drops, so attribute COLUMN ops on a
    # drop-free application (row drops are table-scope-logged separately).
    plan_cols_only = _cell_only(plan)
    cleaned2, log2 = apply_plan(df, plan_cols_only)
    changed_cols = {c for c in df.columns if c in cleaned2.columns
                    and not df[c].equals(cleaned2[c])}
    logged_cols = {e.get("column") for e in log2 if e.get("column")}
    for op in plan_cols_only.get("table_operations", []):
        if op.get("op") == "resolve_by_majority":     # logs table-scope w/ columns
            logged_cols.update(op.get("columns", []))
    silent = sorted(changed_cols - logged_cols)
    return {"plan_valid": is_valid(plan), "ops": len(ops_logged),
            "cells_changed": cells_changed, "review_flags": len(flags),
            "pii_protected_or_flagged": pii_flagged,
            "silent_edit_columns": silent, "sec_per_1k_rows": round(dt / max(len(df), 1) * 1000, 2)}


def inject_recovery(df: pd.DataFrame, seed: int = 7) -> dict:
    out = {}
    for et in INJECT_TYPES:
        dirty = inject(df, et, seed)
        if dirty is None:
            out[et] = None
            continue
        cleaned, _ = apply_plan(dirty, _cell_only(mock_plan(dirty)))
        m = score(dirty, df, cleaned)
        out[et] = round(m["f1"], 3)
    vals = [v for v in out.values() if v is not None]
    out["mean"] = round(sum(vals) / len(vals), 3) if vals else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="eval/results/wild_bench.json")
    args = ap.parse_args()
    rows = []
    for spec in registry():
        if args.only and spec["name"] != args.only:
            continue
        try:
            df = _load(spec["path"])
        except Exception as e:  # noqa: BLE001
            print(f"  {spec['name']}: LOAD FAILED {type(e).__name__}")
            continue
        if df.empty or df.shape[1] < 2:
            continue
        b = behavioral(df)
        r = inject_recovery(df)
        row = {"name": spec["name"], "domain": spec["domain"],
               "rows": len(df), "cols": df.shape[1], **b, "inject": r}
        rows.append(row)
        print(f"  {spec['name']:<18} {spec['domain']:<14} valid={b['plan_valid']} "
              f"chg={b['cells_changed']:<6} flags={b['review_flags']} "
              f"pii={b['pii_protected_or_flagged']} silent={len(b['silent_edit_columns'])} "
              f"| recover: {r['mean']}", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    _write_md(rows)
    n_silent = sum(1 for r in rows if r["silent_edit_columns"])
    means = [r["inject"]["mean"] for r in rows if r["inject"]["mean"] is not None]
    print(f"\n{len(rows)} datasets | plan_valid {sum(r['plan_valid'] for r in rows)}/{len(rows)} "
          f"| silent-edit datasets: {n_silent} | mean inject-recovery: "
          f"{sum(means)/len(means):.3f}" if means else "no recovery rows")
    print(f"written to {args.out} and docs/WILD_BENCH.md")


def _write_md(rows: list[dict]) -> None:
    L = ["# Wild Bench — can the shipped system clean real-world tables?", "",
         "Behavioral audit + seeded inject-recovery per dataset (eval/wild_bench.py).",
         "", "| dataset | domain | rows×cols | valid | changes | flags | PII | silent | typo | ocr | case | ws | mean |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["inject"]["mean"] is not None,
                                         x["inject"]["mean"] or 0)):
        i = r["inject"]
        fmt = lambda v: "—" if v is None else f"{v:.2f}"
        L.append(f"| {r['name']} | {r['domain']} | {r['rows']}×{r['cols']} | "
                 f"{'✓' if r['plan_valid'] else '✗'} | {r['cells_changed']} | "
                 f"{r['review_flags']} | {r['pii_protected_or_flagged']} | "
                 f"{len(r['silent_edit_columns'])} | {fmt(i['typo'])} | {fmt(i['ocr'])} | "
                 f"{fmt(i['case'])} | {fmt(i['whitespace'])} | {fmt(i['mean'])} |")
    (ROOT / "docs" / "WILD_BENCH.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
