"""Command-line cleaner: clean a spreadsheet without the web app.

    uv run python -m scrubdata.cli messy.csv -o clean.csv --report report.md --plan plan.json

Writes the cleaned file, and optionally the plain-English report and the machine-readable
cleaning plan (auditable + replayable — the trust contract from PRODUCT.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import apply_plan, mock_plan, profile_dataframe, render_report
from .prompt import serialize_plan


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scrubdata", description="Hands-off data cleaning.")
    ap.add_argument("input", help="CSV or Excel file to clean")
    ap.add_argument("-o", "--output", default=None, help="cleaned CSV (default: <input>.clean.csv)")
    ap.add_argument("--report", default=None, help="write the markdown report here")
    ap.add_argument("--plan", default=None, help="write the cleaning plan JSON here")
    ap.add_argument("--quiet", action="store_true", help="don't print the report")
    args = ap.parse_args(argv)

    src = Path(args.input)
    if not src.exists():
        print(f"No such file: {src}", file=sys.stderr)
        return 1

    raw = _read(src)
    before = profile_dataframe(raw)
    plan = mock_plan(raw, before)
    cleaned, log = apply_plan(raw, plan)
    after = profile_dataframe(cleaned)
    report = render_report(plan, log, before, after)

    out = Path(args.output) if args.output else src.with_suffix(".clean.csv")
    cleaned.to_csv(out, index=False)
    if args.plan:
        Path(args.plan).write_text(serialize_plan(plan), encoding="utf-8")
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")

    if not args.quiet:
        print(report)
    print(f"\n→ cleaned: {out}  ({before['n_rows']}×{before['n_cols']} → "
          f"{after['n_rows']}×{after['n_cols']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
