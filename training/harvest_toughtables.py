"""ToughTables (2T_WD, SemTab; CC-BY-4.0) -> gold-anchored misspelling pairs.

2T tables contain row-GROUPS of the same entity whose surfaces include deliberate
misspellings/variants; the CEA ground truth maps (table, row, col) -> Wikidata QID.
Within a (table, column, QID) group the CANONICAL surface is the modal one and every
other variant surface is a (dirty -> canonical) correction — derivable with NO
external lookup, anchored by gold (not blind frequency clustering).

Outputs:
  * data/real/tt_<table>/{dirty,clean}.csv — row-aligned pair tables for the largest
    tables (paired bench entries; variant cells replaced by the group canonical).
  * data/toughtables_aliases.jsonl — {canonical, aliases} vocabulary across the corpus
    (training generator material; only _is_variant aliases kept).

    uv run python -m training.harvest_toughtables --zip /tmp/2t_wd.zip
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"


def _is_variant(a: str, b: str) -> bool:
    from training.real_data import _is_variant as iv
    return iv(a, b)


def load_cea(zf: zipfile.ZipFile) -> dict:
    """(table, row, col) -> primary QID."""
    gt = {}
    with zf.open("2T_WD/gt/CEA_2T_WD_gt.csv") as fh:
        for rec in csv.reader(io.TextIOWrapper(fh, encoding="utf-8")):
            tab, row, col, uris = rec[0], int(rec[1]), int(rec[2]), rec[3]
            gt[(tab, row, col)] = uris.split(" ")[0]
    return gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="/tmp/2t_wd.zip")
    ap.add_argument("--top-tables", type=int, default=8,
                    help="emit the N largest tables as paired-bench pairs")
    ap.add_argument("--vocab-out", default="data/toughtables_aliases.jsonl")
    ap.add_argument("--exclude-tables", default="",
                    help="comma list of table stems EXCLUDED from the vocabulary "
                         "(contamination guard: benchmark tables must not feed the "
                         "reference used to clean them)")
    args = ap.parse_args()
    excluded = {t.strip().lower() for t in args.exclude_tables.split(",") if t.strip()}
    zf = zipfile.ZipFile(args.zip)
    gt = load_cea(zf)
    print(f"CEA gold: {len(gt)} annotated cells")

    # global alias vocabulary: QID -> surface counter (across all tables)
    qid_surfaces: dict[str, Counter] = defaultdict(Counter)
    tables = [n for n in zf.namelist()
              if n.startswith("2T_WD/tables/") and n.endswith(".csv")]
    table_dfs = {}
    for tn in tables:
        tab = Path(tn).stem
        try:
            df = pd.read_csv(zf.open(tn), dtype=str, keep_default_na=False)
        except Exception:  # noqa: BLE001
            continue
        table_dfs[tab] = df
        for (t, row, col), qid in []:
            pass
    # collect surfaces (row in CEA is 1-based over data rows)
    for (tab, row, col), qid in gt.items():
        if tab.lower() in excluded:
            continue
        df = table_dfs.get(tab)
        if df is None or row - 1 >= len(df) or col >= df.shape[1]:
            continue
        s = str(df.iat[row - 1, col]).strip()
        if s:
            qid_surfaces[qid][s] += 1

    # vocabulary: canonical = modal surface; aliases = variant minority surfaces
    n_alias = 0
    with open(ROOT / args.vocab_out, "w", encoding="utf-8") as out:
        for qid, ctr in qid_surfaces.items():
            if len(ctr) < 2:
                continue
            canonical, _ = ctr.most_common(1)[0]
            aliases = [s for s, _ in ctr.most_common()
                       if s != canonical and _is_variant(s, canonical)]
            if aliases:
                out.write(json.dumps({"canonical": canonical, "aliases": aliases},
                                     ensure_ascii=False) + "\n")
                n_alias += len(aliases)
    print(f"vocab: {n_alias} variant aliases -> {args.vocab_out}")

    # paired-bench pairs for the largest tables
    sizes = sorted(((len(df), tab) for tab, df in table_dfs.items()), reverse=True)
    made = 0
    for _, tab in sizes:
        if made >= args.top_tables:
            break
        df = table_dfs[tab]
        clean = df.copy()
        fixed = 0
        for (t, row, col), qid in gt.items():
            if t != tab or row - 1 >= len(df) or col >= df.shape[1]:
                continue
            s = str(df.iat[row - 1, col]).strip()
            ctr = qid_surfaces[qid]
            canonical, _ = ctr.most_common(1)[0]
            if s and s != canonical and _is_variant(s, canonical):
                clean.iat[row - 1, col] = canonical
                fixed += 1
        if fixed < 30:
            continue
        d = REAL / f"tt_{tab.lower()}"
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / "dirty.csv", index=False)
        clean.to_csv(d / "clean.csv", index=False)
        print(f"  tt_{tab.lower()}: {len(df)} rows, {fixed} gold-variant corrections")
        made += 1


if __name__ == "__main__":
    main()
