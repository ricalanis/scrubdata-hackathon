"""Stage-2 (D1) harvest: materialize NEW paired dirty/clean sources into data/real/.

Sources (scout-verified 2026-06-10; see memory stage2-goal):
  * ed2_restaurants — BigDaMa/ExampleDrivenErrorDetection Restaurants (28,787 rows,
    565 diff cells, real typo/format variants). `id` column dropped (re-keyed ids are
    not learnable corrections).
  * cleanml_company / cleanml_movie — CleanML inconsistency pairs (raw vs
    inconsistency_clean_raw). Movie is diffed against raha movies_1 and skipped if dup.
  * fodors_zagats / dblp_acm / dblp_scholar — Magellan EM benchmarks turned into
    row-aligned pair TABLES: one row per gold match, dirty side = messy table,
    clean side = canonical table (fodors_zagats: Zagats canonical — normalized
    phones/typo-fixed addresses; DBLP pairs: DBLP canonical — curated). Non-variant
    attribute diffs (e.g. cuisine 'american' vs 'delis') are NOT corrections; the
    derivation gate (_is_variant + never-legit-elsewhere) drops them downstream.

TRAIN/EVAL split (the generalization contract): dblp_scholar is EVAL-ONLY —
registered in eval.generalization.EVAL_SOURCES, never in a training mix.

    uv run python -m training.harvest_stage2
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"

ED2 = "https://raw.githubusercontent.com/BigDaMa/ExampleDrivenErrorDetection/master/datasets"
CLEANML_ZIP = "https://www.dropbox.com/s/nerfrhbrseev928/CleanML-datasets-2020.zip?dl=1"
EM_BASE = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured"
# name -> (zip dir, zip file stem, canonical side)
EM_SETS = {"fodors_zagats": ("Fodors-Zagats", "fodors_zagat", "B"),
           "dblp_acm": ("DBLP-ACM", "dblp_acm", "A"),
           "dblp_scholar": ("DBLP-GoogleScholar", "dblp_scholar", "A")}


def _read_csv(src, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(src, dtype=str, keep_default_na=False, **kw)
    except UnicodeDecodeError:
        if hasattr(src, "seek"):
            src.seek(0)
        return pd.read_csv(src, dtype=str, keep_default_na=False,
                           encoding="latin-1", **kw)


def _write_pair(name: str, dirty: pd.DataFrame, clean: pd.DataFrame) -> None:
    d = REAL / name
    d.mkdir(parents=True, exist_ok=True)
    dirty.to_csv(d / "dirty.csv", index=False)
    clean.to_csv(d / "clean.csv", index=False)
    diff = sum((dirty.iloc[:, j] != clean.iloc[:, j]).sum()
               for j in range(min(dirty.shape[1], clean.shape[1])))
    print(f"  {name}: {dirty.shape[0]} rows x {dirty.shape[1]} cols, "
          f"{diff} raw diff cells -> {d}")


def harvest_ed2_restaurants() -> None:
    frames = []
    for kind in ("dirty", "clean"):
        with urllib.request.urlopen(f"{ED2}/Restaurants_{kind}.csv") as r:
            frames.append(_read_csv(io.BytesIO(r.read())))
    dirty, clean = frames
    for df in (dirty, clean):                       # re-keyed ids aren't corrections
        for c in list(df.columns):
            if c.lower() == "id":
                df.drop(columns=[c], inplace=True)
    _write_pair("ed2_restaurants", dirty, clean)


def harvest_cleanml(members=("Company", "Movie")) -> None:
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "cleanml-2020.zip"
    if not tmp.exists():
        print("  downloading CleanML zip (large, one-time)...")
        urllib.request.urlretrieve(CLEANML_ZIP, tmp)
    zf = zipfile.ZipFile(tmp)
    names = zf.namelist()
    for ds in members:
        raw = next((n for n in names if n.endswith(f"{ds}/raw/raw.csv")), None)
        cln = next((n for n in names
                    if n.endswith(f"{ds}/raw/inconsistency_clean_raw.csv")), None)
        if not raw or not cln:
            print(f"  cleanml_{ds.lower()}: pair not found in zip — skipped")
            continue
        dirty = _read_csv(zf.open(raw))
        clean = _read_csv(zf.open(cln))
        if ds == "Movie":                            # dup check vs raha movies_1
            m1 = REAL / "movies_1" / "dirty.csv"
            if m1.exists() and abs(len(_read_csv(m1)) - len(dirty)) < 50:
                print("  cleanml_movie: shape matches raha movies_1 — SKIPPED as dup")
                continue
        n = min(len(dirty), len(clean))
        _write_pair(f"cleanml_{ds.lower()}", dirty.head(n).reset_index(drop=True),
                    clean.head(n).reset_index(drop=True))


def harvest_em() -> None:
    from training.real_data import _is_variant

    for name, (zdir, stem, canon_side) in EM_SETS.items():
        with urllib.request.urlopen(f"{EM_BASE}/{zdir}/{stem}_raw_data.zip") as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
        A = _read_csv(zf.open("tableA.csv"))
        B = _read_csv(zf.open("tableB.csv"))
        M = _read_csv(zf.open("matches.csv"))
        a = A.set_index(A.columns[0])
        b = B.set_index(B.columns[0])
        cols = [c for c in a.columns if c in b.columns]
        dirty_rows, clean_rows, seen = [], [], set()
        masked = 0
        for _, m in M.iterrows():
            ia, ib = str(m.iloc[0]), str(m.iloc[1])
            if ia not in a.index or ib not in b.index or ia in seen:
                continue
            seen.add(ia)
            ra = a.loc[ia] if not isinstance(a.loc[ia], pd.DataFrame) else a.loc[ia].iloc[0]
            rb = b.loc[ib] if not isinstance(b.loc[ib], pd.DataFrame) else b.loc[ib].iloc[0]
            canon, messy = (ra, rb) if canon_side == "A" else (rb, ra)
            drow, crow = [], []
            for c in cols:
                dv, cv = str(messy[c]), str(canon[c])
                # VARIANT MASK: matched records may legitimately DISAGREE on an
                # attribute (cuisine classification, author-list format) — that is
                # not an error, and neither side is "correct". Only surface-variant
                # diffs are kept as (dirty, clean) corrections; otherwise the messy
                # value is accepted as truth on both sides.
                if dv != cv and not _is_variant(dv, cv):
                    cv = dv
                    masked += 1
                drow.append(dv)
                crow.append(cv)
            dirty_rows.append(drow)
            clean_rows.append(crow)
        print(f"  {name}: masked {masked} non-variant attribute disagreements")
        _write_pair(name, pd.DataFrame(dirty_rows, columns=cols),
                    pd.DataFrame(clean_rows, columns=cols))


def main() -> None:
    print("stage-2 harvest:")
    harvest_ed2_restaurants()
    harvest_em()
    harvest_cleanml()
    print("done. TRAIN candidates: ed2_restaurants, fodors_zagats, dblp_acm, "
          "cleanml_company[, cleanml_movie]. EVAL-ONLY: dblp_scholar.")


if __name__ == "__main__":
    main()
