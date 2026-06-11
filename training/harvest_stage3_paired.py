"""Stage-3 PAIRED harvest -> data/real/<name>/{dirty,clean}.csv (cell-aligned).

Verified sources (stage-3 paired hunter, 2026-06-10):
  * gidcl_imdb — SICS-FRC GIDCL 1M-row imdb pair (verified by diff: ~57.5k error
    cells; license: none stated — research use, do not redistribute). We keep a
    row-aligned SUBSET: every dirty row + a deterministic sample of clean rows.
  * zeroed_billionaire / zeroed_tax100k — WelkinNi/ZeroED injected-error pairs on
    NEW tables (rich categoricals; license none — research use).
  * dgov_* — LUH-DBS Matelda DGov_Typo real data.gov tables with injected typos
    (Apache-2.0). A diverse sample of tables; per-table pairs.

EVAL/TRAIN discipline: assign each new source in eval/generalization.py or the
training mix — never both. Default: these land UNASSIGNED (benchmark-only) until
explicitly placed.

    uv run python -m training.harvest_stage3_paired
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"
UA = {"User-Agent": "scrubdata-research/1.0"}

GIDCL = "https://raw.githubusercontent.com/SICS-Fundamental-Research-Center/GIDCL/main/GEIL_Data/imdb/original"
ZEROED = "https://raw.githubusercontent.com/WelkinNi/ZeroED/main/data"
MATELDA = "https://raw.githubusercontent.com/LUH-DBS/Matelda/main/datasets/DGov_Typo"
DGOV_TABLES = [
    "AH_Provisional_Diabetes_Death_Counts_for_2020",
    "305b_Assessed_Lake_2020",
    "2.10_Budget_Presentation_Award_(summary)",
    "3.09_Census_ACS_Post_Secondary_Education_(detail)",
    "Access_Control",
    # cycle-3 widening: diverse domains/sizes from the 128-table DGov_Typo lake
    "Allegheny_County_Tobacco_Vendors",
    "Grocery_Stores_-_2013",
    "Illinois_Obesity_By_County",
    "Jefferson_County_KY_Post_Offices",
    "LA_County_COVID_Cases",
    "Legislative_Bridge_Names",
    "Louisville_Metro_KY_-_Inspection_Results_Pools",
    "Louisville_Metro_KY_-_Permitted_Hotels_and_Motels",
    "MVA_Vehicle_Sales_Counts_by_Month_for_Calendar_Year_2002_through_August_2022",
    "Median_Household_Income",
    "Medicare_Part_D_Opioid_Prescribing_Rates_-_by_Geography",
    "National_Obesity_By_State_1",
    "Emergency_Operating_Center_Tools",
    "Field_Listings",
    "Health_conditions_among_children_under_age_18__by_selected_characteristics__United_States",
]


def _read(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    try:
        return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False,
                           encoding="latin-1")


def _write(name: str, dirty: pd.DataFrame, clean: pd.DataFrame) -> None:
    n = min(len(dirty), len(clean))
    dirty, clean = dirty.head(n).reset_index(drop=True), clean.head(n).reset_index(drop=True)
    d = REAL / name
    d.mkdir(parents=True, exist_ok=True)
    dirty.to_csv(d / "dirty.csv", index=False)
    clean.to_csv(d / "clean.csv", index=False)
    diff = sum((dirty.iloc[:, j].astype(str) != clean.iloc[:, j].astype(str)).sum()
               for j in range(min(dirty.shape[1], clean.shape[1])))
    print(f"  {name}: {n} rows x {dirty.shape[1]} cols, {diff} diff cells", flush=True)


def gidcl_imdb(max_clean_sample: int = 30000) -> None:
    dirty = _read(f"{GIDCL}/dirty.csv")
    clean = _read(f"{GIDCL}/clean.csv")
    n = min(len(dirty), len(clean))
    dirty, clean = dirty.head(n), clean.head(n)
    neq = (dirty.astype(str).values != clean.astype(str).values).any(axis=1)
    err_idx = dirty.index[neq]
    clean_idx = dirty.index[~neq][:: max(1, (n - len(err_idx)) // max_clean_sample)][:max_clean_sample]
    keep = sorted(set(err_idx) | set(clean_idx))
    _write("gidcl_imdb", dirty.loc[keep], clean.loc[keep])


def zeroed(names=("billionaire", "tax100k")) -> None:
    for nm in names:
        dirty = _read(f"{ZEROED}/{nm}_error-01.csv")
        clean = _read(f"{ZEROED}/{nm}_clean.csv")
        _write(f"zeroed_{nm}", dirty, clean)


def matelda_dgov() -> None:
    import re
    for t in DGOV_TABLES:
        slug = "dgov_" + re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")[:40]
        try:
            q = urllib.parse.quote(t)
            dirty = _read(f"{MATELDA}/{q}/dirty.csv")
            clean = _read(f"{MATELDA}/{q}/clean.csv")
            _write(slug, dirty, clean)
        except Exception as e:  # noqa: BLE001
            print(f"  {slug}: FAILED {type(e).__name__}: {str(e)[:80]}")


import urllib.parse  # noqa: E402


def main() -> None:
    print("stage-3 paired harvest:")
    zeroed()
    matelda_dgov()
    gidcl_imdb()
    print("done.")


if __name__ == "__main__":
    main()
