"""Real paired dirty/clean datasets -> self-verified SFT training examples.

The v4 model aced synthetic data (canon_f1 0.90) but scored 0 on the real Raha
hospital table because it had never trained on real high-cardinality messy data.

KEY INSIGHT: real *paired* (dirty, clean) datasets let us DERIVE a self-verified
ground-truth plan by aligning cells. Wherever dirty[i,j] != clean[i,j], the pair
dirty-value -> clean-value is a canonicalize mapping (or a deterministic format
fix). Executing the derived plan recovers clean -> the example is self-verified
with the SAME executor-recovery gate used for synthetic data.

This module:
  1) fetches the shortlisted PAIRED Raha datasets (disk-aware: small ones cached
     under data/real/ which is gitignored; bulky `tax` is sampled then deleted);
  2) derive_plan(dirty_df, clean_df) -> plan dict (cell-align -> canonicalize +
     obvious format/dup fixes) such that apply_plan(dirty, plan) recovers clean;
  3) emits chat-format ('messages') examples via build_chat_example using the
     AGGREGATED profile of the DIRTY table, keeping ONLY examples whose derived
     plan recovers clean above a threshold (self-verified).

Run:
    uv run training/real_data.py
    uv run training/real_data.py --datasets hospital beers rayyan flights
    uv run training/real_data.py --include-tax        # fetch+sample+delete tax

Does NOT push to HF and does NOT train.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import urllib.request
from pathlib import Path

import pandas as pd

from scrubdata.executor import apply_plan
from scrubdata.profiler import profile_dataframe
from scrubdata.prompt import build_chat_example

ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = ROOT / "data" / "real"
RAW_BASE = "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets"

# Paired datasets. `keep` controls disk policy (HARD constraint: ~5GB free).
# Small tables are cached; `tax` is fetched, sampled, then the raw CSV is deleted.
DATASETS = {
    "hospital": {"keep": True, "sample": None},   # already cached (600K)
    "beers":    {"keep": True, "sample": None},    # ~250K
    "rayyan":   {"keep": True, "sample": None},    # ~150K
    "flights":  {"keep": True, "sample": None},    # ~250K
    "tax":      {"keep": False, "sample": 4000},   # ~30MB raw -> sample then DELETE
}


# --------------------------------------------------------------------------- #
# fetch (disk-aware)
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)


def fetch_pair(name: str, keep_raw: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch (dirty, clean) for a Raha dataset. Bulky raw files are deleted after
    load when keep_raw is False (the small derived JSONL is the only persisted output)."""
    cfg = DATASETS[name]
    base = REAL_DIR / name
    dirty_p = base / "dirty.csv"
    clean_p = base / "clean.csv"
    _download(f"{RAW_BASE}/{name}/dirty.csv", dirty_p)
    _download(f"{RAW_BASE}/{name}/clean.csv", clean_p)

    dirty = pd.read_csv(dirty_p, dtype=str, keep_default_na=False)
    clean = pd.read_csv(clean_p, dtype=str, keep_default_na=False)

    sample_n = cfg.get("sample")
    if sample_n and len(dirty) > sample_n:
        # Row-aligned sampling: take the first N rows of BOTH (positional align).
        dirty = dirty.head(sample_n).reset_index(drop=True)
        clean = clean.head(sample_n).reset_index(drop=True)

    if not (keep_raw and cfg["keep"]):
        # Delete the (possibly bulky) raw CSVs; we already loaded them in memory.
        for p in (dirty_p, clean_p):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        try:
            base.rmdir()
        except OSError:
            pass
    return dirty, clean


# --------------------------------------------------------------------------- #
# cell equality (reused contract with build_dataset._cell_equal)
# --------------------------------------------------------------------------- #
def _cell_equal(a, b) -> bool:
    a_missing = a is None or (isinstance(a, float) and math.isnan(a)) or pd.isna(a)
    b_missing = b is None or (isinstance(b, float) and math.isnan(b)) or pd.isna(b)
    if a_missing or b_missing:
        return a_missing and b_missing
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return str(a) == str(b)


# --------------------------------------------------------------------------- #
# derive a plan from a (dirty, clean) pair
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def _is_variant(dirty: str, clean: str) -> bool:
    """True if `dirty` is a SURFACE VARIANT (typo / casing / punctuation / minor
    abbreviation) of `clean` — i.e. a learnable canonicalization, not a different
    valid value. '9:45'->'9:55' (distinct valid times) is rejected; 'birminghxm'->
    'birmingham' and 'WON'->'Won' are accepted."""
    nd, nc = _norm(dirty), _norm(clean)
    if not nd or not nc:
        return False
    if nd == nc:                       # casing / punctuation only
        return True
    return difflib.SequenceMatcher(None, nd, nc).ratio() >= 0.72


def _column_diff_pairs(dirty_col, clean_col) -> tuple[dict, bool]:
    """Collect {dirty_raw_stripped -> clean_value} for rows that differ, keeping ONLY
    genuine canonicalizations. A pair is kept iff the dirty surface (a) is never a
    CORRECT value elsewhere in the column (else mapping it would corrupt legit rows),
    and (b) is a string VARIANT of its clean target. Returns (mapping, ambiguous);
    rejected/ambiguous pairs set ambiguous=True so they surface as flags."""
    correct = {str(c).strip() for d, c in zip(dirty_col, clean_col)
               if _cell_equal(d, c) and not _is_missing(c)}
    mapping: dict[str, str] = {}
    ambiguous = False
    for dv, cv in zip(dirty_col, clean_col):
        if _cell_equal(dv, cv):
            continue
        if _is_missing(dv) or _is_missing(cv):
            ambiguous = True            # missing source/target: not a canonicalization
            continue
        key = str(dv).strip()
        clean_val = str(cv)
        if key in correct or not _is_variant(key, clean_val):
            ambiguous = True            # legit-elsewhere or arbitrary correction -> skip
            continue
        if key in mapping and mapping[key] != clean_val:
            ambiguous = True
        else:
            mapping[key] = clean_val
    return mapping, ambiguous


def derive_plan(dirty_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """Derive a self-verifying cleaning plan that maps dirty -> clean.

    Columns are aligned POSITIONALLY (Raha hospital/beers rename headers between
    dirty and clean, e.g. provider_number -> ProviderNumber), so we diff by column
    index and emit the plan under the DIRTY column name (what the executor sees).

    Method per column: collect the set of differing (dirty_raw -> clean) pairs and
    emit a canonicalize_categories op with that mapping. The executor does
    mapping.get(str(v).strip(), v), so every changed cell is recovered by
    construction and unchanged cells pass through -> recovery is exact whenever the
    mapping is unambiguous. Ambiguous columns (same dirty raw -> two cleans, or a
    missing dirty source) are emitted as flags so they don't break recovery.

    Table ops: drop_exact_duplicates when clean has fewer rows that are exact dups.
    (Raha tables are row-aligned 1:1, so this is usually a no-op.)
    """
    n = min(len(dirty_df), len(clean_df))
    d = dirty_df.head(n).reset_index(drop=True)
    c = clean_df.head(n).reset_index(drop=True)

    profile = profile_dataframe(d)
    sem_by_idx = {i: profile["columns"][i]["detected_semantic_type"]
                  for i in range(len(profile["columns"]))}
    issues_by_idx = {i: profile["columns"][i]["issues"]
                     for i in range(len(profile["columns"]))}

    columns_plan = []
    flags = []
    n_cols = min(d.shape[1], c.shape[1])
    for j in range(n_cols):
        dirty_name = str(d.columns[j])
        dcol = d.iloc[:, j].tolist()
        ccol = c.iloc[:, j].tolist()
        mapping, ambiguous = _column_diff_pairs(dcol, ccol)

        operations = []
        if mapping:
            operations.append({
                "op": "canonicalize_categories",
                "mapping": mapping,
                "rationale": (
                    f"{len(mapping)} real variant/typo value(s) mapped to their "
                    "canonical form observed in the clean reference."
                ),
            })
        col_record = {
            "name": dirty_name,
            "detected_semantic_type": sem_by_idx.get(j, "unknown"),
            "issues": issues_by_idx.get(j, []),
            "operations": operations,
        }
        columns_plan.append(col_record)

        if ambiguous:
            flags.append({
                "column": dirty_name,
                "issue": "ambiguous_or_missing_source_values",
                "action": "flag_only",
                "rationale": "Some dirty values map to multiple cleans or are "
                             "missing in the source; left for manual review.",
            })

    table_operations = []
    if len(clean_df) < len(dirty_df):
        # Did the missing rows correspond to exact duplicates in dirty?
        if int(dirty_df.duplicated().sum()) >= (len(dirty_df) - len(clean_df)):
            table_operations.append({
                "op": "drop_exact_duplicates",
                "rationale": "Clean reference has the exact-duplicate rows removed.",
            })

    n_map_cols = sum(1 for col in columns_plan if col["operations"])
    return {
        "dataset_summary": (
            f"Real paired dirty/clean table: {n} rows x {n_cols} columns. Derived "
            f"{n_map_cols} canonicalization mapping(s) from cell-level dirty->clean "
            "alignment (real high-cardinality typos/variants)."
        ),
        "table_operations": table_operations,
        "columns": columns_plan,
        "flags": flags,
    }


# --------------------------------------------------------------------------- #
# self-verification: cell recovery of derived plan
# --------------------------------------------------------------------------- #
def recovery_score(dirty_df: pd.DataFrame, clean_df: pd.DataFrame, plan: dict) -> float:
    """Fraction of cells (positional) where apply_plan(dirty, plan) matches clean."""
    cleaned, _ = apply_plan(dirty_df, plan)
    n = min(len(cleaned), len(clean_df))
    n_cols = min(cleaned.shape[1], clean_df.shape[1])
    if n == 0 or n_cols == 0:
        return 0.0
    total = ok = 0
    for j in range(n_cols):
        out_col = cleaned.iloc[:, j].tolist()
        ref_col = clean_df.iloc[:, j].tolist()
        for i in range(n):
            total += 1
            if _cell_equal(out_col[i], ref_col[i]):
                ok += 1
    return ok / total if total else 0.0


def max_categorical_cardinality(plan: dict) -> int:
    """Largest canonicalize mapping (distinct variant count) in the plan."""
    best = 0
    for col in plan.get("columns", []):
        for op in col.get("operations", []):
            if op["op"] == "canonicalize_categories":
                best = max(best, len(op.get("mapping", {})))
    return best


def _sample_mapping(plan: dict, k: int = 6) -> tuple[str, dict]:
    """Pick the column with the largest mapping and return a small sample of it."""
    best_col, best_map = None, {}
    for col in plan.get("columns", []):
        for op in col.get("operations", []):
            if op["op"] == "canonicalize_categories":
                m = op.get("mapping", {})
                if len(m) > len(best_map):
                    best_col, best_map = col["name"], m
    sample = dict(list(best_map.items())[:k]) if best_map else {}
    return best_col or "", sample


# --------------------------------------------------------------------------- #
# learnable-column selection + subsampling into many small real tables
# --------------------------------------------------------------------------- #
def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) \
        or str(v).strip() == ""


def canonicalizable_columns(dirty_df: pd.DataFrame, clean_df: pd.DataFrame,
                            min_nonmissing: int = 12) -> list[int]:
    """Column indices where canonicalization is a LEARNABLE skill: the clean values
    repeat (a small canonical set) AND the dirty->clean corrections CLUSTER onto
    those canonicals (typos/variants), not arbitrary per-cell fixes (flight times,
    IDs, ZIPs). Those arbitrary columns are memorization noise the model can't
    generalize, so we drop them."""
    n = min(len(dirty_df), len(clean_df))
    out = []
    for j in range(min(dirty_df.shape[1], clean_df.shape[1])):
        dcol = dirty_df.iloc[:n, j].tolist()
        ccol = clean_df.iloc[:n, j].tolist()
        clean_vals = [str(c) for c in ccol if not _is_missing(c)]
        if len(clean_vals) < min_nonmissing:
            continue
        # (1) clean column is categorical: values repeat (low distinct ratio).
        if len(set(clean_vals)) / len(clean_vals) > 0.5:
            continue
        # (2) it yields >=2 GENUINE canonicalizations (variant typos of a canonical
        # that isn't a legit value elsewhere) -- this is the learnable signal and it
        # rejects arbitrary value-correction columns (flight times, IDs).
        mapping, _ = _column_diff_pairs(dcol, ccol)
        if len(mapping) >= 2:
            out.append(j)
    return out


def iter_examples(dirty_df, clean_df, rng, n_examples: int, *,
                  threshold: float = 0.97, min_rows: int = 20, max_rows: int = 90,
                  min_cols: int = 2, max_cols: int = 5):
    """Yield (record, plan, recovery) for many small REAL sub-tables drawn from a
    paired dataset, using only learnable canonicalizable columns. Each sub-table is
    profiled (aggregated value_counts) and gets a derived self-verified plan."""
    cols = canonicalizable_columns(dirty_df, clean_df)
    if not cols:
        return
    n = min(len(dirty_df), len(clean_df))
    tries = 0
    made = 0
    while made < n_examples and tries < n_examples * 6:
        tries += 1
        k = rng.randint(min_rows, min(max_rows, n))
        start = rng.randint(0, max(0, n - k))
        hi = min(max_cols, len(cols))
        kc = rng.randint(min(min_cols, hi), hi)
        chosen = sorted(rng.sample(cols, kc))
        d_sub = dirty_df.iloc[start:start + k, chosen].reset_index(drop=True)
        c_sub = clean_df.iloc[start:start + k, chosen].reset_index(drop=True)
        plan = derive_plan(d_sub, c_sub)
        if max_categorical_cardinality(plan) < 1:      # no errors in this window
            continue
        score = recovery_score(d_sub, c_sub, plan)
        if score < threshold:
            continue
        profile = profile_dataframe(d_sub)
        yield build_chat_example(profile, d_sub, plan), plan, score
        made += 1


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def process_dataset(name: str, keep_raw: bool, threshold: float) -> dict | None:
    dirty, clean = fetch_pair(name, keep_raw=keep_raw)
    plan = derive_plan(dirty, clean)

    n = min(len(dirty), len(clean))
    d = dirty.head(n).reset_index(drop=True)
    c = clean.head(n).reset_index(drop=True)

    score = recovery_score(d, c, plan)
    n_err = sum(
        1
        for j in range(min(d.shape[1], c.shape[1]))
        for a, b in zip(d.iloc[:, j].tolist(), c.iloc[:, j].tolist())
        if not _cell_equal(a, b)
    )
    profile = profile_dataframe(d)
    record = build_chat_example(profile, d, plan)
    sample_col, sample_map = _sample_mapping(plan)
    return {
        "name": name,
        "rows": n,
        "cols": min(d.shape[1], c.shape[1]),
        "error_cells": n_err,
        "recovery": score,
        "kept": score >= threshold,
        "max_cardinality": max_categorical_cardinality(plan),
        "sample_col": sample_col,
        "sample_map": sample_map,
        "record": record,
    }


def main() -> None:
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets", nargs="+",
        default=["hospital", "beers", "rayyan", "flights"],
        help="paired datasets to process",
    )
    ap.add_argument("--per-dataset", type=int, default=60,
                    help="how many small sub-table examples to draw per dataset")
    ap.add_argument("--include-tax", action="store_true",
                    help="also fetch+sample+DELETE the bulky tax table")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep raw CSVs on disk even for bulky datasets")
    ap.add_argument("--threshold", type=float, default=0.97,
                    help="min cell recovery to accept a sub-table example (self-verified)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=str, default="data/real_train.jsonl")
    args = ap.parse_args()

    datasets = list(args.datasets)
    if args.include_tax and "tax" not in datasets:
        datasets.append("tax")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = []
    total = 0
    best_overall = (0, "", "", {})  # (card, dataset, col, mapping)
    with out_path.open("w", encoding="utf-8") as f:
        for name in datasets:
            if name not in DATASETS:
                print(f"  skip unknown dataset: {name}")
                continue
            try:
                dirty, clean = fetch_pair(name, keep_raw=args.keep_raw)
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: FETCH FAILED ({type(e).__name__}: {e})")
                continue
            cols = canonicalizable_columns(dirty, clean)
            col_names = [str(dirty.columns[j]) for j in cols]
            made = 0
            maxcard = 0
            for record, plan, _score in iter_examples(
                    dirty, clean, rng, args.per_dataset, threshold=args.threshold):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                made += 1
                card = max_categorical_cardinality(plan)
                maxcard = max(maxcard, card)
                if card > best_overall[0]:
                    col, mp = _sample_mapping(plan)
                    best_overall = (card, name, col, mp)
            total += made
            rows.append((name, len(cols), made, maxcard, col_names[:6]))

    print("\n=== Real-data enrichment (many small self-verified tables) ===")
    hdr = f"{'dataset':<10}{'canon_cols':>11}{'examples':>10}{'maxcard':>9}  learnable columns"
    print(hdr)
    print("-" * len(hdr))
    for name, ncols, made, maxcard, names in rows:
        print(f"{name:<10}{ncols:>11}{made:>10}{maxcard:>9}  {', '.join(names)}")
    print(f"\nWrote {total} self-verified REAL training examples to {out_path}")
    if best_overall[0]:
        card, ds, col, mp = best_overall
        print(f"Richest real mapping: {ds}.{col} ({card} distinct variants). Sample:")
        for raw, canon in list(mp.items())[:6]:
            print(f"    {raw!r:>34} -> {canon!r}")


if __name__ == "__main__":
    main()
