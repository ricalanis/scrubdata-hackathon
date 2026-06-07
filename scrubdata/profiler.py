"""Deterministic data profiling.

Produces a compact, model-friendly description of a dataframe: the exact thing
we feed the planner (mock today, fine-tuned model tomorrow). Keep it small —
the model sees the profile + a tiny row sample, never the whole table.
"""

from __future__ import annotations

import pandas as pd

from . import detect


def _column_issues(series: pd.Series, semantic_type: str) -> list[str]:
    values = series.tolist()
    issues: list[str] = []
    if detect.has_whitespace_issues(values):
        issues.append("whitespace")
    if detect.casing_variants(values):
        issues.append("casing")
    # disguised nulls present but column isn't all-empty
    if any(detect.is_missing(v) and str(v).strip() != "" and not _is_real_nan(v) for v in values):
        issues.append("disguised_nulls")
    if semantic_type in {"currency", "number", "percent"} and series.dtype == object:
        issues.append("numeric_stored_as_text")
    if semantic_type == "date":
        issues.append("mixed_date_formats")
    if semantic_type in {"categorical", "country"}:
        non_null = [str(v).strip().lower() for v in values if not detect.is_missing(v)]
        if len(set(non_null)) > _canonical_cardinality(non_null):
            issues.append("inconsistent_categories")
    return issues


def _is_real_nan(v) -> bool:
    try:
        import math
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


def _canonical_cardinality(lowered: list[str]) -> int:
    """How many distinct categories survive canonicalization (country dict + lower)."""
    canon = set()
    for v in lowered:
        canon.add(detect.COUNTRY_CANON.get(v, v))
    return len(canon)


def profile_column(series: pd.Series) -> dict:
    values = series.tolist()
    non_missing = [v for v in values if not detect.is_missing(v)]
    semantic_type = detect.detect_semantic_type(str(series.name), values)
    # up to 8 distinct sample values, as strings
    seen, samples = set(), []
    for v in non_missing:
        s = str(v).strip()
        if s not in seen:
            seen.add(s)
            samples.append(s)
        if len(samples) >= 8:
            break
    return {
        "name": str(series.name),
        "pandas_dtype": str(series.dtype),
        "detected_semantic_type": semantic_type,
        "n_total": len(values),
        "n_missing": sum(1 for v in values if detect.is_missing(v)),
        "n_unique": len({str(v).strip() for v in non_missing}),
        "sample_values": samples,
        "issues": _column_issues(series, semantic_type),
    }


def profile_dataframe(df: pd.DataFrame) -> dict:
    n_exact_dups = int(df.duplicated().sum())
    empty_cols = [c for c in df.columns
                  if df[c].apply(detect.is_missing).all()]
    empty_rows = int(df.apply(lambda r: all(detect.is_missing(v) for v in r), axis=1).sum())
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_exact_duplicate_rows": n_exact_dups,
        "n_empty_rows": empty_rows,
        "empty_columns": empty_cols,
        "columns": [profile_column(df[c]) for c in df.columns],
    }
