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
    if detect.has_unicode_punctuation(values):
        issues.append("unicode_punctuation")
    if detect.casing_variants(values):
        issues.append("casing")
    # disguised nulls present but column isn't all-empty
    if any(detect.is_missing(v) and str(v).strip() != "" and not _is_real_nan(v) for v in values):
        issues.append("disguised_nulls")
    if semantic_type in {"currency", "number", "percent"} and series.dtype == object:
        issues.append("numeric_stored_as_text")
    if semantic_type == "date":
        issues.append("mixed_date_formats")
    if semantic_type == "phone" and not detect.phone_formats_consistent(values):
        issues.append("inconsistent_formats")
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


# Distinct values to surface per column. Bounded → the profile (and prompt) size is
# invariant to ROW count, so a 1M-row table profiles like a 100-row one. This is what
# lets the planner canonicalize at any scale: it reasons over the value distribution,
# not raw rows. (Columns with more distinct values than this are the high-cardinality
# tail handled by deterministic cluster candidates — see detect.cluster_candidates.)
VALUE_COUNTS_CAP = 80


def profile_column(series: pd.Series) -> dict:
    from collections import Counter

    values = series.tolist()
    non_missing = [str(v).strip() for v in values if not detect.is_missing(v)]
    semantic_type = detect.detect_semantic_type(str(series.name), values)
    counts = Counter(non_missing)
    # High-cardinality columns (IDs / free text — almost all values unique) aren't
    # canonicalizable, so just show a few; categorical-ish columns get the full
    # distribution (up to the cap) so the model sees every variant + its frequency.
    high_card = len(non_missing) >= 12 and len(counts) > 0.8 * len(non_missing)
    k = 8 if high_card else VALUE_COUNTS_CAP
    value_counts = [[val, cnt] for val, cnt in counts.most_common(k)]
    from .pii import detect_column_pii
    pii = detect_column_pii(str(series.name), values)
    if pii is None and semantic_type in ("text", "categorical"):
        import os
        if os.environ.get("SCRUBDATA_PII_NER"):     # tier-2 NER: opt-in (needs transformers)
            from .pii import detect_column_pii_ner
            pii = detect_column_pii_ner(str(series.name), values)
    # visibility for capped/high-card columns: rare anomalous surfaces + their
    # evidence-backed repair candidates (bounded; the value_counts cap hides these)
    suspects = []
    if pii is None and semantic_type in ("text", "categorical", "country", "city"):
        from .pair_profile import suspects_for_column
        suspects = suspects_for_column(values)
    return {
        "name": str(series.name),
        "pandas_dtype": str(series.dtype),
        "detected_semantic_type": semantic_type,
        "n_total": len(values),
        "n_missing": sum(1 for v in values if detect.is_missing(v)),
        "n_unique": len(counts),
        "value_counts": value_counts,
        "truncated_values": max(0, len(counts) - VALUE_COUNTS_CAP),
        "suspect_values": suspects,
        "issues": _column_issues(series, semantic_type),
        # PII typing: tier-1 regex+checksum always; tier-2 NER when opted in. None if not PII.
        "pii": pii,
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
