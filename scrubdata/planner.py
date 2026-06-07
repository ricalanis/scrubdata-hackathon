"""The planner: profile -> structured cleaning plan (JSON).

⚠️  MOCK. This heuristic stands in for the fine-tuned ≤4B model so the rest of
the pipeline + UI are buildable today. It deliberately mimics the model's job and
output shape (PRODUCT.md §5) so the swap is a one-liner:

    from scrubdata.planner import mock_plan as make_plan      # today
    from scrubdata.model    import model_plan as make_plan     # after fine-tune

The model will do the genuinely fuzzy work far better — especially
`canonicalize_categories` mappings — but the *contract* (this dict) stays fixed.
"""

from __future__ import annotations

import pandas as pd

from . import detect
from .profiler import profile_dataframe


def _canonicalize_mapping(values) -> dict:
    """Build a {raw_lower -> canonical} mapping for a categorical column.

    MOCK strategy: collapse by built-in country dict, else by case/whitespace to
    the most common surface form. The model will instead reason semantically
    ('NYC' == 'New York', typos, abbreviations) — this only catches the obvious.
    """
    from collections import Counter

    surface: dict[str, Counter] = {}
    for v in values:
        if detect.is_missing(v):
            continue
        raw = str(v).strip()
        key = detect.COUNTRY_CANON.get(raw.lower(), raw.lower())
        surface.setdefault(key, Counter())[raw] += 1

    mapping = {}
    for key, counter in surface.items():
        if key in detect.COUNTRY_CANON.values():
            canonical = key  # already a canonical country name
        else:
            canonical = counter.most_common(1)[0][0]  # most frequent surface form
        for raw in counter:
            if raw != canonical:
                mapping[raw] = canonical
    return mapping


def _column_operations(col_profile: dict, series: pd.Series) -> list[dict]:
    ops: list[dict] = []
    issues = set(col_profile["issues"])
    stype = col_profile["detected_semantic_type"]

    if "whitespace" in issues:
        ops.append({"op": "strip_whitespace",
                    "rationale": "Trimmed leading/trailing and doubled spaces."})
    if "disguised_nulls" in issues:
        ops.append({"op": "normalize_disguised_nulls",
                    "rationale": "Converted N/A, '-', 'null' etc. to true missing."})

    if stype == "currency":
        ops.append({"op": "parse_currency",
                    "rationale": "Stripped currency symbols/grouping; parsed to number."})
    elif stype == "number":
        ops.append({"op": "parse_number",
                    "rationale": "Parsed numeric text to number."})
    elif stype == "percent":
        ops.append({"op": "parse_percent",
                    "rationale": "Parsed percent text to fraction."})
    elif stype == "date":
        ops.append({"op": "parse_date",
                    "rationale": "Unified mixed date formats to ISO YYYY-MM-DD."})
    elif stype == "boolean":
        ops.append({"op": "standardize_boolean",
                    "rationale": "Mapped Yes/Y/1/TRUE → true, No/N/0/FALSE → false."})
    elif stype == "phone":
        # Conservative: only reformat when the column has mixed phone formats — don't
        # impose our format on a column that's already internally consistent.
        if "inconsistent_formats" in issues:
            ops.append({"op": "standardize_phone",
                        "rationale": "Unified inconsistent phone formats."})
    elif stype == "email":
        ops.append({"op": "normalize_email",
                    "rationale": "Lowercased and trimmed email addresses."})
    elif stype in {"categorical", "country"}:
        if "casing" in issues and not any(o["op"] == "canonicalize_categories" for o in ops):
            pass  # casing folded into canonicalization below
        if "inconsistent_categories" in issues or stype == "country":
            mapping = _canonicalize_mapping(series.tolist())
            if mapping:
                ops.append({
                    "op": "canonicalize_categories",
                    "mapping": mapping,
                    "rationale": f"Unified {len(mapping)} inconsistent spellings "
                                 f"into canonical labels.",
                })
    return ops


def mock_plan(df: pd.DataFrame, profile: dict | None = None) -> dict:
    """Return a cleaning plan dict for `df` (PRODUCT.md §5 schema)."""
    profile = profile or profile_dataframe(df)

    table_ops: list[dict] = []
    if profile["n_empty_rows"]:
        table_ops.append({"op": "drop_empty_rows",
                          "rationale": f"Removed {profile['n_empty_rows']} fully-empty row(s)."})
    if profile["empty_columns"]:
        table_ops.append({"op": "drop_empty_columns", "columns": profile["empty_columns"],
                          "rationale": "Dropped column(s) with no data."})
    if profile["n_exact_duplicate_rows"]:
        table_ops.append({"op": "drop_exact_duplicates",
                          "rationale": f"Removed {profile['n_exact_duplicate_rows']} "
                                       f"exact duplicate row(s)."})

    columns = []
    for col_profile in profile["columns"]:
        if col_profile["name"] in profile["empty_columns"]:
            continue
        ops = _column_operations(col_profile, df[col_profile["name"]])
        if ops:
            columns.append({
                "name": col_profile["name"],
                "detected_semantic_type": col_profile["detected_semantic_type"],
                "issues": col_profile["issues"],
                "operations": ops,
            })

    n_rows, n_cols = profile["n_rows"], profile["n_cols"]
    return {
        "dataset_summary": f"{n_rows} rows × {n_cols} columns. "
                           f"Detected {len(columns)} column(s) needing cleanup "
                           f"and {len(table_ops)} table-level fix(es).",
        "table_operations": table_ops,
        "columns": columns,
        "flags": [],  # the model will populate anomaly flags; mock leaves empty
        "_generated_by": "mock_planner",
    }
