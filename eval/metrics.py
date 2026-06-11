"""Metrics for scoring a predicted cleaning plan against gold + clean reference."""

from __future__ import annotations

import math

import pandas as pd
from jsonschema import Draft202012Validator

from scrubdata.executor import apply_plan

# Plan schema (validity gate). Permissive on extra keys; strict on shape/op names.
OP_NAMES = [
    "strip_whitespace", "collapse_internal_whitespace", "normalize_punctuation",
    "fix_encoding", "resolve_by_majority", "normalize_disguised_nulls",
    "standardize_case", "parse_currency", "parse_number", "parse_percent", "parse_date",
    "standardize_boolean", "standardize_phone", "normalize_email", "canonicalize_categories",
    "drop_empty_rows", "drop_empty_columns", "drop_exact_duplicates",
    "flag_pii", "mask_pii", "hash_pii", "pseudonymize_pii",
]
PLAN_SCHEMA = {
    "type": "object",
    "required": ["table_operations", "columns"],
    "properties": {
        "dataset_summary": {"type": "string"},
        "table_operations": {
            "type": "array",
            "items": {"type": "object", "required": ["op"],
                      "properties": {"op": {"enum": OP_NAMES}}},
        },
        "columns": {
            "type": "array",
            "items": {
                "type": "object", "required": ["name", "operations"],
                "properties": {
                    "name": {"type": "string"},
                    "operations": {
                        "type": "array",
                        "items": {"type": "object", "required": ["op"],
                                  "properties": {"op": {"enum": OP_NAMES}}},
                    },
                },
            },
        },
        "flags": {"type": "array"},
    },
}
_VALIDATOR = Draft202012Validator(PLAN_SCHEMA)


def is_valid(plan: dict) -> bool:
    return _VALIDATOR.is_valid(plan)


# --- feature extraction for set-based F1 -------------------------------------

def op_pairs(plan: dict) -> set:
    """Op-identity pairs for plan F1. PII ops are excluded: they are orthogonal to the
    cleaning gold (which predates PII support) and would unfairly penalize planners
    that flag sensitive columns."""
    s = {("<table>", t["op"]) for t in plan.get("table_operations", [])}
    for c in plan.get("columns", []):
        for o in c.get("operations", []):
            if "pii" not in o.get("op", ""):
                s.add((c["name"], o["op"]))
    return s


def canon_pairs(plan: dict) -> set:
    s = set()
    for c in plan.get("columns", []):
        for o in c.get("operations", []):
            if o["op"] == "canonicalize_categories":
                for raw, canon in o.get("mapping", {}).items():
                    s.add((c["name"], raw, canon))
    return s


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f1": f}


# --- end-to-end recovery -----------------------------------------------------

def _cell_equal(a, b) -> bool:
    am = a is None or (isinstance(a, float) and math.isnan(a)) or pd.isna(a)
    bm = b is None or (isinstance(b, float) and math.isnan(b)) or pd.isna(b)
    if am or bm:
        return am and bm
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return str(a) == str(b)


def recovery(clean_df: pd.DataFrame, dirty_df: pd.DataFrame, plan: dict) -> float:
    """Fraction of clean-reference cells recovered by executing `plan` on `dirty_df`."""
    try:
        cleaned, _ = apply_plan(dirty_df, plan)
    except Exception:
        return 0.0
    total = clean_df.size or 1
    matched = 0
    nrows = min(len(cleaned), len(clean_df))
    for col in clean_df.columns:
        if col not in cleaned.columns:
            continue  # missing column → all its cells count as unrecovered
        cl, pr = clean_df[col].tolist(), cleaned[col].tolist()
        for i in range(nrows):
            if _cell_equal(cl[i], pr[i]):
                matched += 1
    return matched / total
