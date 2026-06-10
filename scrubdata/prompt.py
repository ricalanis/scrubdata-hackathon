"""Single source of truth for the planner's prompt format.

Both the training-data generator and the (future) fine-tuned model planner build
their text from here, so what the model trains on === what it sees at inference.
Change the prompt? Both sides move together.
"""

from __future__ import annotations

import json

import pandas as pd

SYSTEM_PROMPT = (
    "You are ScrubData, a meticulous tabular-data cleaning planner. "
    "Given a PROFILE of a messy spreadsheet (per-column dtype, missing counts, unique "
    "counts, detected semantic type, issues, and a value-frequency distribution — "
    "`value_counts` = [value, count] pairs over the WHOLE column) plus a few sample rows, "
    "output ONLY a JSON cleaning PLAN. Do not transform data yourself; deterministic code "
    "executes your plan. The value_counts let you reason about the whole column at any "
    "table size — canonicalize by mapping rare/misspelled/variant values to the dominant "
    "(high-count) canonical form.\n\n"
    "Plan schema:\n"
    "{\n"
    '  "dataset_summary": str,\n'
    '  "table_operations": [{"op": str, "columns"?: [str], "rationale": str}],\n'
    '  "columns": [{"name": str, "detected_semantic_type": str, "issues": [str],\n'
    '               "operations": [{"op": str, "mapping"?: {raw: canonical}, "rationale": str}]}],\n'
    '  "flags": [{"column": str, "issue": str, "action": "flag_only", "rationale": str}]\n'
    "}\n\n"
    "Operation vocabulary:\n"
    "  table: drop_empty_rows, drop_empty_columns, drop_exact_duplicates\n"
    "  column (safe): strip_whitespace, normalize_disguised_nulls, standardize_case,\n"
    "    parse_currency, parse_number, parse_percent, parse_date, standardize_boolean,\n"
    "    standardize_phone, normalize_email, canonicalize_categories\n"
    "Rules: prefer safe ops; for canonicalize_categories give the full {raw->canonical} "
    "mapping; only flag (never silently change) out-of-range or invalid values; output "
    "valid JSON, nothing else."
)


def _profile_for_prompt(profile: dict) -> dict:
    """Trim the full profile to the compact view the model is trained on."""
    return {
        "n_rows": profile["n_rows"],
        "n_cols": profile["n_cols"],
        "n_exact_duplicate_rows": profile["n_exact_duplicate_rows"],
        "n_empty_rows": profile["n_empty_rows"],
        "empty_columns": profile["empty_columns"],
        "columns": [
            {
                "name": c["name"],
                "pandas_dtype": c["pandas_dtype"],
                "detected_semantic_type": c["detected_semantic_type"],
                "n_missing": c["n_missing"],
                "n_unique": c["n_unique"],
                "issues": c["issues"],
                # [value, frequency] pairs — the distinct-value distribution. Canonicalize
                # by mapping rare/variant/misspelled values to the dominant canonical.
                "value_counts": c["value_counts"],
                "more_distinct_values": c.get("truncated_values", 0),
            }
            for c in profile["columns"]
        ],
    }


def build_user_prompt(profile: dict, sample_rows: pd.DataFrame, n_sample: int = 3,
                      candidate_pairs: dict | None = None) -> str:
    """`candidate_pairs` (WS2, optional): {col: [{"raw": v, "candidates": [c, ...]}]}.
    Default None keeps the exact training-time prompt — parity is the contract."""
    sample = sample_rows.head(n_sample).to_dict(orient="records")
    payload = {
        "profile": _profile_for_prompt(profile),
        "sample_rows": sample,
    }
    extra = ""
    if candidate_pairs:
        payload["candidate_pairs"] = candidate_pairs
        extra = (
            "\nCONSTRAINT: for canonicalize_categories, map a raw value ONLY to one of "
            "its listed candidate_pairs candidates, or omit the entry entirely "
            "(abstain). Never invent a canonical that is not listed for that value."
        )
    return (
        "PROFILE AND SAMPLE:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        + extra
        + "\n\nReturn the JSON cleaning plan."
    )


def serialize_plan(plan: dict) -> str:
    """Canonical JSON the model is trained to emit (drops internal-only keys)."""
    clean = {k: v for k, v in plan.items() if not k.startswith("_")}
    return json.dumps(clean, ensure_ascii=False, indent=2, default=str)


def build_chat_example(profile: dict, sample_rows: pd.DataFrame, plan: dict) -> dict:
    """A chat-format training record (messages) for SFT."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(profile, sample_rows)},
            {"role": "assistant", "content": serialize_plan(plan)},
        ]
    }
