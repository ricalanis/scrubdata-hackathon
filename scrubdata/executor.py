"""Deterministic plan executor.

Consumes a cleaning plan (from the planner/model) and applies it to a copy of the
dataframe with pandas. Records a per-operation change log so the report and the
before/after diff can show *exactly* what changed (PRODUCT.md trust contract).

This module is final/real — the model never executes anything, it only plans.
"""

from __future__ import annotations

import re

import pandas as pd

from . import detect


# ---- value-level transforms -------------------------------------------------

def _strip_ws(v):
    if detect.is_missing(v):
        return v
    return re.sub(r"\s+", " ", str(v)).strip()


def _to_missing_if_disguised(v):
    return pd.NA if detect.is_missing(v) else v


def _parse_currency(v):
    if detect.is_missing(v):
        return pd.NA
    s = str(v).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace("£", "").replace("€", "").strip()
    # EU format "1.200,50" -> "1200.50"; US "1,200.50" -> "1200.50"
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):       # comma is decimal sep (EU)
            s = s.replace(".", "").replace(",", ".")
        else:                                  # comma is thousands sep (US)
            s = s.replace(",", "")
    elif "," in s:
        # ambiguous: treat comma as thousands unless it looks like decimals
        if re.match(r"^\d{1,3},\d{2}$", s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return pd.NA


def _parse_percent(v):
    if detect.is_missing(v):
        return pd.NA
    s = str(v).strip().rstrip("%").replace(",", ".")
    try:
        return float(s) / 100.0
    except ValueError:
        return pd.NA


def _parse_date(v):
    if detect.is_missing(v):
        return pd.NA
    s = str(v).strip()
    # Excel serial date
    if re.match(r"^\d{4,5}$", s):
        try:
            return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(s))).strftime("%Y-%m-%d")
        except Exception:
            return pd.NA
    # day-first if a slash/dash format has day > 12 anywhere; pandas infers else
    dayfirst = bool(re.match(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$", s)) and \
        int(s.split("/")[0].split("-")[0]) > 12
    for kwargs in ({"dayfirst": dayfirst}, {"dayfirst": not dayfirst}):
        try:
            ts = pd.to_datetime(s, **kwargs, errors="raise")
            return ts.strftime("%Y-%m-%d")
        except Exception:
            continue
    return pd.NA


def _standardize_boolean(v):
    if detect.is_missing(v):
        return pd.NA
    t = detect.normalize_token(v)
    if t in detect.BOOL_TRUE:
        return True
    if t in detect.BOOL_FALSE:
        return False
    return pd.NA


def _standardize_phone(v):
    if detect.is_missing(v):
        return pd.NA
    s = str(v).strip()
    plus = s.lstrip().startswith("+")
    digits = re.sub(r"\D", "", s)
    if plus:
        return "+" + digits
    if len(digits) == 10:                       # US-style
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:]}"
    return digits


def _normalize_email(v):
    if detect.is_missing(v):
        return pd.NA
    return str(v).strip().lower()


# ---- operation dispatch -----------------------------------------------------

def _apply_column_op(series: pd.Series, op: dict) -> pd.Series:
    name = op["op"]
    if name == "strip_whitespace":
        return series.map(_strip_ws)
    if name == "normalize_disguised_nulls":
        return series.map(_to_missing_if_disguised)
    if name == "parse_currency":
        return series.map(_parse_currency)
    if name == "parse_number":
        return series.map(_parse_currency)  # same numeric coercion, no symbol expectation
    if name == "parse_percent":
        return series.map(_parse_percent)
    if name == "parse_date":
        return series.map(_parse_date)
    if name == "standardize_boolean":
        return series.map(_standardize_boolean)
    if name == "standardize_phone":
        return series.map(_standardize_phone)
    if name == "normalize_email":
        return series.map(_normalize_email)
    if name == "canonicalize_categories":
        mapping = op.get("mapping", {})
        return series.map(lambda v: mapping.get(str(v).strip(), v) if not detect.is_missing(v) else v)
    # unknown op: no-op (forward-compatible with model-invented ops)
    return series


def apply_plan(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Apply `plan` to a copy of `df`. Returns (clean_df, change_log)."""
    out = df.copy()
    log: list[dict] = []

    # --- table-level ops (order: drop empty cols/rows, then dedup) ---
    for op in plan.get("table_operations", []):
        name = op["op"]
        if name == "drop_empty_columns":
            cols = [c for c in op.get("columns", []) if c in out.columns]
            out = out.drop(columns=cols)
            log.append({"scope": "table", "op": name, "detail": f"dropped columns {cols}"})
        elif name == "drop_empty_rows":
            before = len(out)
            mask = out.apply(lambda r: all(detect.is_missing(v) for v in r), axis=1)
            out = out[~mask]
            log.append({"scope": "table", "op": name, "detail": f"removed {before - len(out)} rows"})
        elif name == "drop_exact_duplicates":
            before = len(out)
            out = out.drop_duplicates()
            log.append({"scope": "table", "op": name, "detail": f"removed {before - len(out)} rows"})

    out = out.reset_index(drop=True)

    # --- column-level ops ---
    for col in plan.get("columns", []):
        cname = col["name"]
        if cname not in out.columns:
            continue
        for op in col.get("operations", []):
            before = out[cname].copy()
            out[cname] = _apply_column_op(out[cname], op)
            changed = int((before.astype(str).values != out[cname].astype(str).values).sum())
            log.append({
                "scope": "column", "column": cname, "op": op["op"],
                "cells_changed": changed,
                "rationale": op.get("rationale", ""),
            })

    return out, log
