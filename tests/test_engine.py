"""Engine regression tests: profiler detection, every executor op, end-to-end."""

import math

import pandas as pd
import pytest

from scrubdata import apply_plan, mock_plan, profile_dataframe
from scrubdata import detect
from eval.metrics import is_valid


def _col_plan(name, ops):
    return {"table_operations": [], "flags": [],
            "columns": [{"name": name, "operations": ops}]}


def _apply(series_vals, ops, col="x"):
    df = pd.DataFrame({col: series_vals})
    out, _ = apply_plan(df, _col_plan(col, ops))
    return out[col].tolist()


# ---- value-level ops --------------------------------------------------------

def test_strip_whitespace():
    assert _apply(["  a  b ", "c"], [{"op": "strip_whitespace"}]) == ["a b", "c"]


def test_parse_currency_us_eu_accounting():
    out = _apply(["$1,200.50", "1.200,50", "(500)", "950"], [{"op": "parse_currency"}])
    assert out == [1200.50, 1200.50, -500.0, 950.0]


def test_parse_percent():
    out = _apply(["12.5%", "100%"], [{"op": "parse_percent"}])
    assert out[0] == pytest.approx(0.125) and out[1] == pytest.approx(1.0)


def test_parse_date_formats():
    out = _apply(["2023-01-05", "1/6/2023", "5 Jan 2023", "44931"],
                 [{"op": "parse_date"}])
    assert out[0] == "2023-01-05"
    assert out[1] == "2023-01-06"
    assert out[2] == "2023-01-05"
    assert out[3] == "2023-01-05"  # excel serial


def test_standardize_boolean():
    out = _apply(["Yes", "Y", "TRUE", "1", "No", "N", "FALSE", "0"],
                 [{"op": "standardize_boolean"}])
    assert out == [True, True, True, True, False, False, False, False]


def test_standardize_phone_us():
    assert _apply(["555.123.4567"], [{"op": "standardize_phone"}]) == ["(555) 123-4567"]


def test_normalize_email():
    assert _apply([" Bob@X.COM "], [{"op": "normalize_email"}]) == ["bob@x.com"]


def test_standardize_case():
    assert _apply(["hello WORLD"], [{"op": "standardize_case", "case": "title"}]) == ["Hello World"]


def test_normalize_disguised_nulls():
    out = _apply(["N/A", "-", "real"], [{"op": "normalize_disguised_nulls"}])
    assert out[0] is None or pd.isna(out[0])
    assert out[2] == "real"


def test_canonicalize_categories():
    out = _apply(["usa", "U.S.A"], [{"op": "canonicalize_categories",
                  "mapping": {"usa": "United States", "U.S.A": "United States"}}])
    assert out == ["United States", "United States"]


# ---- table-level + end-to-end ----------------------------------------------

def test_drop_dupes_empty_rows_cols():
    df = pd.DataFrame({"a": ["1", "1", ""], "junk": ["", "", ""]})
    plan = {"table_operations": [
        {"op": "drop_empty_columns", "columns": ["junk"]},
        {"op": "drop_empty_rows"}, {"op": "drop_exact_duplicates"}],
        "columns": [], "flags": []}
    out, _ = apply_plan(df, plan)
    assert list(out.columns) == ["a"]
    assert out["a"].tolist() == ["1"]


def test_sample_end_to_end():
    df = pd.read_csv("samples/dirty_contacts.csv", dtype=str, keep_default_na=False)
    before = profile_dataframe(df)
    plan = mock_plan(df, before)
    cleaned, log = apply_plan(df, plan)
    assert "notes2" not in cleaned.columns          # empty col dropped
    assert len(cleaned) == 13                        # 16 -> dedup/empty -> 13
    assert is_valid(plan)                            # plan conforms to schema


def test_phone_conservatism():
    # consistent format -> heuristic should NOT emit standardize_phone
    df = pd.DataFrame({"phone": ["5551234567", "5559876543", "5550001111"]})
    plan = mock_plan(df)
    ops = [o["op"] for c in plan["columns"] for o in c["operations"]]
    assert "standardize_phone" not in ops


def test_detect_types():
    assert detect.detect_semantic_type("email", ["a@b.com", "c@d.com"]) == "email"
    assert detect.detect_semantic_type("x", ["Yes", "No", "Y"]) == "boolean"
