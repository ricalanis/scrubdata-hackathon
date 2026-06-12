"""Scorer test suite (GroUSE-style): adversarial unit tests where the correct score
is known BY CONSTRUCTION. Targets eval/run_real_multi.py::score()/abstain_slice()/
_cell_only() and eval/wild_bench.py::behavioral() silent-edit accounting.

    uv run pytest tests/test_wildclean_scorer.py -q
"""

import pandas as pd
import pytest

import eval.wild_bench as wb
from eval.metrics import _cell_equal
from eval.run_real_multi import _cell_only, _sem_equal, abstain_slice, score
from scrubdata.executor import apply_plan

NOOP_PLAN = {"table_operations": [], "columns": [], "flags": []}


def _df(**cols):
    return pd.DataFrame({k: list(v) for k, v in cols.items()}, dtype=str)


def _S(dirty, clean, out):
    return score(_df(**dirty), _df(**clean), _df(**out))


# ---- score(): the four canonical regimes -----------------------------------

def test_noop_recall_zero_damage_zero():
    # 2 errors, 2 clean cells; output == input -> nothing fixed, nothing damaged.
    m = _S({"x": ["Chcago", "Bostton", "Houston", "Dallas"]},
           {"x": ["Chicago", "Boston", "Houston", "Dallas"]},
           {"x": ["Chcago", "Bostton", "Houston", "Dallas"]})
    assert m["recall"] == 0.0 and m["damage"] == 0.0
    assert m["precision"] == 1.0          # vacuous: no changes attempted
    assert m["f1"] == 0.0 and m["_errors"] == 2 and m["_changed"] == 0


def test_oracle_output_f1_one():
    m = _S({"x": ["Chcago", "Bostton", "Houston"], "y": ["1", "2", "3"]},
           {"x": ["Chicago", "Boston", "Houston"], "y": ["1", "2", "3"]},
           {"x": ["Chicago", "Boston", "Houston"], "y": ["1", "2", "3"]})
    assert m["f1"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["damage"] == 0.0 and m["_fixed"] == 2


def test_abstain_everything_via_empty_plan():
    # The actual no-op PLAN path: apply_plan with an empty plan is the identity.
    dirty = _df(x=["Chcago", "Dallas"], y=[" a ", "b"])
    cleaned, log = apply_plan(dirty, NOOP_PLAN)
    assert cleaned.equals(dirty) and log == []
    m = score(dirty, _df(x=["Chicago", "Dallas"], y=["a", "b"]), cleaned)
    assert m["recall"] == 0.0 and m["damage"] == 0.0 and m["_changed"] == 0


def test_vandalism_high_damage_low_precision():
    # 10 clean cells + 2 errors; vandal garbles 5 clean cells, touches no error.
    dirty = {"x": ["Chcago", "Bostton"] + ["Dallas"] * 10}
    clean = {"x": ["Chicago", "Boston"] + ["Dallas"] * 10}
    out = {"x": ["Chcago", "Bostton"] + [f"ZZ{i}" for i in range(5)] + ["Dallas"] * 5}
    m = _S(dirty, clean, out)
    assert m["damage"] == 5 / 10 and m["precision"] == 0.0
    assert m["recall"] == 0.0 and m["f1"] == 0.0 and m["_changed"] == 5


# ---- score(): damage accounting ---------------------------------------------

def test_one_corrupted_clean_cell_damage_is_one_over_clean_cells():
    # 12 clean cells (2 cols x 6 rows), zero errors; corrupt exactly one.
    base = {"a": ["p"] * 6, "b": ["q"] * 6}
    out = {"a": ["GARBAGE"] + ["p"] * 5, "b": ["q"] * 6}
    m = _S(base, base, out)
    assert m["damage"] == pytest.approx(1 / 12)
    assert m["_changed"] == 1 and m["precision"] == 0.0


def test_damage_denominator_excludes_error_cells():
    # 3 errors + 7 clean in one column; corrupt one CLEAN cell -> damage 1/7 not 1/10.
    dirty = {"x": ["e1", "e2", "e3"] + ["ok"] * 7}
    clean = {"x": ["E1", "E2", "E3"] + ["ok"] * 7}
    out = {"x": ["e1", "e2", "e3", "BAD"] + ["ok"] * 6}
    m = _S(dirty, clean, out)
    assert m["damage"] == pytest.approx(1 / 7)


def test_wrong_value_on_error_cell_is_not_damage():
    # Rewriting an ERROR cell to the wrong value hurts precision, not damage
    # (damage counts corrupted CLEAN cells only).
    m = _S({"x": ["Chcago", "Dallas"]}, {"x": ["Chicago", "Dallas"]},
           {"x": ["Houston", "Dallas"]})
    assert m["damage"] == 0.0 and m["precision"] == 0.0
    assert m["recall"] == 0.0 and m["_changed"] == 1


# ---- score(): churn neutrality ----------------------------------------------

def test_churn_only_change_counts_as_nothing():
    # Case/whitespace rewrite of a CLEAN cell that does not restore gold:
    # not a change, not a fix, not damage.
    m = _S({"x": ["Boston", "Dallas"]}, {"x": ["Boston", "Dallas"]},
           {"x": ["boston", " Dallas "]})
    assert m["_changed"] == 0 and m["damage"] == 0.0
    assert m["precision"] == 1.0 and m["recall"] == 0.0


def test_churn_would_count_naively():
    # Counterfactual: a naive raw-equality scorer WOULD count the churn rewrite
    # as a change AND as damage ('boston' != gold 'Boston' raw).
    assert not _cell_equal("boston", "Boston")   # naive chg: out != input
    assert not _cell_equal("boston", "Boston")   # naive damage: out != gold
    assert _sem_equal("boston", "Boston")        # churn-neutral folds it away


def test_error_rewritten_to_case_variant_of_itself_is_abstain():
    # Error cell rewritten to a case-variant of the INPUT (still wrong) is churn:
    # suppressed entirely -> counts as leaving the error untouched.
    m = _S({"x": ["bostton"]}, {"x": ["Boston"]}, {"x": ["BOSTTON"]})
    assert m["_changed"] == 0 and m["_fixed"] == 0
    assert m["recall"] == 0.0 and m["damage"] == 0.0


# ---- score(): semantic-fix semantics (action required) -----------------------

def test_error_untouched_but_sem_equal_to_gold_is_not_a_fix():
    # Case-injected error ('boston' vs gold 'Boston') left untouched: sem-equal
    # to gold but NO action -> not a fix.
    m = _S({"x": ["boston", "Dallas"]}, {"x": ["Boston", "Dallas"]},
           {"x": ["boston", "Dallas"]})
    assert m["_errors"] == 1 and m["_fixed"] == 0 and m["recall"] == 0.0


def test_sem_equal_fix_with_action_is_a_fix():
    # Typo fixed to a case-variant of gold: acted + right value -> full credit.
    m = _S({"x": ["Bostton", "Dallas"]}, {"x": ["Boston", "Dallas"]},
           {"x": ["boston", "Dallas"]})
    assert m["_fixed"] == 1 and m["recall"] == 1.0
    assert m["precision"] == 1.0 and m["f1"] == 1.0


def test_exact_case_fix_is_a_fix_despite_sem_equal_input():
    # 'boston' -> 'Boston' exactly restores gold; sem-equal to input must NOT
    # trigger the churn suppression when raw gold is restored.
    m = _S({"x": ["boston"]}, {"x": ["Boston"]}, {"x": ["Boston"]})
    assert m["_fixed"] == 1 and m["recall"] == 1.0 and m["_changed"] == 1


# ---- score(): structural edge cases ------------------------------------------

def test_missing_column_treated_as_unchanged():
    # A column absent from the output scores as untouched (ov = dirty value).
    dirty = _df(x=["Chcago", "Dallas"], y=["1", "2"])
    clean = _df(x=["Chicago", "Dallas"], y=["1", "2"])
    m = score(dirty, clean, _df(y=["1", "2"]))
    assert m["recall"] == 0.0 and m["_changed"] == 0 and m["damage"] == 0.0


def test_numeric_tolerance_no_false_error_no_false_change():
    # '1.0' vs '1' is the same value; '1.000' output is not a change.
    m = _S({"x": ["1.0", "2"]}, {"x": ["1", "2"]}, {"x": ["1.000", "2"]})
    assert m["_errors"] == 0 and m["_changed"] == 0 and m["damage"] == 0.0


def test_rows_beyond_min_length_are_unscored():
    # n = min(len(dirty), len(out), len(clean)): a short output truncates the
    # scoring window (why _cell_only must strip row-dropping ops).
    dirty = _df(x=["a", "b", "Chcago"])
    clean = _df(x=["a", "b", "Chicago"])
    m = score(dirty, clean, _df(x=["a", "b"]))
    assert m["_errors"] == 0          # the row-3 error fell outside the window


def test_mixed_arithmetic_exact():
    # 2 errors: fix 1, leave 1. 8 clean: vandalize 1.
    # recall 1/2; precision 1/2 (1 good of 2 changes); damage 1/8; f1 1/2.
    dirty = {"x": ["Chcago", "Bostton"] + ["ok"] * 8}
    clean = {"x": ["Chicago", "Boston"] + ["ok"] * 8}
    out = {"x": ["Chicago", "Bostton", "BAD"] + ["ok"] * 7}
    m = _S(dirty, clean, out)
    assert m["recall"] == 0.5 and m["precision"] == 0.5
    assert m["f1"] == pytest.approx(0.5) and m["damage"] == pytest.approx(1 / 8)


def test_multicolumn_independent_accounting():
    # Fix the col-a error; vandalize one clean col-b cell. 3 clean cells total
    # remain undamaged out of 5 clean.
    dirty = {"a": ["Chcago", "x", "y"], "b": ["p", "q", "r"]}
    clean = {"a": ["Chicago", "x", "y"], "b": ["p", "q", "r"]}
    out = {"a": ["Chicago", "x", "y"], "b": ["p", "BAD", "r"]}
    m = _S(dirty, clean, out)
    assert m["recall"] == 1.0 and m["precision"] == 0.5
    assert m["damage"] == pytest.approx(1 / 5)


def test_no_errors_oracle_is_vacuous_not_rewarded():
    # Clean table, untouched output: recall 0 by convention (no errors), f1 0.
    base = {"x": ["a", "b"]}
    m = _S(base, base, base)
    assert m["_errors"] == 0 and m["recall"] == 0.0
    assert m["precision"] == 1.0 and m["f1"] == 0.0


def test_sem_equal_unit():
    assert _sem_equal(" Birmingham ", "birmingham")
    assert _sem_equal("1.0", "1")
    assert not _sem_equal("Birmingham", "Boston")


# ---- abstain_slice(): trap accounting ----------------------------------------

def _canon_planner(mapping):
    return lambda df: {"table_operations": [], "flags": [], "columns": [
        {"name": "city", "operations": [
            {"op": "canonicalize_categories", "mapping": mapping}]}]}


def test_abstain_slice_noop_planner():
    # Abstain-everything: all 4 traps left -> abstain 1.0. typo_recall floor is
    # 1/3, NOT 0: the 'Houston'->'Houston' control is kind='typo' and is scored
    # by sem-equality WITHOUT requiring action (unlike score()).
    r = abstain_slice(lambda df: dict(NOOP_PLAN))
    assert r["abstain_accuracy"] == 1.0
    assert r["typo_recall"] == pytest.approx(1 / 3)
    assert r["_typos"] == 3 and r["_traps"] == 4


def test_abstain_slice_oracle_planner():
    r = abstain_slice(_canon_planner({"Chcago": "Chicago", "Bostton": "Boston"}))
    assert r["typo_recall"] == 1.0 and r["abstain_accuracy"] == 1.0


def test_abstain_slice_overmerger_punished():
    # Freq-cluster-style over-merge: maps every trap into a dominant canon value.
    r = abstain_slice(_canon_planner({"Xqzzyville": "Chicago", "Qwortelby": "Boston",
                                      "Zzanthor Flats": "Houston", "Carmel": "Chicago"}))
    assert r["abstain_accuracy"] == 0.0
    assert r["typo_recall"] == pytest.approx(1 / 3)   # only the Houston control


# ---- _cell_only(): row-op stripping keeps row alignment -----------------------

def test_cell_only_strips_exactly_row_dropping_ops():
    plan = {"table_operations": [{"op": "drop_exact_duplicates"},
                                 {"op": "drop_empty_rows"},
                                 {"op": "drop_empty_columns", "columns": []}],
            "columns": [{"name": "x", "operations": [{"op": "strip_whitespace"}]}],
            "flags": ["f"]}
    p = _cell_only(plan)
    assert [o["op"] for o in p["table_operations"]] == ["drop_empty_columns"]
    assert p["columns"] == plan["columns"] and p["flags"] == plan["flags"]
    assert len(plan["table_operations"]) == 3     # original not mutated


def test_row_op_stripping_keeps_row_alignment():
    df = _df(x=["dup", "dup", "Chcago"], y=["1", "1", "2"])
    plan = {"table_operations": [{"op": "drop_exact_duplicates"}], "flags": [],
            "columns": [{"name": "x", "operations": [
                {"op": "canonicalize_categories", "mapping": {"Chcago": "Chicago"}}]}]}
    full, _ = apply_plan(df, plan)
    stripped, _ = apply_plan(df, _cell_only(plan))
    assert len(full) == 2                          # dedup WOULD break alignment
    assert len(stripped) == 3                      # stripped plan stays aligned
    clean = _df(x=["dup", "dup", "Chicago"], y=["1", "1", "2"])
    m = score(df, clean, stripped)
    assert m["recall"] == 1.0 and m["damage"] == 0.0


# ---- wild_bench behavioral(): silent-edit accounting ---------------------------

WB_DF = pd.DataFrame({"a": [" x ", "y", "z"], "b": ["m", "n", "o"]})
WB_PLAN = {"table_operations": [], "flags": [], "columns": [
    {"name": "a", "operations": [{"op": "strip_whitespace"}]}]}


def _fake_apply(change_col, log_entries):
    def fake(df, plan):
        out = df.copy()
        out[change_col] = out[change_col].str.upper()
        return out, list(log_entries)
    return fake


def test_silent_edit_trips_accounting(monkeypatch):
    # Pipeline changes column 'b' but logs only 'a' -> 'b' must be flagged silent.
    monkeypatch.setattr(wb, "mock_plan", lambda df: WB_PLAN)
    monkeypatch.setattr(wb, "apply_plan", _fake_apply(
        "b", [{"scope": "column", "column": "a", "op": "strip_whitespace",
               "cells_changed": 1}]))
    assert wb.behavioral(WB_DF)["silent_edit_columns"] == ["b"]


def test_logged_edit_is_not_silent(monkeypatch):
    monkeypatch.setattr(wb, "mock_plan", lambda df: WB_PLAN)
    monkeypatch.setattr(wb, "apply_plan", _fake_apply(
        "b", [{"scope": "column", "column": "b", "op": "standardize_case",
               "cells_changed": 3}]))
    assert wb.behavioral(WB_DF)["silent_edit_columns"] == []


def test_resolve_by_majority_columns_credited(monkeypatch):
    # Table-scope voting logs no 'column' key; its plan-declared columns must
    # still be credited so they are not flagged silent.
    plan = {"table_operations": [{"op": "resolve_by_majority", "key_column": "a",
                                  "columns": ["b"]}], "columns": [], "flags": []}
    monkeypatch.setattr(wb, "mock_plan", lambda df: plan)
    monkeypatch.setattr(wb, "apply_plan", _fake_apply(
        "b", [{"scope": "table", "op": "resolve_by_majority", "cells_changed": 3}]))
    assert wb.behavioral(WB_DF)["silent_edit_columns"] == []


def test_shipped_pipeline_has_no_silent_edits():
    # Regression: the REAL mock_plan + apply_plan must attribute every changed
    # column on a table it actually edits (whitespace + disguised nulls).
    df = pd.DataFrame({"name": [" alice ", "bob ", " carol", "dan", "eve "],
                       "city": ["NYC", "N/A", "NYC", "NYC", "n/a"]})
    b = wb.behavioral(df)
    assert b["silent_edit_columns"] == []
    assert b["plan_valid"] and b["cells_changed"] > 0


def test_cell_equal_nan_string_is_self_equal():
    # regression: the literal string "Nan" (a person's name) parses to float NaN,
    # which is unequal to itself under isclose — must fall through to str equality
    from eval.metrics import _cell_equal
    assert _cell_equal("Nan", "Nan")
    assert not _cell_equal("Nan", "Dan")
    assert _cell_equal("nan", "nan")
    assert not _cell_equal("nan", "1.0")
    assert _cell_equal("1.0", "1")          # numeric tolerance still works
    assert _cell_equal(float("nan"), None)  # real missing still missing-equal
