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


def test_batched_planner():
    # agentic column-batching wrapper merges per-batch plans + deterministic table ops
    from scrubdata.model_planner import make_batched_planner
    from scrubdata.planner import mock_plan
    df = pd.read_csv("samples/dirty_contacts.csv", dtype=str, keep_default_na=False)
    plan = make_batched_planner(mock_plan, batch_size=3)(df)
    names = {c["name"] for c in plan["columns"]}
    assert {"country", "amount"} <= names                       # all columns covered
    assert any(o["op"] == "drop_empty_columns" for o in plan["table_operations"])
    assert is_valid(plan)


def test_reconcile_grounds_and_abstains():
    from scrubdata.reconcile import default_index
    idx = default_index()
    assert idx.reconcile("USA", "country")[0] == "United States"
    assert idx.reconcile("Germny", "country")[0] == "Germany"      # fuzzy
    assert idx.reconcile("Xyzzylandia", "country") is None          # ABSTAIN
    assert idx.reconcile("Califrnia", "state")[0] == "California"


def test_grounded_planner_no_wrong_merge():
    # 'guntxrsvillx' (a town not in the reference) must NOT be merged into a similar real
    # city — the structural fix for guntxrsvillx->huntsville.
    df = pd.DataFrame({"loc": ["birminghxm", "Birmingham", "guntxrsvillx", "Chicago",
                               "Chcago", "Birmingham", "Chicago", "Birmingham"]})
    plan = mock_plan(df)
    mapping = {k: v for c in plan["columns"] for o in c["operations"]
               if o["op"] == "canonicalize_categories" for k, v in o["mapping"].items()}
    assert mapping.get("birminghxm") == "Birmingham"
    assert mapping.get("guntxrsvillx", "") != "Huntsville"


def test_grounded_wrapper_overrides_model_overcorrection():
    from scrubdata.grounded import make_grounded_planner
    # a model that over-corrects (invents canonicals + wrong-merges)
    def fake_model(df, *a):
        return {"table_operations": [], "flags": [], "columns": [
            {"name": "city", "detected_semantic_type": "categorical", "issues": [],
             "operations": [{"op": "canonicalize_categories",
                             "mapping": {"birminghxm": "Birmingham City USA",
                                         "guntxrsvillx": "Huntsville"}}]}]}
    df = pd.DataFrame({"city": ["birminghxm", "Birmingham", "guntxrsvillx", "Chicago",
                               "Chcago", "Birmingham", "Chicago", "Birmingham"]})
    plan = make_grounded_planner(fake_model)(df)
    m = {k: v for c in plan["columns"] for o in c["operations"]
         if o["op"] == "canonicalize_categories" for k, v in o["mapping"].items()}
    assert m.get("birminghxm") == "Birmingham"            # grounded, not "Birmingham City USA"
    assert m.get("guntxrsvillx", "") != "Huntsville"      # wrong-merge blocked
    assert any(f["column"] == "city" for f in plan["flags"])   # abstained -> review flag


def test_pii_validators():
    from scrubdata.pii import luhn_ok, _is_credit_card, _is_iban
    assert luhn_ok("4532015112830366")
    assert not luhn_ok("4532015112830367")
    assert _is_credit_card("4532-0151-1283-0366")
    assert not _is_credit_card("1234567890123456")          # fails Luhn
    assert _is_iban("DE89370400440532013000")
    assert not _is_iban("DE89370400440532013001")           # fails mod-97


def test_pii_column_detection_and_negatives():
    from scrubdata.pii import detect_column_pii
    cards = ["4532015112830366", "4716461583322103", "5425233430109903", "4024007103939509"]
    r = detect_column_pii("card", cards)
    assert r and r["pii_type"] == "credit_card" and r["checksum"]
    r = detect_column_pii("ssn", ["123-45-6789", "987-65-4321", "111-22-3333"])
    assert r and r["pii_type"] == "ssn"
    assert detect_column_pii("city", ["Boston", "Chicago", "Dallas", "Boston"]) is None
    assert detect_column_pii("qty", ["1", "2", "3", "4", "5"]) is None


def test_pii_planner_masks_and_never_reformats_identifiers():
    df = pd.DataFrame({
        "card": ["4532015112830366", "4716461583322103", "5425233430109903",
                 "4024007103939509", "370434978549371"],   # last one fails Luhn (80% rate)
        "email": ["ana@corp.io", "luis@mail.com", "sofia@test.org", "raul@corp.io",
                  "mia@mail.com"],
        "city": ["Boston", "Chicago", "Boston", "Dallas", "Chicago"],
    })
    plan = mock_plan(df)
    assert is_valid(plan)
    ops = {c["name"]: [o["op"] for o in c["operations"]] for c in plan["columns"]}
    # checksum-confirmed at 80% coverage -> still auto-masked, never parse_number'd
    assert ops["card"] == ["flag_pii", "mask_pii"]
    assert "flag_pii" in ops["email"] and "mask_pii" not in ops["email"]
    assert "city" not in ops or not any("pii" in o for o in ops.get("city", []))
    cleaned, _ = apply_plan(df, plan)
    from scrubdata.pii import detect_column_pii
    assert detect_column_pii("card", cleaned["card"].tolist()) is None   # leak-free
    assert cleaned["card"][0].endswith("0366") and cleaned["card"][0].startswith("*")
    assert cleaned["email"][0] == "ana@corp.io"                          # flagged, not destroyed


def test_pii_hash_and_pseudonymize_deterministic():
    from scrubdata.pii import hash_value, pseudonymize_value
    assert hash_value("4532015112830366", "s1") == hash_value("4532015112830366", "s1")
    assert hash_value("4532015112830366", "s1") != hash_value("4532015112830366", "s2")
    p1 = pseudonymize_value("ana@corp.io", "s1", "email")
    assert p1 == pseudonymize_value("ana@corp.io", "s1", "email")   # join-stable
    assert p1.startswith("EMAIL_") and "ana" not in p1


def test_active_planner_defaults_to_heuristic(monkeypatch):
    monkeypatch.delenv("SCRUBDATA_MODEL", raising=False)
    from scrubdata.active import get_planner
    from scrubdata.planner import mock_plan
    assert get_planner() is mock_plan


def test_union_plans_model_wins_and_heuristic_extends():
    from scrubdata.verifier import union_plans
    primary = {"columns": [{"name": "city", "operations": [
        {"op": "canonicalize_categories", "mapping": {"bostn": "Boston"}}]}], "flags": []}
    secondary = {"columns": [
        {"name": "city", "operations": [{"op": "canonicalize_categories",
                                         "mapping": {"bostn": "BOSTON", "chcago": "Chicago"}}]},
        {"name": "state", "operations": [{"op": "canonicalize_categories",
                                          "mapping": {"texs": "Texas"}}]},
    ]}
    out = union_plans(primary, secondary)
    maps = {c["name"]: c["operations"][0]["mapping"] for c in out["columns"]}
    assert maps["city"]["bostn"] == "Boston"          # primary wins on conflict
    assert maps["city"]["chcago"] == "Chicago"        # secondary extends coverage
    assert maps["state"] == {"texs": "Texas"}         # secondary-only column added
    assert primary["columns"][0]["operations"][0]["mapping"] == {"bostn": "Boston"}  # no mutation


def test_active_planner_is_verified_union(monkeypatch):
    monkeypatch.setenv("SCRUBDATA_MODEL", "test-model")
    from scrubdata.active import get_planner
    planner = get_planner()
    # the model backend isn't reachable in tests -> per-batch heuristic fallback kicks
    # in; the plan must still come out tagged as the verified union pipeline
    df = pd.DataFrame({"city": ["Boston", "Boston", "Bostn", "Chicago", "Chicago"]})
    plan = planner(df)
    assert plan["_generated_by"] == "verified-union(model:test-model, tau=0.5)"
    assert is_valid(plan)


def test_pair_profile_candidates_and_constraint():
    from scrubdata.pair_profile import candidate_pairs, constrain_plan
    col = ["Boston"] * 8 + ["Chicago"] * 6 + ["Bostn", "Chcago", "Qwortelby"]
    pairs = candidate_pairs(col)
    by_raw = {p["raw"]: [c["canon"] for c in p["candidates"]] for p in pairs}
    assert "Boston" in by_raw.get("Bostn", [])
    assert "Chicago" in by_raw.get("Chcago", [])
    assert "Qwortelby" not in by_raw                  # garbage gets no candidates
    assert "Boston" not in by_raw                     # frequent values are not suspicious
    plan = {"columns": [{"name": "city", "operations": [{
        "op": "canonicalize_categories", "rationale": "typos",
        "mapping": {"Bostn": "Boston", "Chcago": "Dallas", "Qwortelby": "Boston"}}]}],
        "flags": []}
    out = constrain_plan(plan, {"city": [{"raw": p["raw"],
                                          "candidates": [c["canon"] for c in p["candidates"]]}
                                         for p in pairs]})
    kept = out["columns"][0]["operations"][0]["mapping"]
    assert kept == {"Bostn": "Boston"}                # off-candidate + garbage dropped
    assert out["flags"] and out["flags"][0]["issue"] == "outside_candidate_pairs"


def test_jellyfish_prompt_construction():
    from eval.baselines_learned import di_prompt, ed_prompt, parse_di, parse_ed
    rec = {"city": "Bostn", "state": "MA"}
    ed = ed_prompt(rec, "city")
    assert "Record [city: Bostn, state: MA]" in ed
    assert "Attribute for Verification: [city: Bostn]" in ed
    assert ed.endswith("### Response:\n\n")
    di = di_prompt(rec, "city", "geography")
    assert "Record: [state: MA]" in di          # flagged attribute removed
    assert "city" in di and "Bostn" not in di   # model infers, never copies
    assert parse_ed("Yes, there is an error") and not parse_ed("No.")
    assert parse_di(" Boston ", "Bostn") == "Boston"
    assert parse_di("", "Bostn") == "Bostn"                      # abstain on empty
    assert parse_di("The value is\nBoston", "Bostn") == "Bostn"  # abstain on rambling


def test_value_counts_profile():
    df = pd.DataFrame({"country": ["USA", "USA", "usa", "Canada"]})
    prof = profile_dataframe(df)
    vc = dict((v, n) for v, n in prof["columns"][0]["value_counts"])
    assert vc["USA"] == 2 and "value_counts" in prof["columns"][0]


def test_cli(tmp_path):
    from scrubdata.cli import main
    out = tmp_path / "clean.csv"
    plan = tmp_path / "plan.json"
    rc = main(["samples/dirty_contacts.csv", "-o", str(out), "--plan", str(plan), "--quiet"])
    assert rc == 0
    assert out.exists() and plan.exists()
    cleaned = pd.read_csv(out)
    assert "notes2" not in cleaned.columns and len(cleaned) == 13
