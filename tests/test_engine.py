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
    # the model backend isn't reachable in tests -> every batch falls back to the
    # heuristic; get_planner must return the verified-union wrapper (only it emits this
    # honest label) and tag the plan as fallback rather than claiming the model ran.
    df = pd.DataFrame({"city": ["Boston", "Boston", "Bostn", "Chicago", "Chicago"]})
    plan = planner(df)
    assert plan["_generated_by"] == "deterministic (model unavailable, fell back)"
    assert is_valid(plan)


def test_convention_gates_regression():
    from scrubdata import detect
    from scrubdata.executor import _parse_percent, _standardize_phone
    # date gate: uniform slash / uniform month-name = consistent; mixed = not
    assert detect.date_formats_consistent(["1/4/2016", "12/23/2015", "3/7/2014"])
    assert detect.date_formats_consistent(["28 July 2016", "4 May 2015"])
    assert not detect.date_formats_consistent(["1/4/2016", "2015-12-23", "3/7/2014",
                                               "2014-01-02"])
    # 90% boundary: 1 stray in 20 stays consistent
    assert detect.date_formats_consistent(["1/4/2016"] * 19 + ["2016-01-04"])
    # percent gate: uniform-% gated; one stray of 20 still gated (no cliff)
    assert detect.percent_formats_consistent(["10%", "20%", "30%"])
    assert detect.percent_formats_consistent(["10%"] * 19 + ["0.6"])
    assert not detect.percent_formats_consistent(["10%", "0.2", "0.3"])
    # parse_percent abstains on bare values instead of /100 corruption
    assert _parse_percent("0.6") == "0.6"
    assert _parse_percent("45%") == 0.45
    # zip guard + Excel-serial name gate + 7-digit phone
    assert detect.detect_semantic_type("zipcode(long)", ["40231", "40213"] * 10) == "text"
    assert detect.detect_semantic_type("zcta", ["48371", "48380"] * 10) == "text"
    assert detect.detect_semantic_type("record_id", ["40231", "40213"] * 10) == "number"
    assert _standardize_phone("454.1763") == "454-1763"
    # end-to-end: consistent date column -> NO parse_date op + minority flagged
    df = pd.DataFrame({"issue_date": ["1/4/2016"] * 18 + ["1/5/2016", "2016-01-04"]})
    plan = mock_plan(df)
    ops = [o["op"] for c in plan["columns"] for o in c["operations"]]
    assert "parse_date" not in ops
    assert any(f["issue"] == "off_convention_dates" for f in plan["flags"])
    # mixed date column -> op present
    df2 = pd.DataFrame({"start": ["1/4/2016", "2015-12-23", "Apr-2014", "04/16/23"] * 5})
    ops2 = [o["op"] for c in mock_plan(df2)["columns"] for o in c["operations"]]
    assert "parse_date" in ops2


def test_verifier_gates_model_format_ops():
    from scrubdata.verifier import verify_plan
    df = pd.DataFrame({"d": ["1/4/2016", "2/5/2016", "3/6/2016"] * 4,
                       "p": ["10%", "20%", "30%"] * 4})
    model_plan = {"table_operations": [], "flags": [], "columns": [
        {"name": "d", "operations": [{"op": "parse_date", "rationale": "x"}]},
        {"name": "p", "operations": [{"op": "parse_percent", "rationale": "x"}]},
    ]}
    out = verify_plan(df, model_plan, tau=0.5)
    ops = [o["op"] for c in out["columns"] for o in c["operations"]]
    assert "parse_date" not in ops and "parse_percent" not in ops
    assert sum(1 for f in out["flags"] if f["issue"] == "convention_preserved") == 2


def test_voting_guards_regression():
    from scrubdata.planner import detect_entity_groups
    from scrubdata.executor import apply_plan
    # numeric votable column: detection excludes it; executor never crashes
    rows = []
    for f in range(25):
        for s in range(5):
            rows.append({"sku": f"SKU-{f}", "src": f"s{s}",
                         "label": ("ok" if not (f == 2 and s == 1) else "okk")
                                  + str(f % 4),
                         "qty": f * 10 + s})
    df = pd.DataFrame(rows)
    df["qty"] = df["qty"].astype("int64")
    eg = detect_entity_groups(df)
    if eg:
        assert "qty" not in eg[1]
    apply_plan(df, mock_plan(df))                  # must not raise
    # missing-like keys never form an entity group
    plan = {"table_operations": [{"op": "resolve_by_majority", "key_column": "k",
                                  "columns": ["v"]}], "columns": [], "flags": []}
    df2 = pd.DataFrame({"k": ["N/A"] * 6 + ["X-1"] * 3,
                        "v": ["a", "a", "a", "a", "b", "c", "z", "z", "y"]})
    cleaned, _ = apply_plan(df2, plan)
    assert list(cleaned["v"][:6]) == ["a", "a", "a", "a", "b", "c"]   # untouched
    # plan params are clamped: model-emitted min_share=0 cannot force rewrites
    plan2 = {"table_operations": [{"op": "resolve_by_majority", "key_column": "k",
                                   "columns": ["v"], "min_share": 0.0,
                                   "min_group": 1}], "columns": [], "flags": []}
    df3 = pd.DataFrame({"k": ["G1"] * 4, "v": ["a", "b", "b", "c"]})   # 50% max
    cleaned3, _ = apply_plan(df3, plan2)
    assert list(cleaned3["v"]) == ["a", "b", "b", "c"]
    # false-consensus guard: fat minorities (1 of 4 = legitimate updates) decline;
    # thin minorities (1 of 10 = reporting errors) proceed
    df4 = pd.DataFrame({"k": [f"G{i//4}" for i in range(40)],
                        "v": ["m", "m", "m", "x"] * 10})
    plan4 = {"table_operations": [{"op": "resolve_by_majority", "key_column": "k",
                                   "columns": ["v"]}], "columns": [], "flags": []}
    cleaned4, log4 = apply_plan(df4, plan4)
    entry = next(e for e in log4 if e["op"] == "resolve_by_majority")
    assert entry["cells_changed"] == 0 and "declined" in entry["detail"]
    df5 = pd.DataFrame({"k": [f"G{i//10}" for i in range(40)],
                        "v": (["m"] * 9 + ["x"]) * 4})
    cleaned5, log5 = apply_plan(df5, plan4)
    entry5 = next(e for e in log5 if e["op"] == "resolve_by_majority")
    assert entry5["cells_changed"] == 4                  # thin dissenters resolved
    # date-shaped keys are rejected
    rows = [{"date": f"2024-01-{d+1:02d}", "site": f"site-{r % 3}", "crew": f"c{r % 4}",
             "reading": f"v{r}"} for d in range(25) for r in range(5)]
    assert detect_entity_groups(pd.DataFrame(rows)) is None


def test_union_inherits_vote_op_and_preserves_op_order():
    from scrubdata.verifier import union_plans
    primary = {"table_operations": [], "columns": [], "flags": []}
    secondary = {"table_operations": [{"op": "resolve_by_majority", "key_column": "k",
                                       "columns": ["v"], "rationale": "vote"}],
                 "columns": [{"name": "t", "operations": [
                     {"op": "fix_encoding", "rationale": "enc"},
                     {"op": "normalize_punctuation", "rationale": "punct"},
                 ]}], "flags": []}
    out = union_plans(primary, secondary)
    assert any(o["op"] == "resolve_by_majority" for o in out["table_operations"])
    t_ops = [o["op"] for c in out["columns"] if c["name"] == "t"
             for o in c["operations"]]
    assert t_ops.index("fix_encoding") < t_ops.index("normalize_punctuation")


def test_fix_encoding_op():
    from scrubdata.executor import _fix_encoding
    assert _fix_encoding("café".encode("utf-8").decode("cp1252")) == "café"
    assert _fix_encoding("naïve résumé".encode("utf-8").decode("latin-1")) == "naïve résumé"
    assert _fix_encoding("plain text") == "plain text"        # untouched
    df = pd.DataFrame({"title": ["cafÃ© latte", "normal row"] * 6})
    plan = mock_plan(df)
    ops = [o["op"] for c in plan["columns"] for o in c["operations"]]
    assert "fix_encoding" in ops
    cleaned, _ = apply_plan(df, plan)
    assert cleaned["title"][0] == "café latte"


def test_resolve_by_majority_voting():
    rows = []
    for f in range(25):                       # 25 flights x 5 source reports
        for s in range(5):
            dep = f"{(f % 12) + 1}:58 p.m."
            arr = f"{(f % 11) + 1}:10 a.m."
            if (f, s) in ((3, 4), (9, 1)):
                dep = "7:59 p.m."             # two corrupted reports, two groups
            if (f, s) in ((7, 2), (12, 0)):
                arr = "9:40 a.m."
            rows.append({"flight": f"AA-{1000+f}", "src": f"src{s}", "dep": dep,
                         "arr": arr, "gate": f"G{f}"})
    df = pd.DataFrame(rows)
    plan = mock_plan(df)
    vote = [o for o in plan["table_operations"] if o["op"] == "resolve_by_majority"]
    assert vote and vote[0]["key_column"] == "flight"
    cleaned, log = apply_plan(df, plan)
    assert set(cleaned[cleaned["flight"] == "AA-1003"]["dep"]) == {"4:58 p.m."}
    entry = next(e for e in log if e["op"] == "resolve_by_majority")
    assert entry["cells_changed"] >= 1        # the minority report was resolved
    # no key regime -> no vote op
    df2 = pd.DataFrame({"a": [str(i) for i in range(40)], "b": ["x"] * 40})
    assert not any(o["op"] == "resolve_by_majority"
                   for o in mock_plan(df2)["table_operations"])


def test_suspects_visibility_high_cardinality():
    from scrubdata.profiler import profile_column
    # high-card "text" column: 60 unique names + one near-dup of a repeated one
    names = [f"unique business {i}" for i in range(57)]
    col = names + ["acme holdings", "acme holdings", "acme holdngs"]
    prof = profile_column(pd.Series(col, name="business"))
    assert prof["detected_semantic_type"] == "text"
    sus = {s["raw"]: s["candidates"] for s in prof["suspect_values"]}
    assert "acme holdngs" in sus and "acme holdings" in sus["acme holdngs"]
    assert len(prof["suspect_values"]) <= 25            # bounded
    # heuristic now repairs it (verifier-gated), where before it emitted nothing
    df = pd.DataFrame({"business": col})
    plan = mock_plan(df)
    maps = {r: c for col_ in plan["columns"] for o in col_["operations"]
            if o["op"] == "canonicalize_categories" for r, c in o["mapping"].items()}
    assert maps.get("acme holdngs") == "acme holdings"
    cleaned, _ = apply_plan(df, plan)
    assert "acme holdngs" not in set(cleaned["business"])
    # garbage suspect-free value stays put; plan still schema-valid
    assert is_valid(plan)


def test_suspects_garbage_flagged_not_mapped():
    from scrubdata.profiler import profile_column
    col = [f"item {i}" for i in range(40)] + ["it€m ’junk", "it€m ’junk"]
    df = pd.DataFrame({"thing": col})
    plan = mock_plan(df)
    maps = {r for c in plan["columns"] for o in c["operations"]
            if o["op"] == "canonicalize_categories" for r in o["mapping"]}
    assert "it€m ’junk" not in maps                     # no invented target
    assert is_valid(plan)


def test_normalize_punctuation_op():
    df = pd.DataFrame({"name": ["palm’s thai", "joe‘s “grill”", "a–b — c", "plain's ok"]})
    plan = mock_plan(df)
    ops = [o["op"] for c in plan["columns"] for o in c["operations"]]
    assert "normalize_punctuation" in ops
    cleaned, _ = apply_plan(df, plan)
    assert cleaned["name"][0] == "palm's thai"
    assert cleaned["name"][1] == 'joe\'s "grill"'
    assert cleaned["name"][2] == "a-b - c"
    # a clean column must NOT get the op
    plan2 = mock_plan(pd.DataFrame({"name": ["plain's ok", "also fine"]}))
    ops2 = [o["op"] for c in plan2["columns"] for o in c["operations"]]
    assert "normalize_punctuation" not in ops2


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
