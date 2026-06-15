import pandas as pd

from scrubdata.planner import mock_plan
from scrubdata.reconcile import default_index


def test_empty_string_abstains_without_candidates():
    idx = default_index()

    assert idx.best("", "country") is None
    assert idx.reconcile("", "country") is None
    assert idx.candidates("", "country") == []


def test_whitespace_only_abstains_without_candidates():
    idx = default_index()

    assert idx.best("   ", "country") is None
    assert idx.reconcile("   ", "country") is None
    assert idx.candidates("   ", "country") == []


def test_very_long_string_abstains_without_candidates():
    idx = default_index()
    value = "x" * 1000

    assert idx.best(value, "country") is None
    assert idx.reconcile(value, "country") is None
    assert idx.candidates(value, "country") == []


def test_unicode_accented_input_matches_existing_country_reference():
    idx = default_index()

    assert idx.best("Cote dIvoire", "country") == ("Côte d'Ivoire", 0.909, 0.409)
    assert idx.reconcile("Cote dIvoire", "country") == ("Côte d'Ivoire", 0.909)
    assert idx.candidates("Cote dIvoire", "country", k=2) == [("Côte d'Ivoire", 0.909)]


def test_tied_country_candidates_keep_current_order_and_zero_margin():
    idx = default_index()

    assert idx.best("Slovia", "country") == ("Slovakia", 0.857, 0.0)
    assert idx.reconcile("Slovia", "country") == ("Slovakia", 0.857)
    assert idx.candidates("Slovia", "country", k=2) == [
        ("Slovakia", 0.857),
        ("Slovenia", 0.857),
    ]


def test_planner_surfaces_tied_country_candidates_for_review():
    df = pd.DataFrame({"country": ["Slovia", "Slovakia", "Slovenia", "Somalia"]})

    plan = mock_plan(df)

    assert plan["columns"] == []
    assert plan["flags"] == [{
        "column": "country",
        "issue": "uncertain_canonicalization",
        "values": ["Slovia"],
        "action": "left_for_review",
        "candidates": {
            "Slovia": [
                {"canon": "Slovakia", "score": 0.857},
                {"canon": "Slovenia", "score": 0.857},
            ],
        },
        "rationale": "1 value(s) look like typos but did not confidently match the "
                     "reference — left unchanged for review.",
    }]


def test_value_with_no_close_country_match_abstains_without_candidates():
    idx = default_index()

    assert idx.best("Xyzzylandia", "country") is None
    assert idx.reconcile("Xyzzylandia", "country") is None
    assert idx.candidates("Xyzzylandia", "country") == []
