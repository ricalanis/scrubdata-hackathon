import pandas as pd

from scrubdata import apply_plan, mock_plan


def test_already_clean_dataframe_cells_are_unchanged():
    df = pd.DataFrame({
        "name": ["Alice Carter", "Bruno Diaz", "Cora Evans"],
        "email": ["alice@example.com", "bruno@example.com", "cora@example.com"],
        "signup_date": ["2024-01-05", "2024-02-10", "2024-03-15"],
    })

    cleaned, _ = apply_plan(df, mock_plan(df))

    pd.testing.assert_frame_equal(cleaned, df)


def test_encoding_repair_changes_only_mojibake_cells():
    df = pd.DataFrame({
        "label": [
            "Cafe",
            "Caf\u00c3\u00a9",
            "M\u00fcnchen",
            "Fran\u00c3\u00a7ois",
        ],
        "status": ["clean", "clean", "clean", "clean"],
    })

    cleaned, _ = apply_plan(df, mock_plan(df))

    assert cleaned["label"].tolist() == [
        "Cafe",
        "Caf\u00e9",
        "M\u00fcnchen",
        "Fran\u00e7ois",
    ]
    assert cleaned["status"].tolist() == df["status"].tolist()
    assert cleaned.loc[[0, 2], "label"].tolist() == df.loc[[0, 2], "label"].tolist()


def test_exact_duplicate_removal_only_drops_true_duplicates_and_preserves_order():
    df = pd.DataFrame({
        "name": ["Ada", "Ben", "Ada", "Ada", "Cara"],
        "city": ["Boston", "Austin", "Boston", "Boston", "Denver"],
        "note": ["alpha", "beta", "alpha", "almost-alpha", "gamma"],
    })

    cleaned, _ = apply_plan(df, mock_plan(df))

    assert cleaned.to_dict("records") == [
        {"name": "Ada", "city": "Boston", "note": "alpha"},
        {"name": "Ben", "city": "Austin", "note": "beta"},
        {"name": "Ada", "city": "Boston", "note": "almost-alpha"},
        {"name": "Cara", "city": "Denver", "note": "gamma"},
    ]


def test_planner_abstained_values_are_left_exactly_as_is():
    df = pd.DataFrame({
        "row_key": [f"row-{i}" for i in range(8)],
        "city": [
            "birminghxm",
            "Birmingham",
            "guntxrsvillx",
            "Chicago",
            "Chcago",
            "Birmingham",
            "Chicago",
            "Birmingham",
        ],
    })

    plan = mock_plan(df)
    assert any("guntxrsvillx" in flag.get("values", []) for flag in plan["flags"])

    cleaned, _ = apply_plan(df, plan)

    assert cleaned.loc[df["city"] == "guntxrsvillx", "city"].tolist() == ["guntxrsvillx"]
    assert cleaned["row_key"].tolist() == df["row_key"].tolist()
