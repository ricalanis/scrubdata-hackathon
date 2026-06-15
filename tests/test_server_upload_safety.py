import pandas as pd

from server import _read_any, clean_data
from scrubdata import apply_plan, mock_plan


def test_clean_data_handles_zero_byte_upload(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")

    result = clean_data(str(path), use_model=False)

    assert result["before"] == []
    assert result["after"] == []
    assert result["total_rows_before"] == 0
    assert "Couldn't read this file" in result["summary"]


def test_clean_data_handles_headers_without_rows(tmp_path):
    path = tmp_path / "headers.csv"
    path.write_text("name,email\n")

    result = clean_data(str(path), use_model=False)

    assert result["before"] == []
    assert result["after"] == []
    assert result["total_rows_before"] == 0
    assert result["summary"] == "That file looks empty — no rows or columns to clean."


def test_sanitized_header_suffixes_remain_unique(tmp_path):
    path = tmp_path / "colliding_headers.csv"
    path.write_text("a,a.1, a\nx,y,z\n")

    raw = _read_any(str(path))
    result = clean_data(str(path), use_model=False)

    assert list(raw.columns) == ["a", "a.1", "a.2"]
    assert raw.columns.is_unique
    assert result["columns_before"] == ["a", "a.1", "a.2"]
    assert result["total_rows_before"] == 1
    assert "Something went wrong while cleaning" not in result["summary"]


def test_maria_sample_cleaning_is_preserved():
    raw = pd.read_csv("samples/maria_crm_export.csv", dtype=str, keep_default_na=False)
    via_server = _read_any("samples/maria_crm_export.csv")

    expected, expected_log = apply_plan(raw, mock_plan(raw))
    actual, actual_log = apply_plan(via_server, mock_plan(via_server))

    pd.testing.assert_frame_equal(actual, expected)
    assert actual_log == expected_log
