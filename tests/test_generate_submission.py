import pandas as pd

from src.generate_submission import REQUIRED_COLUMNS, build_submission_df, validate_submission


def test_valid_submission_has_no_errors():
    df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "MISSING", "verbatimDate_confidence": 0.5,
         "verbatimLocality": "Kb", "verbatimLocality_confidence": 0.9},
    ])
    assert validate_submission(df, {"a.jpeg"}) == []


def test_missing_column_is_an_error():
    df = pd.DataFrame([{"image_file": "a.jpeg"}])
    errors = validate_submission(df, {"a.jpeg"})
    assert any("column" in e.lower() for e in errors)


def test_empty_string_value_is_an_error():
    df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "", "verbatimDate_confidence": 0.5,
         "verbatimLocality": "Kb", "verbatimLocality_confidence": 0.9},
    ])
    errors = validate_submission(df, {"a.jpeg"})
    assert any("empty" in e.lower() for e in errors)


def test_confidence_out_of_range_is_an_error():
    df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "MISSING", "verbatimDate_confidence": 1.5,
         "verbatimLocality": "Kb", "verbatimLocality_confidence": 0.9},
    ])
    errors = validate_submission(df, {"a.jpeg"})
    assert any("confidence" in e.lower() for e in errors)


def test_missing_image_file_is_an_error():
    df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "MISSING", "verbatimDate_confidence": 0.5,
         "verbatimLocality": "Kb", "verbatimLocality_confidence": 0.9},
    ])
    errors = validate_submission(df, {"a.jpeg", "b.jpeg"})
    assert any("b.jpeg" in e for e in errors)


def test_build_submission_df_has_one_row_per_image_file_in_required_column_order():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"]}
    lines_by_crop = {"a_card00.jpeg": [{"text": "Kb", "confidence": 0.9}]}
    df = build_submission_df(["a.jpeg", "b.jpeg"], manifest_by_image, lines_by_crop)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert df.set_index("image_file").loc["a.jpeg", "verbatimLocality"] == "Kb"
    assert df.set_index("image_file").loc["b.jpeg", "verbatimDate"] == "MISSING"
