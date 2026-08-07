from src.generate_submission import REQUIRED_COLUMNS
from src.generate_submission_gemini import build_submission_df


def test_build_submission_df_merges_and_cleans_text():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"]}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("Bötö.", 0.6)},
    }

    df = build_submission_df(["a.jpeg"], manifest_by_image, predictions_by_crop)

    assert list(df.columns) == REQUIRED_COLUMNS
    row = df.set_index("image_file").loc["a.jpeg"]
    assert row["verbatimDate"] == "1.7.2000"
    assert row["verbatimDate_confidence"] == 0.9
    assert row["verbatimLocality"] == "Bøtø"  # cleaned: ö->ø, trailing period stripped
    assert row["verbatimLocality_confidence"] == 0.6


def test_build_submission_df_card_missing_a_field_does_not_join_missing_in():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 0.9), "verbatimLocality": ("Kb", 0.9)},
        "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.7), "verbatimLocality": ("MISSING", 0.7)},
    }
    df = build_submission_df(["a.jpeg"], manifest_by_image, predictions_by_crop)
    row = df.set_index("image_file").loc["a.jpeg"]
    assert row["verbatimDate"] == "22.5.1977"
    assert row["verbatimLocality"] == "Kb"


def test_build_submission_df_defaults_absent_specimen_to_missing():
    df = build_submission_df(["a.jpeg"], {}, {})
    row = df.set_index("image_file").loc["a.jpeg"]
    assert row["verbatimDate"] == "MISSING"
    assert row["verbatimLocality"] == "MISSING"


def test_build_submission_df_applies_wholeimage_fallback_and_cleans_it():
    wholeimage_predictions_by_specimen = {
        "a.jpeg": {"verbatimDate": ("22.5.1977", 0.8), "verbatimLocality": ("Bötö.", 0.8)},
    }
    df = build_submission_df(["a.jpeg"], {}, {}, wholeimage_predictions_by_specimen)
    row = df.set_index("image_file").loc["a.jpeg"]
    assert row["verbatimDate"] == "22.5.1977"
    assert row["verbatimLocality"] == "Bøtø"  # fallback value also gets ö->ø + trailing-period cleanup
