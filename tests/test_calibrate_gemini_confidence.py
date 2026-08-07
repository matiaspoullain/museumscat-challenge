import tempfile

import pandas as pd

from src.calibrate_gemini_confidence import (
    fit_calibrators,
    load_calibrators,
    merge_predictions_by_specimen,
    oof_calibrated_confidences,
    save_calibrators,
)


def test_merge_predictions_by_specimen_joins_two_cards():
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("MISSING", 0.9), "verbatimLocality": ("Ti", 0.8)},
        "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.9), "verbatimLocality": ("Kulhuset", 0.6)},
    }
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}
    merged = merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, ["a.jpeg"])
    assert merged["a.jpeg"]["verbatimLocality"] == ("Ti | Kulhuset", 0.7)
    # neither card had a date -- the field is MISSING outright, not "MISSING | MISSING"
    assert merged["a.jpeg"]["verbatimDate"] == ("MISSING", 0.5)


def test_merge_predictions_by_specimen_card_missing_a_field_does_not_join_the_word_missing_in():
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 0.9), "verbatimLocality": ("Kb", 0.9)},
        "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.7), "verbatimLocality": ("MISSING", 0.7)},
    }
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}
    merged = merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, ["a.jpeg"])
    # card01 has no date/locality -- must not contribute a literal "MISSING" pipe segment
    assert merged["a.jpeg"]["verbatimDate"] == ("22.5.1977", 0.9)
    assert merged["a.jpeg"]["verbatimLocality"] == ("Kb", 0.9)


def test_merge_predictions_by_specimen_missing_crop_contributes_nothing():
    predictions_by_crop = {"a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("Kb", 0.8)}}
    # a_card01.jpeg was skipped (confound/not transcribed) -- absent from predictions_by_crop
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}
    merged = merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, ["a.jpeg"])
    assert merged["a.jpeg"]["verbatimDate"] == ("1.7.2000", 0.9)


def test_merge_predictions_by_specimen_no_crops_is_missing():
    merged = merge_predictions_by_specimen({}, {}, ["a.jpeg"])
    assert merged["a.jpeg"]["verbatimDate"] == ("MISSING", 0.5)
    assert merged["a.jpeg"]["verbatimLocality"] == ("MISSING", 0.5)


def _synthetic_dev_set():
    # 6 specimens: verbatimLocality predictions range from exact-match (high
    # confidence should map to high correctness) to way off (low confidence
    # should map to low correctness) -- enough signal for isotonic regression
    # to fit a non-degenerate monotonic curve.
    merged = {
        "a.jpeg": {"verbatimDate": ("1.7.2000", 0.95), "verbatimLocality": ("Kb", 0.95)},
        "b.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("Kb", 0.9)},
        "c.jpeg": {"verbatimDate": ("2.7.2000", 0.6), "verbatimLocality": ("Dyrehaven", 0.6)},
        "d.jpeg": {"verbatimDate": ("2.7.2000", 0.55), "verbatimLocality": ("Dyrehaven", 0.55)},
        "e.jpeg": {"verbatimDate": ("MISSING", 0.2), "verbatimLocality": ("MISSING", 0.2)},
        "f.jpeg": {"verbatimDate": ("MISSING", 0.15), "verbatimLocality": ("MISSING", 0.15)},
    }
    ground_truth_df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
        {"image_file": "b.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
        {"image_file": "c.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Rude Skov"},
        {"image_file": "d.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Rude Skov"},
        {"image_file": "e.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Rude Skov"},
        {"image_file": "f.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Rude Skov"},
    ])
    return merged, ground_truth_df


def test_fit_calibrators_is_monotonic_in_confidence():
    merged, ground_truth_df = _synthetic_dev_set()
    calibrators = fit_calibrators(merged, ground_truth_df)
    assert set(calibrators) == {"verbatimDate", "verbatimLocality"}
    calibrator = calibrators["verbatimLocality"]
    low = calibrator.predict([0.2])[0]
    high = calibrator.predict([0.95])[0]
    assert high >= low


def test_fit_calibrators_output_in_unit_range():
    merged, ground_truth_df = _synthetic_dev_set()
    calibrators = fit_calibrators(merged, ground_truth_df)
    for calibrator in calibrators.values():
        for raw in (0.0, 0.3, 0.7, 1.0):
            calibrated = calibrator.predict([raw])[0]
            assert 0.0 <= calibrated <= 1.0


def test_oof_calibrated_confidences_covers_every_specimen():
    merged, ground_truth_df = _synthetic_dev_set()
    result = oof_calibrated_confidences(merged, ground_truth_df)
    assert set(result) == {"verbatimDate", "verbatimLocality"}
    for values in result.values():
        assert len(values) == len(ground_truth_df)
        assert all(v is not None and 0.0 <= v <= 1.0 for v in values)


def test_save_and_load_calibrators_roundtrip():
    merged, ground_truth_df = _synthetic_dev_set()
    calibrators = fit_calibrators(merged, ground_truth_df)
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_calibrators(calibrators, models_dir=tmp_dir)
        loaded = load_calibrators(models_dir=tmp_dir)
        for field in calibrators:
            original = calibrators[field].predict([0.6])[0]
            restored = loaded[field].predict([0.6])[0]
            assert original == restored
