import os
import tempfile

import cv2
import numpy as np

from src.gemini_calibration_features import (
    FEATURE_NAMES,
    FEATURE_ORDER,
    build_feature_rows,
    contributing_crops_by_specimen,
    fit_multi_feature_calibrator,
    fit_pca_rf_calibrator,
    oof_multi_feature_calibrated_confidences,
    oof_pca_rf_calibrated_confidences,
    predict_multi_feature,
)


def test_contributing_crops_by_specimen_only_counts_non_missing_fields():
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.9)},
        "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.7), "verbatimLocality": ("Kb", 0.7)},
    }
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}
    result = contributing_crops_by_specimen(manifest_by_image, predictions_by_crop, ["a.jpeg"])
    assert result["a.jpeg"]["verbatimDate"] == ["a_card00.jpeg"]
    assert result["a.jpeg"]["verbatimLocality"] == ["a_card01.jpeg"]


def test_contributing_crops_by_specimen_skips_crops_never_sent_to_gemini():
    predictions_by_crop = {"a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.9)}}
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}  # card01 filtered as confound, never called
    result = contributing_crops_by_specimen(manifest_by_image, predictions_by_crop, ["a.jpeg"])
    assert result["a.jpeg"]["verbatimDate"] == ["a_card00.jpeg"]


def _write_tiny_crop(images_dir, crop_filename, fill_value):
    image = np.full((40, 60, 3), fill_value, dtype=np.uint8)
    cv2.imwrite(os.path.join(images_dir, crop_filename), image)


def test_build_feature_rows_single_card_has_visual_features():
    with tempfile.TemporaryDirectory() as images_dir:
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)
        merged = {"a.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.5)}}
        predictions_by_crop = {"a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.9)}}
        contributing_crops = {"a.jpeg": {"verbatimDate": ["a_card00.jpeg"], "verbatimLocality": []}}

        rows = build_feature_rows(["a.jpeg"], merged, contributing_crops, predictions_by_crop, images_dir=images_dir)

        date_row = rows["verbatimDate"][0]
        assert date_row["has_visual_features"] == 1.0
        assert date_row["n_cards"] == 1.0
        assert date_row["cross_card_agreement"] == 0.5  # not applicable with <2 cards
        for name in FEATURE_NAMES:
            assert not np.isnan(date_row[name])

        locality_row = rows["verbatimLocality"][0]
        assert locality_row["has_visual_features"] == 0.0
        assert locality_row["n_cards"] == 0.0
        assert locality_row["text_length"] == 0.0  # "MISSING" contributes zero length
        for name in FEATURE_NAMES:
            assert np.isnan(locality_row[name])


def test_build_feature_rows_cross_card_agreement_and_disagreement():
    with tempfile.TemporaryDirectory() as images_dir:
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)
        _write_tiny_crop(images_dir, "a_card01.jpeg", fill_value=50)
        predictions_by_crop = {
            "a_card00.jpeg": {"verbatimDate": ("MISSING", 0.9), "verbatimLocality": ("Kb", 0.9)},
            "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.8), "verbatimLocality": ("Kb", 0.8)},
        }
        merged = {"a.jpeg": {"verbatimDate": ("MISSING", 0.5), "verbatimLocality": ("Kb | Kb", 0.85)}}
        contributing_crops = {"a.jpeg": {"verbatimDate": [], "verbatimLocality": ["a_card00.jpeg", "a_card01.jpeg"]}}
        rows = build_feature_rows(["a.jpeg"], merged, contributing_crops, predictions_by_crop, images_dir=images_dir)
        assert rows["verbatimLocality"][0]["cross_card_agreement"] == 1.0

        predictions_by_crop["a_card01.jpeg"]["verbatimLocality"] = ("Ti", 0.8)
        merged["a.jpeg"]["verbatimLocality"] = ("Kb | Ti", 0.85)
        rows = build_feature_rows(["a.jpeg"], merged, contributing_crops, predictions_by_crop, images_dir=images_dir)
        assert rows["verbatimLocality"][0]["cross_card_agreement"] == 0.0


def test_build_feature_rows_from_wholeimage_fallback_flag():
    merged = {"a.jpeg": {"verbatimDate": ("22.5.1977", 0.7), "verbatimLocality": ("MISSING", 0.5)}}
    contributing_crops = {"a.jpeg": {"verbatimDate": [], "verbatimLocality": []}}
    rows = build_feature_rows(["a.jpeg"], merged, contributing_crops, {}, images_dir="unused")
    # 0 contributing card crops but a non-MISSING date -- must have come from the
    # whole-image fallback, since the per-card merge alone would give "MISSING"
    assert rows["verbatimDate"][0]["from_wholeimage_fallback"] == 1.0
    assert rows["verbatimLocality"][0]["from_wholeimage_fallback"] == 0.0


def test_feature_order_starts_with_self_consistency():
    assert FEATURE_ORDER[0] == "self_consistency"
    assert set(FEATURE_ORDER) - {"self_consistency"} == set(
        ["text_length", "n_cards", "cross_card_agreement", "from_wholeimage_fallback", "has_visual_features"]
        + FEATURE_NAMES
    )


def _synthetic_rows(n=20):
    # self_consistency and correctness move together (with noise from n_cards)
    # so a fitted logistic regression has real signal to learn, not just noise
    rows = []
    correctness = []
    for i in range(n):
        confident = i % 2 == 0
        row = {name: 0.0 for name in FEATURE_ORDER}
        row["self_consistency"] = 0.95 if confident else 0.4
        row["n_cards"] = 1.0
        row["cross_card_agreement"] = 0.5
        row["has_visual_features"] = 0.0
        for name in FEATURE_NAMES:
            row[name] = float("nan")
        rows.append(row)
        correctness.append(1.0 if confident else 0.2)
    return rows, correctness


def test_fit_multi_feature_calibrator_predicts_higher_confidence_for_higher_self_consistency():
    rows, correctness = _synthetic_rows()
    pipeline = fit_multi_feature_calibrator(rows, correctness)
    assert pipeline is not None
    high_row = dict(rows[0])   # self_consistency=0.95 case
    low_row = dict(rows[1])    # self_consistency=0.4 case
    predictions = predict_multi_feature(pipeline, [high_row, low_row])
    assert predictions[0] > predictions[1]
    assert all(0.0 <= p <= 1.0 for p in predictions)


def test_fit_multi_feature_calibrator_degenerate_labels_returns_none():
    rows, _ = _synthetic_rows(n=4)
    all_correct = [1.0, 1.0, 1.0, 1.0]  # binarizes to a single class -- no boundary to fit
    assert fit_multi_feature_calibrator(rows, all_correct) is None


def test_oof_multi_feature_calibrated_confidences_covers_every_row():
    rows, correctness = _synthetic_rows()
    result = oof_multi_feature_calibrated_confidences(rows, correctness, n_folds=4, random_state=0)
    assert len(result) == len(rows)
    assert all(v is not None and 0.0 <= v <= 1.0 for v in result)


def test_fit_multi_feature_calibrator_l2_default_sets_l2_penalty():
    rows, correctness = _synthetic_rows()
    pipeline = fit_multi_feature_calibrator(rows, correctness)
    logistic = pipeline.named_steps["logisticregression"]
    assert logistic.penalty == "l2"


def test_fit_multi_feature_calibrator_l1_sets_l1_penalty_and_liblinear_solver():
    rows, correctness = _synthetic_rows()
    pipeline = fit_multi_feature_calibrator(rows, correctness, regularization="l1")
    logistic = pipeline.named_steps["logisticregression"]
    assert logistic.penalty == "l1"
    assert logistic.solver == "liblinear"


def test_fit_multi_feature_calibrator_l1_predictions_in_unit_range():
    rows, correctness = _synthetic_rows()
    pipeline = fit_multi_feature_calibrator(rows, correctness, regularization="l1")
    predictions = predict_multi_feature(pipeline, rows)
    assert all(0.0 <= p <= 1.0 for p in predictions)


def test_fit_multi_feature_calibrator_l1_degenerate_labels_returns_none():
    rows, _ = _synthetic_rows(n=4)
    all_correct = [1.0, 1.0, 1.0, 1.0]
    assert fit_multi_feature_calibrator(rows, all_correct, regularization="l1") is None


def test_oof_multi_feature_calibrated_confidences_l1_regularization_covers_every_row():
    rows, correctness = _synthetic_rows()
    result = oof_multi_feature_calibrated_confidences(
        rows, correctness, n_folds=4, random_state=0, regularization="l1"
    )
    assert len(result) == len(rows)
    assert all(v is not None and 0.0 <= v <= 1.0 for v in result)


def test_fit_pca_rf_calibrator_predicts_higher_confidence_for_higher_self_consistency():
    rows, correctness = _synthetic_rows(n=30)
    pipeline = fit_pca_rf_calibrator(rows, correctness, n_components=3, n_estimators=50)
    assert pipeline is not None
    high_row = dict(rows[0])   # self_consistency=0.95 case
    low_row = dict(rows[1])    # self_consistency=0.4 case
    predictions = predict_multi_feature(pipeline, [high_row, low_row])
    assert predictions[0] > predictions[1]
    assert all(0.0 <= p <= 1.0 for p in predictions)


def test_fit_pca_rf_calibrator_degenerate_labels_returns_none():
    rows, _ = _synthetic_rows(n=4)
    all_correct = [1.0, 1.0, 1.0, 1.0]
    assert fit_pca_rf_calibrator(rows, all_correct, n_components=2, n_estimators=10) is None


def test_oof_pca_rf_calibrated_confidences_covers_every_row():
    rows, correctness = _synthetic_rows(n=30)
    result = oof_pca_rf_calibrated_confidences(
        rows, correctness, n_folds=4, random_state=0, n_components=3, n_estimators=50
    )
    assert len(result) == len(rows)
    assert all(v is not None and 0.0 <= v <= 1.0 for v in result)
