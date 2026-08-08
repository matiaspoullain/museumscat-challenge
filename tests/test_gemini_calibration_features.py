import os
import tempfile

import cv2
import numpy as np

from src.gemini_calibration_features import (
    FEATURE_NAMES,
    FEATURE_ORDER,
    build_feature_rows,
    contributing_crops_by_specimen,
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
