import os
import tempfile

import cv2
import numpy as np
import pandas as pd

import src.experiment_confidence_grid as experiment_confidence_grid
from src.experiment_confidence_grid import enumerate_configs, evaluate_config
from src.gemini_calibration_features import contributing_crops_by_specimen
from src.metrics import aurc


def test_enumerate_configs_has_twenty_six_valid_combinations():
    configs = enumerate_configs()
    assert len(configs) == 26
    # every (agreement, feature_set, penalty) triple appears, and calibrator_family
    # varies over 3 options for both "scalar" and "multi" feature sets
    scalar_families = {c["calibrator_family"] for c in configs if c["feature_set"] == "scalar"}
    assert scalar_families == {"isotonic", "platt", "shrinkage"}
    multi_families = {c["calibrator_family"] for c in configs if c["feature_set"] == "multi"}
    assert multi_families == {"logistic_multi_l2", "logistic_multi_l1", "pca_rf"}
    # the new "no fitted calibrator at all" arm: raw confidence, agreement
    # fixed to "hard", penalty toggled both ways -- 2 new configs total
    none_configs = [c for c in configs if c["feature_set"] == "none"]
    assert len(none_configs) == 2
    assert {c["calibrator_family"] for c in none_configs} == {"raw"}
    assert {c["agreement"] for c in none_configs} == {"hard"}
    assert {c["penalty"] for c in none_configs} == {False, True}


def _tiny_context(tmp_dir):
    images_dir = os.path.join(tmp_dir, "images")
    os.makedirs(images_dir)
    manifest_by_image = {}
    predictions_by_crop = {"hard": {}, "soft": {}}
    rows = []
    for i in range(12):
        image_file = f"s{i}.jpeg"
        crop = f"s{i}_card00.jpeg"
        cv2.imwrite(os.path.join(images_dir, crop), np.full((30, 40, 3), 100 + i, dtype=np.uint8))
        manifest_by_image[image_file] = [crop]
        correct = i % 2 == 0
        text = "1.7.2000" if correct else "9.9.9999"
        predictions_by_crop["hard"][crop] = {
            "verbatimDate": (text, 0.9 if correct else 0.3),
            "verbatimLocality": ("MISSING", 0.5),
        }
        predictions_by_crop["soft"][crop] = predictions_by_crop["hard"][crop]
        rows.append({"image_file": image_file, "verbatimDate": "1.7.2000", "verbatimLocality": "MISSING"})
    ground_truth_df = pd.DataFrame(rows)
    image_files = ground_truth_df["image_file"].tolist()
    contributing_crops = contributing_crops_by_specimen(manifest_by_image, predictions_by_crop["hard"], image_files)
    return {
        "ground_truth_df": ground_truth_df, "manifest_by_image": manifest_by_image,
        "image_files": image_files, "predictions_by_crop": predictions_by_crop,
        "wholeimage_predictions": {}, "contributing_crops": contributing_crops,
        "images_dir": images_dir,
    }


def test_evaluate_config_runs_for_every_combination():
    with tempfile.TemporaryDirectory() as tmp_dir:
        context = _tiny_context(tmp_dir)
        for config in enumerate_configs():
            overall, per_field = evaluate_config(config, context)
            assert 0.0 <= overall <= 1.0
            assert set(per_field) == {"verbatimDate", "verbatimLocality"}


def test_evaluate_config_none_feature_set_uses_raw_confidence_without_fitting(monkeypatch):
    # feature_set == "none" must not call either calibrator-fitting function --
    # it's the "no fitted calibrator at all" arm, so fitting anything here would
    # defeat the point. Monkeypatching both to raise proves evaluate_config
    # never reaches them for this feature_set.
    def _boom(*args, **kwargs):
        raise AssertionError("calibrator-fitting function should not be called for feature_set='none'")

    monkeypatch.setattr(experiment_confidence_grid, "oof_calibrated_confidences", _boom)
    monkeypatch.setattr(experiment_confidence_grid, "oof_multi_feature_calibrated_confidences", _boom)

    with tempfile.TemporaryDirectory() as tmp_dir:
        context = _tiny_context(tmp_dir)
        config = {"agreement": "hard", "feature_set": "none", "calibrator_family": "raw", "penalty": False}
        overall, per_field = evaluate_config(config, context)
        assert 0.0 <= overall <= 1.0

        # with penalty=False, the confidence values feeding aurc() must be
        # exactly with_fallback's raw confidence, unchanged
        with_fallback = experiment_confidence_grid._with_fallback(context["predictions_by_crop"]["hard"], context)
        errors_by_field = experiment_confidence_grid._errors_by_field(with_fallback, context["ground_truth_df"])
        for field in ("verbatimDate", "verbatimLocality"):
            raw_confidences = [with_fallback[image_file][field][1] for image_file in context["image_files"]]
            expected_aurc = aurc(errors_by_field[field], raw_confidences)
            assert per_field[field] == expected_aurc
