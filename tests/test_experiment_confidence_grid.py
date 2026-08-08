import os
import tempfile

import cv2
import numpy as np
import pandas as pd

from src.experiment_confidence_grid import enumerate_configs, evaluate_config
from src.gemini_calibration_features import contributing_crops_by_specimen


def test_enumerate_configs_has_sixteen_valid_combinations():
    configs = enumerate_configs()
    assert len(configs) == 16
    # every (agreement, feature_set, penalty) triple appears, and calibrator_family
    # is only meaningful (i.e. varies) when feature_set == "scalar"
    scalar_families = {c["calibrator_family"] for c in configs if c["feature_set"] == "scalar"}
    assert scalar_families == {"isotonic", "platt", "shrinkage"}
    multi_families = {c["calibrator_family"] for c in configs if c["feature_set"] == "multi"}
    assert multi_families == {"logistic_multi"}


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
