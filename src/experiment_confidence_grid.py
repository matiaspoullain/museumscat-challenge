"""Grid-evaluates every combination of this repo's four confidence-calibration
experiments (docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md)
against honest (leakage-free, out-of-fold) dev-set AURC, to pick which
combination (if any) beats the current production config -- agreement="hard",
feature_set="scalar", calibrator_family="isotonic", penalty=False is NOT the
production default (production actually skips calibration's isotonic step in
favor of raw confidence, per src.evaluate_gemini_dev's best-performing block) --
this grid's job is to find the best config among ones that DO calibrate.

Usage:
    conda run -n museumscat-challenge python -m src.experiment_confidence_grid
"""
import os

import pandas as pd

from src.calibrate_gemini_confidence import (
    apply_wholeimage_fallback,
    load_predictions_by_crop,
    load_wholeimage_predictions,
    merge_predictions_by_specimen,
    oof_calibrated_confidences,
    rescored_predictions_by_crop,
)
from src.clean_gemini_predictions import clean_merged_predictions
from src.confidence_penalty_rules import apply_confidence_penalty, count_processed_and_contributing
from src.eda_card_visual_features import IMAGES_DIR
from src.gemini_calibration_features import (
    build_feature_rows,
    contributing_crops_by_specimen,
    oof_multi_feature_calibrated_confidences,
)
from src.gemini_transcribe import self_consistency_scores, soft_consistency_scores
from src.metrics import aurc, best_pipe_ned
from src.pipeline import load_manifest

TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
DEV_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "dev_card_predictions.csv")
FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}
SCORERS = {"hard": self_consistency_scores, "soft": soft_consistency_scores}
SCALAR_CALIBRATOR_FAMILIES = ("isotonic", "platt", "shrinkage")
BASELINE_AURC = 0.075  # src.evaluate_gemini_dev's current best (raw confidence + cleanup + fallback)


def enumerate_configs():
    """Every valid (agreement, feature_set, calibrator_family, penalty)
    combination: 16 total. calibrator_family only varies when feature_set ==
    "scalar" (isotonic/platt/shrinkage are all 1-D fits over the self-
    consistency scalar); feature_set == "multi" always uses the multi-feature
    logistic regression from src.gemini_calibration_features, so
    calibrator_family is fixed to "logistic_multi" there instead."""
    configs = []
    for agreement in ("hard", "soft"):
        for penalty in (False, True):
            for family in SCALAR_CALIBRATOR_FAMILIES:
                configs.append({
                    "agreement": agreement, "feature_set": "scalar",
                    "calibrator_family": family, "penalty": penalty,
                })
            configs.append({
                "agreement": agreement, "feature_set": "multi",
                "calibrator_family": "logistic_multi", "penalty": penalty,
            })
    return configs


def _label(config):
    return (f"agreement={config['agreement']:<4} feature_set={config['feature_set']:<6} "
            f"family={config['calibrator_family']:<14} penalty={config['penalty']}")


def _prepare_context(images_dir=IMAGES_DIR):
    ground_truth_df = pd.read_csv(TRAIN_CSV_PATH)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    image_files = ground_truth_df["image_file"].tolist()
    predictions_by_crop = {
        "hard": load_predictions_by_crop(DEV_PREDICTIONS_PATH),
        "soft": rescored_predictions_by_crop(DEV_PREDICTIONS_PATH, soft_consistency_scores),
    }
    contributing_crops = contributing_crops_by_specimen(manifest_by_image, predictions_by_crop["hard"], image_files)
    return {
        "ground_truth_df": ground_truth_df, "manifest_by_image": manifest_by_image,
        "image_files": image_files, "predictions_by_crop": predictions_by_crop,
        "wholeimage_predictions": load_wholeimage_predictions(), "contributing_crops": contributing_crops,
        "images_dir": images_dir,
    }


def _with_fallback(predictions_by_crop_variant, context):
    merged = merge_predictions_by_specimen(
        predictions_by_crop_variant, context["manifest_by_image"], context["image_files"]
    )
    cleaned = clean_merged_predictions(merged)
    return clean_merged_predictions(apply_wholeimage_fallback(cleaned, context["wholeimage_predictions"]))


def _errors_by_field(with_fallback, ground_truth_df):
    errors = {field: [] for field in FIELD_IS_DATE}
    for _, r in ground_truth_df.iterrows():
        for field, is_date in FIELD_IS_DATE.items():
            text, _ = with_fallback[r["image_file"]][field]
            errors[field].append(best_pipe_ned(text, str(r[field]), is_date=is_date))
    return errors


def evaluate_config(config, context):
    predictions_by_crop_variant = context["predictions_by_crop"][config["agreement"]]
    with_fallback = _with_fallback(predictions_by_crop_variant, context)
    ground_truth_df = context["ground_truth_df"]
    image_files = context["image_files"]
    errors_by_field = _errors_by_field(with_fallback, ground_truth_df)

    confidences_by_field = {}
    if config["feature_set"] == "scalar":
        confidences_by_field = oof_calibrated_confidences(
            with_fallback, ground_truth_df, family=config["calibrator_family"]
        )
    else:
        for field in FIELD_IS_DATE:
            rows = build_feature_rows(
                image_files, with_fallback, context["contributing_crops"], predictions_by_crop_variant,
                images_dir=context["images_dir"],
            )[field]
            correctness = [1.0 - e for e in errors_by_field[field]]
            confidences_by_field[field] = oof_multi_feature_calibrated_confidences(rows, correctness)

    if config["penalty"]:
        for field in FIELD_IS_DATE:
            penalized = []
            for i, image_file in enumerate(image_files):
                n_proc, n_contrib = count_processed_and_contributing(
                    image_file, field, context["manifest_by_image"], predictions_by_crop_variant
                )
                penalized.append(apply_confidence_penalty(confidences_by_field[field][i], n_proc, n_contrib))
            confidences_by_field[field] = penalized

    aurcs = {field: aurc(errors_by_field[field], confidences_by_field[field]) for field in FIELD_IS_DATE}
    return sum(aurcs.values()) / len(aurcs), aurcs


def main():
    context = _prepare_context()
    results = []
    for config in enumerate_configs():
        overall, per_field = evaluate_config(config, context)
        results.append((overall, per_field, config))
    results.sort(key=lambda r: r[0])

    print(f"{'overall_aurc':>12}  {'date_aurc':>10}  {'locality_aurc':>14}  config")
    for overall, per_field, config in results:
        print(f"{overall:12.4f}  {per_field['verbatimDate']:10.4f}  "
              f"{per_field['verbatimLocality']:14.4f}  {_label(config)}")

    best_overall, best_per_field, best_config = results[0]
    print(f"\nbest: {_label(best_config)} -> overall {best_overall:.4f} "
          f"(current best baseline {BASELINE_AURC:.4f}, delta {best_overall - BASELINE_AURC:+.4f})")


if __name__ == "__main__":
    main()
