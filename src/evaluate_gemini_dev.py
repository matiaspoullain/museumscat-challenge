"""Dev-set go/no-go gate for the Gemini per-card VLM approach: computes AURC
(same methodology as src.evaluate_local) with (a) raw self-consistency
confidence, (b) raw confidence plus src.clean_gemini_predictions' text cleanup,
and (c) honest out-of-fold isotonic-calibrated confidence, printed against the
classical doctr pipeline's documented dev-set baseline (see
docs/superpowers/plans/2026-07-29-submission-pipeline.md's Outcome section).

Only run src.run_gemini_test on the full 3300-image test set if this shows an
improvement over the baseline -- that's the whole point of doing the cheap
200-image dev run first.

Usage:
    conda run -n museumscat-challenge python -m src.evaluate_gemini_dev
"""
import os

import pandas as pd

from src.calibrate_gemini_confidence import (
    load_predictions_by_crop,
    merge_predictions_by_specimen,
    oof_calibrated_confidences,
)
from src.clean_gemini_predictions import clean_merged_predictions
from src.metrics import aurc, best_pipe_ned
from src.pipeline import load_manifest

TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")

FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}

# Classical doctr pipeline's documented dev-set result (with visual card-relevance
# filter, its best local config) -- docs/superpowers/plans/2026-07-29-submission-pipeline.md
BASELINE_AURC = {"verbatimDate": 0.538, "verbatimLocality": 0.613}
BASELINE_OVERALL_AURC = 0.576


def _errors_by_field(merged, ground_truth_df):
    errors = {field: [] for field in FIELD_IS_DATE}
    for _, row in ground_truth_df.iterrows():
        for field, is_date in FIELD_IS_DATE.items():
            text, _ = merged[row["image_file"]][field]
            errors[field].append(best_pipe_ned(text, str(row[field]), is_date=is_date))
    return errors


def _print_report(label, errors_by_field, confidences_by_field):
    print(f"--- {label} ---")
    aurcs = {}
    for field in FIELD_IS_DATE:
        field_aurc = aurc(errors_by_field[field], confidences_by_field[field])
        aurcs[field] = field_aurc
        mean_ned = sum(errors_by_field[field]) / len(errors_by_field[field])
        delta = field_aurc - BASELINE_AURC[field]
        print(f"{field}: mean_ned={mean_ned:.3f} aurc={field_aurc:.3f} "
              f"(baseline {BASELINE_AURC[field]:.3f}, delta {delta:+.3f})")
    overall = sum(aurcs.values()) / len(aurcs)
    delta = overall - BASELINE_OVERALL_AURC
    print(f"overall AURC: {overall:.3f} (baseline {BASELINE_OVERALL_AURC:.3f}, delta {delta:+.3f})")
    return overall


def main():
    ground_truth_df = pd.read_csv(TRAIN_CSV_PATH)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    predictions_by_crop = load_predictions_by_crop()
    merged = merge_predictions_by_specimen(
        predictions_by_crop, manifest_by_image, ground_truth_df["image_file"].tolist()
    )

    errors_by_field = _errors_by_field(merged, ground_truth_df)
    raw_confidences = {
        field: [merged[row["image_file"]][field][1] for _, row in ground_truth_df.iterrows()]
        for field in FIELD_IS_DATE
    }
    _print_report("raw self-consistency confidence", errors_by_field, raw_confidences)

    cleaned = clean_merged_predictions(merged)
    cleaned_errors_by_field = _errors_by_field(cleaned, ground_truth_df)
    _print_report("raw confidence + text cleanup", cleaned_errors_by_field, raw_confidences)

    calibrated_confidences = oof_calibrated_confidences(merged, ground_truth_df)
    _print_report("out-of-fold calibrated confidence", errors_by_field, calibrated_confidences)


if __name__ == "__main__":
    main()
