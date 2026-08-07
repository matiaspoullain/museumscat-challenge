"""Builds data/processed/submission_gemini.csv from the Gemini per-card
test-set predictions (src.run_gemini_test's output): merges per-specimen
(src.calibrate_gemini_confidence.merge_predictions_by_specimen) and applies
src.clean_gemini_predictions' text cleanup.

Confidence is left as raw self-consistency, not isotonic-calibrated -- dev-set
evaluation (src.evaluate_gemini_dev) showed calibration make AURC *worse*
here (0.146 vs 0.124 raw+cleanup): AURC only depends on confidence rank, and
self-consistency's few distinct values (only n_samples+1 possible levels)
give isotonic regression no ties left to usefully break, while its 5-fold
out-of-fold evaluation introduces small rank inconsistencies across fold
boundaries. src.calibrate_gemini_confidence is kept for future recalibration
if that changes (e.g. with continuous-valued confidence), just not used here.

Reuses src.generate_submission.validate_submission so both the classical and
Gemini submissions pass the exact same format checks.

Usage:
    conda run -n museumscat-challenge python -m src.generate_submission_gemini
"""
import os

import pandas as pd

from src.calibrate_gemini_confidence import load_predictions_by_crop, merge_predictions_by_specimen
from src.clean_gemini_predictions import clean_merged_predictions
from src.generate_submission import REQUIRED_COLUMNS, validate_submission
from src.pipeline import load_manifest

TEST_CSV_PATH = os.path.join("data", "raw", "test.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
TEST_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "test_card_predictions.csv")
OUTPUT_PATH = os.path.join("data", "processed", "submission_gemini.csv")

FIELDS = ("verbatimDate", "verbatimLocality")


def build_submission_df(image_files, manifest_by_image, predictions_by_crop):
    merged = merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, image_files)
    cleaned = clean_merged_predictions(merged)
    rows = []
    for image_file in image_files:
        row = {"image_file": image_file}
        for field in FIELDS:
            text, confidence = cleaned[image_file][field]
            row[field] = text
            row[f"{field}_confidence"] = confidence
        rows.append(row)
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def generate_submission_gemini(test_csv_path=TEST_CSV_PATH, manifest_path=MANIFEST_PATH,
                                predictions_path=TEST_PREDICTIONS_PATH, output_path=OUTPUT_PATH):
    image_files = pd.read_csv(test_csv_path)["image_file"].tolist()
    manifest_by_image = load_manifest(manifest_path)
    predictions_by_crop = load_predictions_by_crop(predictions_path)

    out_df = build_submission_df(image_files, manifest_by_image, predictions_by_crop)
    errors = validate_submission(out_df, set(image_files))
    if errors:
        raise ValueError(f"submission failed validation: {errors}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    generate_submission_gemini()
