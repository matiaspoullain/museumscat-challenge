"""Mining companion to src.old.analyze_gemini_errors: instead of text-diff
patterns, finds (specimen, field) pairs where confidence is high but the
transcription is still wrong -- the cases AURC punishes hardest, and the
evidence behind src.confidence_penalty_rules' fixed penalty.

Usage:
    conda run -n museumscat-challenge python -m src.old.analyze_confidence_errors
"""
import os

import pandas as pd

from src.calibrate_gemini_confidence import (
    apply_wholeimage_fallback,
    load_predictions_by_crop,
    load_wholeimage_predictions,
    merge_predictions_by_specimen,
)
from src.clean_gemini_predictions import clean_merged_predictions
from src.metrics import CORRECT_THRESHOLD, best_pipe_ned
from src.pipeline import load_manifest

TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}
CONFIDENCE_THRESHOLD = 0.9


def high_confidence_errors(merged, ground_truth_df, confidence_threshold=CONFIDENCE_THRESHOLD,
                            error_threshold=CORRECT_THRESHOLD):
    rows = []
    for _, r in ground_truth_df.iterrows():
        for field, is_date in FIELD_IS_DATE.items():
            text, confidence = merged[r["image_file"]][field]
            error = best_pipe_ned(text, str(r[field]), is_date=is_date)
            if confidence >= confidence_threshold and error > error_threshold:
                rows.append({
                    "image_file": r["image_file"], "field": field, "confidence": confidence,
                    "prediction": text, "truth": str(r[field]), "error": error,
                })
    rows.sort(key=lambda row: -row["confidence"])
    return rows


def main():
    ground_truth_df = pd.read_csv(TRAIN_CSV_PATH)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    predictions_by_crop = load_predictions_by_crop()
    merged = merge_predictions_by_specimen(
        predictions_by_crop, manifest_by_image, ground_truth_df["image_file"].tolist()
    )
    cleaned = clean_merged_predictions(merged)
    wholeimage_predictions = load_wholeimage_predictions()
    with_fallback = clean_merged_predictions(apply_wholeimage_fallback(cleaned, wholeimage_predictions))

    rows = high_confidence_errors(with_fallback, ground_truth_df)
    print(f"{len(rows)} high-confidence (>={CONFIDENCE_THRESHOLD}) wrong cases")
    for row in rows:
        print(f"{row['image_file']} {row['field']} conf={row['confidence']:.2f} "
              f"pred={row['prediction']!r} truth={row['truth']!r} err={row['error']:.2f}")


if __name__ == "__main__":
    main()
