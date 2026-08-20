"""Does Track A3's local-VLM agreement signal (src.local_vlm_signal) actually
need the dev pilot's N_SAMPLES=10 self-consistency vote, or does a cheaper
sample count give the same within-tier NED gradient?

src.local_vlm_signal.local_vlm_agreement_score only ever reads the merged
*text* of each crop's majority-vote sample -- it never reads the local VLM's
own confidence. That means the 10x per-crop inference cost the dev pilot and
the (user-cancelled) Kaggle test-set run paid for is entirely spent computing
a self-consistency confidence that Track A3 then throws away. This script
answers "how few samples can we get away with" by re-deriving the majority
vote from the first N of each row's already-collected 10 raw samples
(data/processed/gemini/kaggle_vlm_spike_dataset/previous_card_results.csv) --
zero GPU, zero new inference, pure re-analysis of samples already paid for.

Usage:
    conda run -n museumscat-challenge python -m src.analyze_local_vlm_sample_count
"""
import csv
import json
import os

import pandas as pd

from src.calibrate_gemini_confidence import (
    apply_wholeimage_fallback,
    load_predictions_by_crop,
    load_wholeimage_predictions,
    merge_predictions_by_specimen,
)
from src.clean_gemini_predictions import clean_merged_predictions
from src.gemini_transcribe import self_consistency_scores
from src.local_vlm_signal import DEV_LOCAL_VLM_CARD_PATH, covered_crops_by_specimen, local_vlm_agreement_score
from src.metrics import best_pipe_ned
from src.pipeline import load_manifest

TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
DEV_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "dev_card_predictions.csv")
DEV_WHOLEIMAGE_PATH = os.path.join("data", "processed", "gemini", "dev_wholeimage_predictions.csv")

FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}
N_SAMPLE_COUNTS = [1, 3, 5, 10]


def truncated_local_vlm_predictions_by_crop(path=DEV_LOCAL_VLM_CARD_PATH, n_samples=10):
    """Re-derives {crop_filename: {field: (text, self_consistency_confidence)}}
    using only the first n_samples of each row's raw_samples_json, instead of
    the full 10 the pilot collected -- simulates "what if we'd only sampled
    N times" without spending any new GPU inference. Same missing-file
    convention as src.local_vlm_signal.build_local_vlm_predictions_by_crop
    (returns {} rather than raising)."""
    if not os.path.exists(path):
        return {}
    by_crop = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples = json.loads(row["raw_samples_json"])[:n_samples]
            by_crop[row["crop_filename"]] = self_consistency_scores(samples)
    return by_crop


def agreement_gradient_for_n(field, image_files, gemini_confidences, gemini_texts, errors,
                              manifest_by_image, local_vlm_predictions_by_crop):
    """Within Gemini's own top confidence tier (max(gemini_confidences)),
    splits dev specimens by local-VLM agreement and reports each side's mean
    NED -- the same shape as the 21/25 vs 20/23 gradient that justified
    Track A3 at n_samples=10, so it's directly comparable across N."""
    covered_crops = covered_crops_by_specimen(manifest_by_image, local_vlm_predictions_by_crop)
    local_merged = merge_predictions_by_specimen(local_vlm_predictions_by_crop, manifest_by_image, image_files)

    top_tier = max(gemini_confidences)
    agree_errors, disagree_errors = [], []
    for image_file, confidence, gemini_text, error in zip(image_files, gemini_confidences, gemini_texts, errors):
        if confidence != top_tier:
            continue
        agreement = local_vlm_agreement_score(image_file, field, gemini_text, local_merged, covered_crops)
        if agreement == 1.0:
            agree_errors.append(error)
        elif agreement == 0.0:
            disagree_errors.append(error)
    return {
        "agree_n": len(agree_errors),
        "agree_mean_ned": sum(agree_errors) / len(agree_errors) if agree_errors else None,
        "disagree_n": len(disagree_errors),
        "disagree_mean_ned": sum(disagree_errors) / len(disagree_errors) if disagree_errors else None,
    }


def _dev_gemini_context():
    ground_truth_df = pd.read_csv(TRAIN_CSV_PATH)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    predictions_by_crop = load_predictions_by_crop(DEV_PREDICTIONS_PATH)
    wholeimage_predictions = load_wholeimage_predictions(DEV_WHOLEIMAGE_PATH)
    image_files = ground_truth_df["image_file"].tolist()

    merged = merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, image_files)
    merged = clean_merged_predictions(merged)
    merged = clean_merged_predictions(apply_wholeimage_fallback(merged, wholeimage_predictions))

    by_field = {}
    for field, is_date in FIELD_IS_DATE.items():
        confidences, texts, errors = [], [], []
        for _, row in ground_truth_df.iterrows():
            text, confidence = merged[row["image_file"]][field]
            confidences.append(confidence)
            texts.append(text)
            errors.append(best_pipe_ned(text, str(row[field]), is_date=is_date))
        by_field[field] = {"confidences": confidences, "texts": texts, "errors": errors}
    return manifest_by_image, image_files, by_field


def main():
    manifest_by_image, image_files, by_field = _dev_gemini_context()

    print("=== local VLM sample-count sensitivity (dev, within Gemini's top confidence tier) ===")
    for field in FIELD_IS_DATE:
        print(f"\n-- {field} --")
        data = by_field[field]
        for n in N_SAMPLE_COUNTS:
            local_vlm_predictions_by_crop = truncated_local_vlm_predictions_by_crop(n_samples=n)
            if not local_vlm_predictions_by_crop:
                print("   no local VLM dev data found, skipping")
                break
            row = agreement_gradient_for_n(
                field, image_files, data["confidences"], data["texts"], data["errors"],
                manifest_by_image, local_vlm_predictions_by_crop,
            )
            agree_str = f"{row['agree_mean_ned']:.3f}" if row["agree_mean_ned"] is not None else "n/a"
            disagree_str = f"{row['disagree_mean_ned']:.3f}" if row["disagree_mean_ned"] is not None else "n/a"
            print(f"   n_samples={n:<3} agree: n={row['agree_n']:<3} mean_ned={agree_str:<6}  "
                  f"disagree: n={row['disagree_n']:<3} mean_ned={disagree_str}")


if __name__ == "__main__":
    main()
