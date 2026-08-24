"""Dev-set go/no-go gate for the Gemini per-card VLM approach: computes AURC
(same methodology as src.old.evaluate_local) with (a) raw self-consistency
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
    apply_wholeimage_fallback,
    load_predictions_by_crop,
    load_wholeimage_predictions,
    merge_predictions_by_specimen,
    oof_calibrated_confidences,
    rescored_predictions_by_crop,
)
from src.clean_gemini_predictions import clean_merged_predictions
from src.confidence_penalty_rules import apply_confidence_penalty, count_processed_and_contributing
from src.confidence_rank_rules import apply_structural_missing_tier, combine_tier_and_within_tier_score, within_tier_score
from src.gemini_transcribe import soft_consistency_scores
from src.local_vlm_signal import (
    build_local_vlm_predictions_by_crop,
    covered_crops_by_specimen,
    local_vlm_agreement_score,
)
from src.locality_corpus import build_locality_corpus
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

    wholeimage_predictions = load_wholeimage_predictions()
    with_fallback = clean_merged_predictions(apply_wholeimage_fallback(cleaned, wholeimage_predictions))
    fallback_errors_by_field = _errors_by_field(with_fallback, ground_truth_df)
    fallback_confidences = {
        field: [with_fallback[row["image_file"]][field][1] for _, row in ground_truth_df.iterrows()]
        for field in FIELD_IS_DATE
    }
    _print_report("raw confidence + cleanup + whole-image fallback",
                  fallback_errors_by_field, fallback_confidences)

    penalized_confidences = {field: [] for field in FIELD_IS_DATE}
    for field in FIELD_IS_DATE:
        for _, row in ground_truth_df.iterrows():
            n_processed, n_contributing = count_processed_and_contributing(
                row["image_file"], field, manifest_by_image, predictions_by_crop
            )
            raw_confidence = with_fallback[row["image_file"]][field][1]
            penalized_confidences[field].append(
                apply_confidence_penalty(raw_confidence, n_processed, n_contributing)
            )
    _print_report("raw confidence + cleanup + whole-image fallback + coverage penalty (production config)",
                  fallback_errors_by_field, penalized_confidences)

    # Track A: src.confidence_rank_rules layered on top of the production
    # config -- structural-MISSING tier + soft-consistency/corpus-support
    # within-tier tiebreak. Text is unaffected (fallback_errors_by_field is
    # reused as-is); only confidence changes. See
    # docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md.
    soft_predictions_by_crop = rescored_predictions_by_crop(
        os.path.join("data", "processed", "gemini", "dev_card_predictions.csv"), soft_consistency_scores
    )
    soft_merged = merge_predictions_by_specimen(
        soft_predictions_by_crop, manifest_by_image, ground_truth_df["image_file"].tolist()
    )
    soft_merged = clean_merged_predictions(soft_merged)
    soft_merged = clean_merged_predictions(apply_wholeimage_fallback(soft_merged, wholeimage_predictions))
    locality_corpus = build_locality_corpus()

    rank_refined_confidences = {field: [] for field in FIELD_IS_DATE}
    for field in FIELD_IS_DATE:
        for _, row in ground_truth_df.iterrows():
            image_file = row["image_file"]
            text, _ = with_fallback[image_file][field]
            n_processed, n_contributing = count_processed_and_contributing(
                image_file, field, manifest_by_image, predictions_by_crop
            )
            confidence = penalized_confidences[field][len(rank_refined_confidences[field])]
            confidence = apply_structural_missing_tier(confidence, text, n_processed, n_contributing)
            soft_confidence = soft_merged[image_file][field][1]
            score = within_tier_score(field, text, soft_confidence, locality_corpus)
            confidence = combine_tier_and_within_tier_score(confidence, score)
            rank_refined_confidences[field].append(confidence)
    _print_report(
        "production config + confidence rank refinement (Track A: structural-MISSING tier "
        "+ soft-consistency/corpus-support within-tier tiebreak, text unchanged)",
        fallback_errors_by_field, rank_refined_confidences,
    )

    # Track A3: local VLM (Qwen2.5-VL) agreement, re-purposed as a within-tier
    # signal rather than the closed spike's blend/gate designs -- zero
    # additional Gemini quota, reuses the already-collected 152/197 dev crops
    # in data/processed/gemini/kaggle_vlm_spike_dataset/previous_card_results.csv.
    local_predictions_by_crop = build_local_vlm_predictions_by_crop()
    if local_predictions_by_crop:
        local_covered_crops = covered_crops_by_specimen(manifest_by_image, local_predictions_by_crop)
        local_merged = merge_predictions_by_specimen(
            local_predictions_by_crop, manifest_by_image, ground_truth_df["image_file"].tolist()
        )
        with_local_vlm_confidences = {field: [] for field in FIELD_IS_DATE}
        for field in FIELD_IS_DATE:
            for i, (_, row) in enumerate(ground_truth_df.iterrows()):
                image_file = row["image_file"]
                text, _ = with_fallback[image_file][field]
                n_processed, n_contributing = count_processed_and_contributing(
                    image_file, field, manifest_by_image, predictions_by_crop
                )
                confidence = penalized_confidences[field][i]
                confidence = apply_structural_missing_tier(confidence, text, n_processed, n_contributing)
                soft_confidence = soft_merged[image_file][field][1]
                agreement = local_vlm_agreement_score(image_file, field, text, local_merged, local_covered_crops)
                score = within_tier_score(field, text, soft_confidence, locality_corpus, agreement)
                confidence = combine_tier_and_within_tier_score(confidence, score)
                with_local_vlm_confidences[field].append(confidence)
        _print_report(
            "production config + rank refinement + local VLM agreement (Track A3, 152/197 dev "
            "crop coverage, text unchanged)",
            fallback_errors_by_field, with_local_vlm_confidences,
        )
    else:
        print("--- local VLM dev data not found, skipping Track A3 report ---")


if __name__ == "__main__":
    main()
