"""Builds data/processed/submission_gemini.csv from the Gemini per-card
test-set predictions (src.run_gemini_test's output): merges per-specimen
(src.calibrate_gemini_confidence.merge_predictions_by_specimen), applies
src.clean_gemini_predictions' text cleanup, overlays the whole-image fallback
(src.calibrate_gemini_confidence.apply_wholeimage_fallback) for specimens
missing both fields, and applies src.confidence_penalty_rules' fixed
under-coverage penalty. This is the winning config from
docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md
(dev AURC 0.0715): raw self-consistency confidence (hard agreement, i.e.
src.gemini_transcribe.self_consistency_scores -- already how
src.run_gemini_test's predictions CSV is scored, so no rescoring needed here),
no fitted calibrator.

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

--apply-rank-refinement layers src.confidence_rank_rules on top (Track A,
docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md): with
n_samples=3, raw self-consistency only takes ~4 distinct values, so roughly
half of all rows tie at confidence 1.0 and Python's stable sort resolves
those ties by arbitrary CSV row order -- oracle reordering *within existing
ties alone* (no text change) took dev AURC 0.0715 -> 0.0456. This flag adds a
structural-MISSING tier plus a soft-consistency/corpus-support within-tier
tiebreaker, taking dev AURC to 0.0575, while leaving transcribed text
byte-identical to the non-refined submission -- writes to
RANK_REFINED_OUTPUT_PATH, a separate file, so the two can be diffed and
submitted independently.

Usage:
    conda run -n museumscat-challenge python -m src.generate_submission_gemini
    conda run -n museumscat-challenge python -m src.generate_submission_gemini --apply-rank-refinement
"""
import os

import pandas as pd

from src.calibrate_gemini_confidence import (
    apply_wholeimage_fallback,
    load_predictions_by_crop,
    load_wholeimage_predictions,
    merge_predictions_by_specimen,
    rescored_predictions_by_crop,
)
from src.clean_gemini_predictions import clean_merged_predictions
from src.confidence_penalty_rules import apply_confidence_penalty, count_processed_and_contributing
from src.confidence_rank_rules import apply_structural_missing_tier, combine_tier_and_within_tier_score, within_tier_score
from src.gemini_transcribe import soft_consistency_scores
from src.generate_submission import REQUIRED_COLUMNS, validate_submission
from src.local_vlm_signal import (
    build_local_vlm_predictions_by_crop,
    covered_crops_by_specimen,
    local_vlm_agreement_score,
)
from src.locality_corpus import build_locality_corpus
from src.pipeline import load_manifest

TEST_CSV_PATH = os.path.join("data", "raw", "test.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
TEST_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "test_card_predictions.csv")
TEST_WHOLEIMAGE_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini",
                                                 "test_wholeimage_predictions.csv")
OUTPUT_PATH = os.path.join("data", "processed", "submission_gemini.csv")
# Track A (docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md):
# confidence-only refinement, transcribed text byte-identical to OUTPUT_PATH.
# Kept as a separate output file/flag rather than replacing OUTPUT_PATH so the
# two can be diffed and submitted independently -- this pipeline's convention
# after two prior spikes both regressed the real leaderboard: attribute any
# leaderboard movement to one isolated change at a time.
RANK_REFINED_OUTPUT_PATH = os.path.join("data", "processed", "submission_gemini_rank_refined.csv")
# Track A3: rank refinement + local VLM agreement (src.local_vlm_signal),
# once test-set local VLM predictions exist (they don't yet as of this
# writing -- see the Kaggle GPU run status in
# docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md). Also text-
# identical to OUTPUT_PATH; a third separate file for the same attribution
# reason as RANK_REFINED_OUTPUT_PATH.
RANK_REFINED_WITH_LOCAL_VLM_OUTPUT_PATH = os.path.join(
    "data", "processed", "submission_gemini_rank_refined_local_vlm.csv"
)

FIELDS = ("verbatimDate", "verbatimLocality")


def build_submission_df(image_files, manifest_by_image, predictions_by_crop,
                         wholeimage_predictions_by_specimen=None,
                         apply_rank_refinement=False, soft_predictions_by_crop=None,
                         locality_corpus=None, local_vlm_predictions_by_crop=None):
    """apply_rank_refinement=True layers src.confidence_rank_rules on top of
    the production merge/penalty pipeline: the structural-MISSING tier, then
    within_tier_score's soft-consistency (+ corpus-support for
    verbatimLocality, + src.local_vlm_signal agreement if
    local_vlm_predictions_by_crop is given) as a bounded within-tier
    tiebreaker (combine_tier_and_within_tier_score). Text output is
    completely unaffected -- only the two *_confidence columns can change --
    since every rank-refinement step operates on confidence alone, never on
    `text`. soft_predictions_by_crop (src.calibrate_gemini_confidence.
    rescored_predictions_by_crop(path, src.gemini_transcribe.
    soft_consistency_scores)) and locality_corpus (src.locality_corpus.
    build_locality_corpus()) are required for the base refinement to
    actually apply; omitting them with apply_rank_refinement=True still runs
    (falls back to the structural-missing tier alone) but is not the shipped
    configuration -- callers should pass both. local_vlm_predictions_by_crop
    (src.local_vlm_signal.build_local_vlm_predictions_by_crop's output) is
    optional on top of that; omitting it (the default) is equivalent to no
    local VLM run being available for this split, which degrades gracefully
    (local_vlm_agreement_score's neutral 0.5)."""
    merged = merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, image_files)
    cleaned = clean_merged_predictions(merged)
    if wholeimage_predictions_by_specimen is not None:
        cleaned = clean_merged_predictions(
            apply_wholeimage_fallback(cleaned, wholeimage_predictions_by_specimen)
        )

    soft_merged = None
    if apply_rank_refinement and soft_predictions_by_crop is not None:
        soft_merged = merge_predictions_by_specimen(soft_predictions_by_crop, manifest_by_image, image_files)
        soft_merged = clean_merged_predictions(soft_merged)
        if wholeimage_predictions_by_specimen is not None:
            soft_merged = clean_merged_predictions(
                apply_wholeimage_fallback(soft_merged, wholeimage_predictions_by_specimen)
            )

    local_vlm_merged, local_vlm_covered_crops = None, None
    if apply_rank_refinement and local_vlm_predictions_by_crop:
        local_vlm_merged = merge_predictions_by_specimen(
            local_vlm_predictions_by_crop, manifest_by_image, image_files
        )
        local_vlm_covered_crops = covered_crops_by_specimen(manifest_by_image, local_vlm_predictions_by_crop)

    rows = []
    for image_file in image_files:
        row = {"image_file": image_file}
        for field in FIELDS:
            text, confidence = cleaned[image_file][field]
            n_processed, n_contributing = count_processed_and_contributing(
                image_file, field, manifest_by_image, predictions_by_crop
            )
            confidence = apply_confidence_penalty(confidence, n_processed, n_contributing)
            if apply_rank_refinement:
                confidence = apply_structural_missing_tier(confidence, text, n_processed, n_contributing)
                if soft_merged is not None:
                    soft_confidence = soft_merged[image_file][field][1]
                    agreement = None
                    if local_vlm_merged is not None:
                        agreement = local_vlm_agreement_score(
                            image_file, field, text, local_vlm_merged, local_vlm_covered_crops
                        )
                    score = within_tier_score(field, text, soft_confidence, locality_corpus, agreement)
                    confidence = combine_tier_and_within_tier_score(confidence, score)
            row[field] = text
            row[f"{field}_confidence"] = confidence
        rows.append(row)
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def generate_submission_gemini(test_csv_path=TEST_CSV_PATH, manifest_path=MANIFEST_PATH,
                                predictions_path=TEST_PREDICTIONS_PATH,
                                wholeimage_predictions_path=TEST_WHOLEIMAGE_PREDICTIONS_PATH,
                                output_path=OUTPUT_PATH, apply_rank_refinement=False,
                                local_vlm_predictions_path=None):
    image_files = pd.read_csv(test_csv_path)["image_file"].tolist()
    manifest_by_image = load_manifest(manifest_path)
    predictions_by_crop = load_predictions_by_crop(predictions_path)
    wholeimage_predictions = load_wholeimage_predictions(wholeimage_predictions_path)

    soft_predictions_by_crop = None
    locality_corpus = None
    local_vlm_predictions_by_crop = None
    if apply_rank_refinement:
        soft_predictions_by_crop = rescored_predictions_by_crop(predictions_path, soft_consistency_scores)
        locality_corpus = build_locality_corpus()
        if local_vlm_predictions_path is not None:
            local_vlm_predictions_by_crop = build_local_vlm_predictions_by_crop(local_vlm_predictions_path)

    out_df = build_submission_df(
        image_files, manifest_by_image, predictions_by_crop, wholeimage_predictions,
        apply_rank_refinement=apply_rank_refinement,
        soft_predictions_by_crop=soft_predictions_by_crop, locality_corpus=locality_corpus,
        local_vlm_predictions_by_crop=local_vlm_predictions_by_crop,
    )
    errors = validate_submission(out_df, set(image_files))
    if errors:
        raise ValueError(f"submission failed validation: {errors}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-rank-refinement", action="store_true",
                         help="Layer src.confidence_rank_rules on top; writes to RANK_REFINED_OUTPUT_PATH "
                              "instead of OUTPUT_PATH, text unchanged. See module docstring.")
    parser.add_argument("--local-vlm-predictions-path", default=None,
                         help="Also blend in src.local_vlm_signal agreement (requires "
                              "--apply-rank-refinement); writes to "
                              "RANK_REFINED_WITH_LOCAL_VLM_OUTPUT_PATH instead.")
    args = parser.parse_args()
    if args.local_vlm_predictions_path:
        generate_submission_gemini(
            output_path=RANK_REFINED_WITH_LOCAL_VLM_OUTPUT_PATH, apply_rank_refinement=True,
            local_vlm_predictions_path=args.local_vlm_predictions_path,
        )
    elif args.apply_rank_refinement:
        generate_submission_gemini(output_path=RANK_REFINED_OUTPUT_PATH, apply_rank_refinement=True)
    else:
        generate_submission_gemini()
