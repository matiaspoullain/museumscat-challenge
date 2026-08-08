"""Deterministic confidence penalty for a structural risk pattern mined via
src.analyze_confidence_errors on the dev set: when a specimen has >=2 Gemini-
processed card crops but at least one of them didn't contribute a value to a
given field, that field's transcription is at real risk of having missed a
second card's value entirely -- self-consistency alone can't see this, since it
only measures agreement across resamples of the SAME card, never coverage
across cards. Among dev-set predictions with confidence >= 0.9, this pattern's
wrong-rate is 34.3% (12/35) vs. 18.1% (31/171) when it doesn't apply -- roughly
1.9x (mined 2026-08-07, see docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md).

Kept as a fixed multiplicative penalty rather than folded into
src.gemini_calibration_features' learned multi-feature calibrator, so the two
approaches (hardcoded rule vs. fit-on-dev-data) can be compared head to head."""

PENALTY_FACTOR = 0.7


def count_processed_and_contributing(image_file, field, manifest_by_image, predictions_by_crop):
    n_processed = 0
    n_contributing = 0
    for crop_filename in manifest_by_image.get(image_file, []):
        card = predictions_by_crop.get(crop_filename)
        if card is None:
            continue
        n_processed += 1
        text, _ = card[field]
        if text != "MISSING":
            n_contributing += 1
    return n_processed, n_contributing


def apply_confidence_penalty(confidence, n_processed, n_contributing, penalty_factor = PENALTY_FACTOR):
    flagged = n_processed >= 2 and n_contributing < n_processed
    return confidence * penalty_factor if flagged else confidence
