"""Builds a per-(specimen, field) feature vector for a multi-feature confidence
calibrator. Self-consistency alone is a single scalar; this adds structural
signals already implicit in the merge (how many cards contributed, whether they
agreed, whether the value came from the whole-image fallback) and per-card
visual features (src.eda_card_visual_features -- color/texture/border, built
for the card-relevance classifier but equally relevant to legibility: an older,
faded card should look different from a crisp modern one, and that difference
should predict how trustworthy its transcription is)."""
import os

import cv2
import numpy as np

from src.eda_card_visual_features import FEATURE_NAMES, IMAGES_DIR, extract_features

FIELDS = ("verbatimDate", "verbatimLocality")
STRUCTURAL_FEATURE_NAMES = [
    "text_length", "n_cards", "cross_card_agreement", "from_wholeimage_fallback", "has_visual_features",
]
FEATURE_ORDER = ["self_consistency"] + STRUCTURAL_FEATURE_NAMES + FEATURE_NAMES


def contributing_crops_by_specimen(manifest_by_image, predictions_by_crop, image_files):
    """image_file -> {field: [crop_filename, ...]} for crops whose text for that
    field isn't "MISSING" -- the same filter src.calibrate_gemini_confidence.
    merge_predictions_by_specimen applies when assembling merged text, but this
    keeps the individual crop filenames instead of collapsing them into one
    string, since build_feature_rows needs the source crop images."""
    result = {}
    for image_file in image_files:
        per_field = {field: [] for field in FIELDS}
        for crop_filename in manifest_by_image.get(image_file, []):
            card = predictions_by_crop.get(crop_filename)
            if card is None:
                continue
            for field in FIELDS:
                text, _ = card[field]
                if text != "MISSING":
                    per_field[field].append(crop_filename)
        result[image_file] = per_field
    return result


def _visual_features_for_crops(crop_filenames, images_dir=IMAGES_DIR):
    """Mean of extract_features across crop_filenames. Returns (None, False)
    when there's nothing to average, so the caller fills NaN for the pipeline's
    SimpleImputer rather than guessing a default value."""
    per_crop = []
    for crop_filename in crop_filenames:
        image_bgr = cv2.imread(os.path.join(images_dir, crop_filename))
        if image_bgr is None:
            continue
        per_crop.append(extract_features(image_bgr))
    if not per_crop:
        return None, False
    means = {name: float(np.mean([f[name] for f in per_crop])) for name in FEATURE_NAMES}
    return means, True


def build_feature_rows(image_files, merged, contributing_crops, predictions_by_crop, images_dir=IMAGES_DIR):
    """One row per (image_file, field), as {feature_name: value}, aligned to
    image_files' order -- callers zip it against a same-order correctness list.
    "self_consistency" is read straight from merged[image_file][field][1]: the
    confidence merge_predictions_by_specimen already averaged across cards,
    computed from whichever scorer (hard or soft) produced the underlying
    per-card predictions -- this module doesn't need to know which."""
    rows = {field: [] for field in FIELDS}
    for image_file in image_files:
        for field in FIELDS:
            text, self_consistency = merged[image_file][field]
            crops = contributing_crops.get(image_file, {}).get(field, [])
            n_cards = len(crops)
            if n_cards >= 2:
                crop_texts = [predictions_by_crop[c][field][0] for c in crops]
                cross_card_agreement = 1.0 if len({t.strip().lower() for t in crop_texts}) == 1 else 0.0
            else:
                cross_card_agreement = 0.5  # not applicable with 0-1 contributing cards
            visual, has_visual = _visual_features_for_crops(crops, images_dir=images_dir)
            row = {
                "self_consistency": self_consistency,
                "text_length": float(len(text)) if text != "MISSING" else 0.0,
                "n_cards": float(n_cards),
                "cross_card_agreement": cross_card_agreement,
                # 0 contributing card crops but a non-MISSING value can only have
                # come from the whole-image fallback -- apply_wholeimage_fallback
                # only ever fills a field the per-card merge left "MISSING".
                "from_wholeimage_fallback": 1.0 if (n_cards == 0 and text != "MISSING") else 0.0,
                "has_visual_features": 1.0 if has_visual else 0.0,
            }
            for name in FEATURE_NAMES:
                row[name] = visual[name] if visual is not None else float("nan")
            rows[field].append(row)
    return rows
