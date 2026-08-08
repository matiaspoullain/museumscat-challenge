"""Calibrates Gemini self-consistency confidence against real dev-set
correctness via isotonic regression -- raw self-consistency (like the doctr
pipeline's raw softmax confidence before it) isn't the same thing as
"probability this transcription is correct", which is exactly what AURC
punishes when miscalibrated.

Two calibration passes, mirroring src.card_relevance_visual's leakage split:
- fit_calibrators(): fit on all dev data -- the production calibrator saved to
  models/ and applied to the disjoint test set by src.generate_submission_gemini.
- oof_calibrated_confidences(): 5-fold out-of-fold predictions, for an honest
  dev-set AURC estimate in src.evaluate_gemini_dev (fitting and evaluating
  calibration on the same 200 rows would be optimistic/circular).

Usage:
    conda run -n museumscat-challenge python -m src.calibrate_gemini_confidence
"""
import csv
import json
import os

import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from src.eda_card_visual_features import N_FOLDS, RANDOM_STATE
from src.merge_records import assemble_field
from src.metrics import best_pipe_ned
from src.pipeline import load_manifest

DEV_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "dev_card_predictions.csv")
DEV_WHOLEIMAGE_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini",
                                                "dev_wholeimage_predictions.csv")
TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
MODELS_DIR = "models"

FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}
FIELDS = tuple(FIELD_IS_DATE)


def load_predictions_by_crop(path=DEV_PREDICTIONS_PATH):
    by_crop = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_crop[row["crop_filename"]] = {
                "verbatimDate": (row["verbatimDate"], float(row["verbatimDate_confidence"])),
                "verbatimLocality": (row["verbatimLocality"], float(row["verbatimLocality_confidence"])),
            }
    return by_crop


def rescored_predictions_by_crop(path, scorer_fn):
    """Re-derives {crop_filename: {field: (text, confidence)}} from each row's
    stored raw_samples_json using scorer_fn (self_consistency_scores or
    soft_consistency_scores from src.gemini_transcribe), instead of the
    confidence columns already written to the CSV -- lets a new confidence-
    scoring function be evaluated against the existing dev predictions without
    re-calling the Gemini API."""
    by_crop = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples = json.loads(row["raw_samples_json"])
            by_crop[row["crop_filename"]] = scorer_fn(samples)
    return by_crop


def merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, image_files):
    """image_file -> {field: (text, confidence)}, same shape as
    src.pipeline.run_pipeline's output. Crops with no row in predictions_by_crop
    (skipped as confound, or not yet transcribed) simply don't contribute --
    same effect as src.pipeline.process_specimen's is_relevant filter.

    A card whose own text for a field is "MISSING" (this specific card has no
    date/locality) also doesn't contribute to that field's pipe join -- joining
    it in would produce e.g. "22.5.1977 | MISSING" for a specimen whose true
    verbatimDate is just "22.5.1977". "MISSING" is reserved for when *no* card
    has a value for the field at all, via assemble_field([])."""
    merged = {}
    for image_file in image_files:
        per_field_cards = {field: [] for field in FIELD_IS_DATE}
        for crop_filename in manifest_by_image.get(image_file, []):
            card = predictions_by_crop.get(crop_filename)
            if card is None:
                continue
            for field in per_field_cards:
                text, confidence = card[field]
                if text != "MISSING":
                    per_field_cards[field].append((text, confidence))
        merged[image_file] = {field: assemble_field(cards) for field, cards in per_field_cards.items()}
    return merged


def load_wholeimage_predictions(path=DEV_WHOLEIMAGE_PREDICTIONS_PATH):
    """source_image_file -> {field: (text, confidence)}, one row per specimen
    (no crop-level nesting, unlike load_predictions_by_crop -- a whole-image
    call already covers the whole specimen). Missing file (fallback pass not
    yet run) returns an empty dict rather than raising, so callers can treat
    "not run yet" the same as "no fallback available"."""
    if not os.path.exists(path):
        return {}
    by_specimen = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_specimen[row["source_image_file"]] = {
                "verbatimDate": (row["verbatimDate"], float(row["verbatimDate_confidence"])),
                "verbatimLocality": (row["verbatimLocality"], float(row["verbatimLocality_confidence"])),
            }
    return by_specimen


def apply_wholeimage_fallback(merged, wholeimage_predictions):
    """Overlays whole-image fallback predictions onto a per-card merge,
    field by field: fills in a field only where the per-card side is
    "MISSING" and the fallback found a real value, never overwriting a
    per-card value that was already present. A specimen only gets a
    whole-image call when *both* fields were MISSING, but the whole-image
    response itself may find only one of the two -- no reason to discard the
    field it did find."""
    result = {}
    for image_file, fields in merged.items():
        fallback = wholeimage_predictions.get(image_file)
        updated = dict(fields)
        if fallback is not None:
            for field in FIELDS:
                if updated[field][0] == "MISSING" and fallback[field][0] != "MISSING":
                    updated[field] = fallback[field]
        result[image_file] = updated
    return result


def _confidences_and_correctness(merged, ground_truth_df):
    """field -> (confidences, correctness=1-ned) lists, aligned by dev specimen."""
    by_field = {field: {"confidences": [], "correctness": []} for field in FIELD_IS_DATE}
    for _, row in ground_truth_df.iterrows():
        for field, is_date in FIELD_IS_DATE.items():
            text, confidence = merged[row["image_file"]][field]
            error = best_pipe_ned(text, str(row[field]), is_date=is_date)
            by_field[field]["confidences"].append(confidence)
            by_field[field]["correctness"].append(1.0 - error)
    return by_field


def fit_calibrators(merged, ground_truth_df):
    by_field = _confidences_and_correctness(merged, ground_truth_df)
    calibrators = {}
    for field, data in by_field.items():
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(data["confidences"], data["correctness"])
        calibrators[field] = calibrator
    return calibrators


def oof_calibrated_confidences(merged, ground_truth_df):
    """5-fold out-of-fold isotonic-calibrated confidence per field, for an
    honest (leakage-free) dev-set AURC estimate."""
    by_field = _confidences_and_correctness(merged, ground_truth_df)
    result = {}
    for field, data in by_field.items():
        confidences, correctness = data["confidences"], data["correctness"]
        calibrated = [None] * len(confidences)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, test_idx in kf.split(confidences):
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit([confidences[i] for i in train_idx], [correctness[i] for i in train_idx])
            for i in test_idx:
                calibrated[i] = float(calibrator.predict([confidences[i]])[0])
        result[field] = calibrated
    return result


def save_calibrators(calibrators, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    paths = {}
    for field, calibrator in calibrators.items():
        path = os.path.join(models_dir, f"gemini_calibrator_{field}.joblib")
        joblib.dump(calibrator, path)
        paths[field] = path
    return paths


def load_calibrators(models_dir=MODELS_DIR):
    # joblib.load deserializes via pickle, but these files are only ever written by
    # save_calibrators() in this repo (models/ is gitignored, never fetched from a
    # remote/untrusted source), so there's no untrusted-input risk here.
    return {
        field: joblib.load(os.path.join(models_dir, f"gemini_calibrator_{field}.joblib"))
        for field in FIELD_IS_DATE
    }


def apply_calibration(calibrator, raw_confidence):
    return float(calibrator.predict([raw_confidence])[0])


def main():
    ground_truth_df = pd.read_csv(TRAIN_CSV_PATH)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    predictions_by_crop = load_predictions_by_crop()
    merged = merge_predictions_by_specimen(
        predictions_by_crop, manifest_by_image, ground_truth_df["image_file"].tolist()
    )

    calibrators = fit_calibrators(merged, ground_truth_df)
    for field, path in save_calibrators(calibrators).items():
        print(f"saved {field} calibrator to {path}")


if __name__ == "__main__":
    main()
