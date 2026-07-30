"""Visual (color/texture/border) card-relevance classifier -- reuses the
hand-crafted features and random forest from src.eda_card_visual_features,
which found it separates informative from confound card crops far better than
the keyword-based src.classify_card_relevance (0.870 vs 0.505 cross-validated
accuracy). This module turns that into a crop_filename -> bool lookup for the
200 train.csv (dev) specimens, for plugging into src.pipeline via
src.evaluate_local, so the pipeline-level AURC/NED impact can be measured
directly instead of just the classifier's own accuracy.

Leakage note: a labeled crop's flag is its out-of-fold cross-validated
prediction (never produced by a model trained on that crop). Everything else
-- labeled-but-"skipped" crops, and crops belonging to dev specimens that were
never labeled at all -- gets the prediction of a model fit on all labeled
crops; that's still leakage-free, since none of those crops were used to fit
it either.

Usage:
    conda run -n museumscat-challenge python -m src.card_relevance_visual
"""
import csv
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.eda_card_visual_features import (
    FEATURE_NAMES,
    IMAGES_DIR,
    MATCHES_PATH,
    N_FOLDS,
    RANDOM_STATE,
    _load_labeled_crops,
    build_feature_table,
    extract_features,
)

MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")


def _all_crops_by_specimen(manifest_path):
    by_image = {}
    with open(manifest_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            specimen = os.path.splitext(row["source_image_file"])[0]
            by_image.setdefault(specimen, []).append(row["crop_filename"])
    return by_image


def _feature_matrix(rows):
    return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)


def _load_labeled_features():
    with open(MATCHES_PATH, encoding="utf-8") as f:
        matches = json.load(f)
    labeled_rows = build_feature_table(_load_labeled_crops(matches))
    x_labeled = _feature_matrix(labeled_rows)
    y_labeled = np.array([row["informative"] for row in labeled_rows], dtype=int)
    return labeled_rows, x_labeled, y_labeled


def train_full_model():
    """Random forest fit on every labeled crop (all 323, "skipped" excluded).
    Safe to apply to any crop that wasn't part of that labeled set -- e.g. the
    3300 test.csv specimens, none of which have ever been labeled."""
    _, x_labeled, y_labeled = _load_labeled_features()
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(x_labeled, y_labeled)
    return model


def _predict_crop(model, crop_filename):
    image_bgr = cv2.imread(os.path.join(IMAGES_DIR, crop_filename))
    if image_bgr is None:
        return True  # fail open: unreadable crop, don't drop it silently
    features = extract_features(image_bgr)
    x = np.array([[features[name] for name in FEATURE_NAMES]])
    return bool(model.predict(x)[0])


def compute_relevance_flags_for_specimens(specimens, model=None):
    """crop_filename -> bool for every card crop belonging to `specimens`
    (specimen ids, no extension), all predicted fresh by `model` (trained via
    train_full_model() if not given). Meant for specimens with no relevance
    labels at all (e.g. the test set) -- for the 200 train.csv specimens, use
    compute_dev_relevance_flags instead to avoid leakage on the labeled ones."""
    if model is None:
        model = train_full_model()
    by_image = _all_crops_by_specimen(MANIFEST_PATH)
    return {
        crop_filename: _predict_crop(model, crop_filename)
        for specimen in specimens
        for crop_filename in by_image.get(specimen, [])
    }


def compute_dev_relevance_flags(dev_specimens):
    """crop_filename -> bool for every card crop belonging to `dev_specimens`
    (specimen ids, no extension). Labeled crops get their out-of-fold
    cross-validated prediction (never produced by a model trained on that
    crop); everything else gets train_full_model()'s prediction, which is
    leakage-free too since none of those crops were used to fit it."""
    labeled_rows, x_labeled, y_labeled = _load_labeled_features()

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = cross_val_predict(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), x_labeled, y_labeled, cv=cv
    )
    flags = {row["crop_filename"]: bool(pred) for row, pred in zip(labeled_rows, oof_pred)}

    full_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    full_model.fit(x_labeled, y_labeled)

    by_image = _all_crops_by_specimen(MANIFEST_PATH)
    for specimen in dev_specimens:
        for crop_filename in by_image.get(specimen, []):
            if crop_filename not in flags:
                flags[crop_filename] = _predict_crop(full_model, crop_filename)
    return flags


def main():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    dev_specimens = {os.path.splitext(f)[0] for f in train_df["image_file"]}
    flags = compute_dev_relevance_flags(dev_specimens)
    n_relevant = sum(flags.values())
    print(f"computed relevance for {len(flags)} crops across {len(dev_specimens)} dev specimens "
          f"({n_relevant} relevant, {len(flags) - n_relevant} confound)")


if __name__ == "__main__":
    main()
