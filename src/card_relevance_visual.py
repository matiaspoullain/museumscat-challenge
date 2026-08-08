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
TEST_RELEVANCE_CACHE_PATH = os.path.join("data", "processed", "gemini", "test_relevance_flags.csv")


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


def compute_relevance_flags_for_specimens(specimens, model=None, manifest_path=MANIFEST_PATH,
                                           images_dir=IMAGES_DIR, progress_every=500):
    """crop_filename -> bool for every card crop belonging to `specimens`
    (specimen ids, no extension), all predicted fresh by `model` (trained via
    train_full_model() if not given). Meant for specimens with no relevance
    labels at all (e.g. the test set) -- for the 200 train.csv specimens, use
    compute_dev_relevance_flags instead to avoid leakage on the labeled ones.

    Feature extraction (one cv2.imread + extract_features call per crop) is
    unavoidable and dominates the wall-clock cost -- on the ~6500-crop test
    set this step alone takes several minutes with no other output otherwise,
    which previously looked hung (observed 2026-08-08). progress_every prints
    a line periodically so it doesn't look stuck; pass 0/None to disable.
    model.predict is called once on the whole batch instead of once per crop
    -- RandomForestClassifier.predict's fixed per-call overhead adds up
    at this scale (measured ~9ms/crop one-by-one vs ~0.2ms/crop batched)."""
    if model is None:
        model = train_full_model()
    by_image = _all_crops_by_specimen(manifest_path)
    crop_filenames = [crop_filename for specimen in specimens for crop_filename in by_image.get(specimen, [])]
    total = len(crop_filenames)

    features_by_crop = {}
    for i, crop_filename in enumerate(crop_filenames, start=1):
        image_bgr = cv2.imread(os.path.join(images_dir, crop_filename))
        if image_bgr is not None:
            features_by_crop[crop_filename] = extract_features(image_bgr)
        if progress_every and i % progress_every == 0:
            print(f"[relevance] extracted features for {i}/{total} crops...", flush=True)

    predicted_by_crop = {}
    if features_by_crop:
        x = _feature_matrix(list(features_by_crop.values()))
        predictions = model.predict(x)
        predicted_by_crop = dict(zip(features_by_crop.keys(), (bool(p) for p in predictions)))

    # fail open: an unreadable crop (image_bgr is None) keeps its True default
    # rather than being silently dropped from the relevance-flagged set.
    return {crop_filename: predicted_by_crop.get(crop_filename, True) for crop_filename in crop_filenames}


def _load_relevance_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, encoding="utf-8") as f:
        return {row["crop_filename"]: row["is_relevant"] == "True" for row in csv.DictReader(f)}


def _save_relevance_cache(cache_path, flags):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_filename", "is_relevant"])
        writer.writeheader()
        for crop_filename, is_relevant in flags.items():
            writer.writerow({"crop_filename": crop_filename, "is_relevant": is_relevant})


def load_or_compute_relevance_flags(specimens, cache_path, model=None, manifest_path=MANIFEST_PATH,
                                     images_dir=IMAGES_DIR, progress_every=500):
    """Like compute_relevance_flags_for_specimens, but persists the result to
    cache_path (crop_filename,is_relevant CSV) so a later re-run -- e.g. the
    next day's src.run_gemini_test invocation, resuming after a daily quota
    429, or src.run_gemini_test_wholeimage running after it -- can skip the
    several-minutes feature-extraction pass entirely and just read the file,
    instead of recomputing it from scratch every single invocation.

    Recomputes (and overwrites the cache) if the cache doesn't cover every
    crop the given specimens actually have in the manifest right now -- e.g.
    the very first call, or a manifest that grew since the cache was written.
    A partial/stale cache is treated the same as no cache: safer to redo the
    (idempotent) computation than to silently serve incomplete flags."""
    by_image = _all_crops_by_specimen(manifest_path)
    needed = {crop_filename for specimen in specimens for crop_filename in by_image.get(specimen, [])}

    cached = _load_relevance_cache(cache_path)
    if needed <= cached.keys():
        print(f"[relevance] using cached flags from {cache_path} ({len(needed)} crops)", flush=True)
        return {crop_filename: cached[crop_filename] for crop_filename in needed}

    print(f"[relevance] cache at {cache_path} missing {len(needed - cached.keys())} of "
          f"{len(needed)} needed crops -- computing fresh...", flush=True)
    flags = compute_relevance_flags_for_specimens(specimens, model=model, manifest_path=manifest_path,
                                                   images_dir=images_dir, progress_every=progress_every)
    _save_relevance_cache(cache_path, flags)
    return flags


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
