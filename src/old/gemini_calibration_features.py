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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Fixed, not tuned -- same "deliberately simple, not fitted" posture as
# _SHRINKAGE_ALPHA in src.calibrate_gemini_confidence. 5 components is an
# arbitrary cut deep enough to matter (19 -> 5) without being tuned per fold,
# which would need its own nested CV and isn't justified for a first pass.
PCA_N_COMPONENTS = 5
RF_N_ESTIMATORS = 200

from src.eda_card_visual_features import FEATURE_NAMES, IMAGES_DIR, N_FOLDS, RANDOM_STATE, extract_features
from src.metrics import CORRECT_THRESHOLD

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


def _feature_matrix(rows, feature_order=FEATURE_ORDER):
    return np.array([[row[name] for name in feature_order] for row in rows], dtype=float)


def fit_multi_feature_calibrator(rows, correctness_values, feature_order=FEATURE_ORDER, regularization="l2"):
    """Regularized logistic regression on feature_order's columns, predicting
    P(correct). NaN visual features (crops with 0 contributing cards) are mean-
    imputed inside the pipeline, fit only on the training fold -- leakage-free
    the same way StandardScaler's fit is.

    regularization="l1" (Lasso) does feature selection -- zeroing out
    uninformative coefficients -- instead of L2's shrink-everything-toward-zero,
    which src.old.experiment_confidence_grid's "logistic_multi_l1" arm tests as an
    alternative way to fight overfitting on ~200 dev rows with 19 columns,
    without reducing to a black-box representation the way PCA would. L1 needs
    "liblinear" (the default "lbfgs" solver doesn't support it)."""
    labels = [1 if (1.0 - c) <= CORRECT_THRESHOLD else 0 for c in correctness_values]
    if len(set(labels)) < 2:
        return None
    solver = "liblinear" if regularization == "l1" else "lbfgs"
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(),
        LogisticRegression(max_iter=1000, penalty=regularization, solver=solver),
    )
    pipeline.fit(_feature_matrix(rows, feature_order), labels)
    return pipeline


def predict_multi_feature(pipeline, rows, feature_order=FEATURE_ORDER):
    return list(pipeline.predict_proba(_feature_matrix(rows, feature_order))[:, 1])


def oof_multi_feature_calibrated_confidences(rows, correctness_values, n_folds=N_FOLDS, random_state=RANDOM_STATE,
                                              regularization="l2"):
    calibrated = [None] * len(rows)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(rows):
        train_rows = [rows[i] for i in train_idx]
        train_correctness = [correctness_values[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        pipeline = fit_multi_feature_calibrator(train_rows, train_correctness, regularization=regularization)
        if pipeline is None:
            # degenerate fold (every training label the same) -- predict the
            # observed correct-rate for every held-out row, same fallback
            # _ConstantCalibrator gives the 1-D "platt" family
            fallback_rate = sum(
                1 for c in train_correctness if (1.0 - c) <= CORRECT_THRESHOLD
            ) / len(train_correctness)
            predictions = [fallback_rate] * len(test_rows)
        else:
            predictions = predict_multi_feature(pipeline, test_rows)
        for i, p in zip(test_idx, predictions):
            calibrated[i] = float(p)
    return calibrated


def fit_pca_rf_calibrator(rows, correctness_values, feature_order=FEATURE_ORDER,
                           n_components=PCA_N_COMPONENTS, n_estimators=RF_N_ESTIMATORS,
                           random_state=RANDOM_STATE):
    """Dimensionality-reduced alternative to fit_multi_feature_calibrator: PCA
    collapses feature_order's 19 columns to n_components before a random forest
    classifier, instead of L2/L1 logistic regression's linear shrinkage/selection
    -- tests whether a nonlinear model on fewer, decorrelated inputs generalizes
    better than a linear model on all 19 raw features at this dev set's size
    (~200 rows). PCA is unsupervised (fit on X only, blind to correctness_values),
    so which components come out first isn't guaranteed to align with what
    predicts correctness -- that's an open question this arm is testing, not an
    assumption it makes."""
    labels = [1 if (1.0 - c) <= CORRECT_THRESHOLD else 0 for c in correctness_values]
    if len(set(labels)) < 2:
        return None
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(),
        PCA(n_components=n_components, random_state=random_state),
        RandomForestClassifier(n_estimators=n_estimators, random_state=random_state),
    )
    pipeline.fit(_feature_matrix(rows, feature_order), labels)
    return pipeline


def oof_pca_rf_calibrated_confidences(rows, correctness_values, n_folds=N_FOLDS, random_state=RANDOM_STATE,
                                       n_components=PCA_N_COMPONENTS, n_estimators=RF_N_ESTIMATORS):
    calibrated = [None] * len(rows)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(rows):
        train_rows = [rows[i] for i in train_idx]
        train_correctness = [correctness_values[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        pipeline = fit_pca_rf_calibrator(
            train_rows, train_correctness, n_components=n_components, n_estimators=n_estimators,
            random_state=random_state,
        )
        if pipeline is None:
            fallback_rate = sum(
                1 for c in train_correctness if (1.0 - c) <= CORRECT_THRESHOLD
            ) / len(train_correctness)
            predictions = [fallback_rate] * len(test_rows)
        else:
            predictions = predict_multi_feature(pipeline, test_rows)
        for i, p in zip(test_idx, predictions):
            calibrated[i] = float(p)
    return calibrated
