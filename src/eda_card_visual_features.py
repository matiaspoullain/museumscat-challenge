"""EDA: do hand-crafted visual features (color, texture, drawn border) separate
informative card crops from non-informative ("confound") ones, independent of
the words doctr recognized on them?

Motivated by visual inspection of data/processed/card_crops/card_text_matches.json's
labeled crops: confound cards ("no_relevant_text") are overwhelmingly printed/stamped
collector or determination labels -- uniform machine typeface, often a drawn border
around the whole card, sometimes tinted (yellow/tan) paper -- while informative cards
("matched") are overwhelmingly handwritten on plain white/cream card with no border.
This tests whether that visual pattern is a usable signal on its own, and whether it
adds anything on top of the existing keyword-based classify_card_relevance.is_relevant_card.

Usage:
    conda run -n museumscat-challenge python -m src.eda_card_visual_features

Output:
    data/processed/card_crops/visual_features.csv -- one row per labeled crop
    tables_figures/eda/card_visual_features/feature_distributions.png
    tables_figures/eda/card_visual_features/notes.md
"""
import csv
import json
import os
from typing import cast

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.classify_card_relevance import CardMatch, MATCHES_PATH, is_relevant_card, _load_predictions_by_card

IMAGES_DIR = os.path.join("data", "processed", "card_crops", "images")
OUT_DIR = os.path.join("tables_figures", "eda", "card_visual_features")
FEATURES_PATH = os.path.join("data", "processed", "card_crops", "visual_features.csv")

BORDER_FRAC = 0.07  # outer-ring width as a fraction of min(width, height)
DARK_THRESHOLD = 100  # grayscale value below which a pixel counts as "ink"
YELLOW_HUE_RANGE = (12, 45)  # OpenCV hue (0-180) range covering tan/yellow tints
MIN_SATURATION_FOR_TINT = 25
HIGH_FREQ_RADIUS_FRAC = 0.25  # FFT radius (as a fraction of min(h, w)) above which energy counts as "high frequency"
N_FOLDS = 5
RANDOM_STATE = 0

FEATURE_NAMES = [
    "aspect_ratio", "mean_hue", "mean_saturation", "std_saturation", "mean_value",
    "std_value", "tint_fraction", "laplacian_var", "fft_highfreq_ratio",
    "edge_density", "border_ink_fraction", "interior_ink_fraction", "border_contrast",
]

# One crop's visual features (extract_features: FEATURE_NAMES plus width/height),
# extended in place by build_feature_table with "crop_filename" (str) and
# "informative" (bool) -- hence the str in the value type.
FeatureRow = dict[str, float | str]


def extract_features(image_bgr: np.ndarray) -> FeatureRow:
    """Hand-crafted visual features for one card crop: color (hue/saturation/tint
    fraction), texture (sharpness, high-frequency FFT energy, edge density), a
    drawn-border proxy (ink fraction in the outer ring vs. the interior), and shape."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    height, width = gray.shape

    tint_mask = (h >= YELLOW_HUE_RANGE[0]) & (h <= YELLOW_HUE_RANGE[1]) & (s >= MIN_SATURATION_FOR_TINT)

    margin = max(2, int(BORDER_FRAC * min(height, width)))
    border_mask = np.zeros_like(gray, dtype=bool)
    border_mask[:margin, :] = True
    border_mask[-margin:, :] = True
    border_mask[:, :margin] = True
    border_mask[:, -margin:] = True
    interior_mask = ~border_mask
    border_ink_fraction = float((gray[border_mask] < DARK_THRESHOLD).mean())
    interior_ink_fraction = float((gray[interior_mask] < DARK_THRESHOLD).mean())

    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32))))
    fy, fx = np.indices(fft_mag.shape)
    radius = np.sqrt((fy - height / 2) ** 2 + (fx - width / 2) ** 2)
    high_freq_cutoff = HIGH_FREQ_RADIUS_FRAC * min(height, width)
    fft_highfreq_ratio = float(fft_mag[radius > high_freq_cutoff].sum() / (fft_mag.sum() + 1e-9))

    edges = cv2.Canny(gray, 80, 160)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "mean_hue": float(h.mean()),
        "mean_saturation": float(s.mean()),
        "std_saturation": float(s.std()),
        "mean_value": float(v.mean()),
        "std_value": float(v.std()),
        "tint_fraction": float(tint_mask.mean()),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "fft_highfreq_ratio": fft_highfreq_ratio,
        "edge_density": float((edges > 0).mean()),
        "border_ink_fraction": border_ink_fraction,
        "interior_ink_fraction": interior_ink_fraction,
        "border_contrast": border_ink_fraction - interior_ink_fraction,
    }


def _load_labeled_crops(matches: dict[str, CardMatch]) -> dict[str, bool]:
    """crop_filename -> True (informative/"matched") or False ("no_relevant_text"),
    skipping "skipped" entries (bad detections, not a relevance judgment)."""
    return {k: v["status"] == "matched" for k, v in matches.items() if v["status"] in ("matched", "no_relevant_text")}


def build_feature_table(labels: dict[str, bool]) -> list[FeatureRow]:
    rows: list[FeatureRow] = []
    for crop_filename, is_informative in labels.items():
        image_bgr = cv2.imread(os.path.join(IMAGES_DIR, crop_filename))
        if image_bgr is None:
            continue
        features = extract_features(image_bgr)
        features["crop_filename"] = crop_filename
        features["informative"] = is_informative
        rows.append(features)
    return rows


def _write_features_csv(rows: list[FeatureRow]) -> None:
    fieldnames = ["crop_filename", "informative", "width", "height"] + FEATURE_NAMES
    with open(FEATURES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_classifiers(rows: list[FeatureRow],
                         ) -> tuple[dict[str, dict[str, float]], list[tuple[str, float]]]:
    """Cross-validated accuracy/precision/recall for: the existing keyword
    classifier alone, each visual classifier alone, and a combined model that
    adds the keyword prediction as one more feature. All evaluated on the same
    labeled subset so numbers are directly comparable."""
    y = np.array([row["informative"] for row in rows], dtype=int)
    x_visual = np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=float)

    predictions_by_card = _load_predictions_by_card()
    keyword_pred = np.array(
        [int(is_relevant_card(predictions_by_card.get(cast(str, row["crop_filename"]), []))) for row in rows]
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {"keyword_baseline": _score(y, keyword_pred)}

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }
    for name, model in models.items():
        pred = cross_val_predict(model, x_visual, y, cv=cv)
        results[f"visual_{name}"] = _score(y, pred)

    x_combined = np.hstack([x_visual, keyword_pred.reshape(-1, 1)])
    combined_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    combined_pred = cross_val_predict(combined_model, x_combined, y, cv=cv)
    results["combined_random_forest"] = _score(y, combined_pred)

    # Feature importance from a single fit on all data, for reporting only.
    importance_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    importance_model.fit(x_visual, y)
    importances = sorted(zip(FEATURE_NAMES, importance_model.feature_importances_), key=lambda t: -t[1])

    return results, importances


def plot_feature_distributions(rows: list[FeatureRow], top_features: list[str], out_path: str) -> None:
    fig, axes = plt.subplots(1, len(top_features), figsize=(4 * len(top_features), 4))
    if len(top_features) == 1:
        axes = [axes]
    informative = [row for row in rows if row["informative"]]
    confound = [row for row in rows if not row["informative"]]
    for ax, feature_name in zip(axes, top_features):
        ax.boxplot(
            [[row[feature_name] for row in informative], [row[feature_name] for row in confound]],
            tick_labels=["informative", "confound"],
        )
        ax.set_title(feature_name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_notes(out_path: str, n_informative: int, n_confound: int, results: dict[str, dict[str, float]],
                 importances: list[tuple[str, float]], plot_filename: str) -> None:
    lines = [
        "# Card visual features EDA\n",
        f"\nLabeled crops: {n_informative} informative (\"matched\") vs {n_confound} confound "
        "(\"no_relevant_text\"), from data/processed/card_crops/card_text_matches.json "
        "(\"skipped\" entries excluded -- bad detections, not a relevance judgment).\n",
        "\n## Cross-validated performance (5-fold, same labeled subset for all rows)\n",
        "\n| classifier | accuracy | precision | recall |",
        "\n|---|---|---|---|",
    ]
    for name, metrics in results.items():
        lines.append(f"\n| {name} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} |")
    lines.append(
        "\n\n`keyword_baseline` is the existing `classify_card_relevance.is_relevant_card` "
        "(no cross-validation needed, it has no fitted parameters). The rest are 5-fold "
        "cross-validated on `data/processed/card_crops/visual_features.csv`.\n"
    )
    lines.append("\n## Feature importance (random forest, fit on all labeled crops)\n")
    for name, importance in importances:
        lines.append(f"\n- {name}: {importance:.3f}")
    lines.append(f"\n\nDistributions for the top features: `{plot_filename}`\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MATCHES_PATH, encoding="utf-8") as f:
        matches = json.load(f)
    labels = _load_labeled_crops(matches)

    rows = build_feature_table(labels)
    _write_features_csv(rows)
    print(f"wrote {len(rows)} rows to {FEATURES_PATH}")

    results, importances = evaluate_classifiers(rows)
    for name, metrics in results.items():
        print(f"{name}: accuracy={metrics['accuracy']:.3f} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}")
    print("feature importance:", importances)

    top_features = [name for name, _ in importances[:4]]
    plot_path = os.path.join(OUT_DIR, "feature_distributions.png")
    plot_feature_distributions(rows, top_features, plot_path)

    n_informative = sum(1 for row in rows if row["informative"])
    n_confound = len(rows) - n_informative
    notes_path = os.path.join(OUT_DIR, "notes.md")
    _write_notes(notes_path, n_informative, n_confound, results, importances, "feature_distributions.png")
    print(f"wrote {notes_path}")


if __name__ == "__main__":
    main()
