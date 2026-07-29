"""EDA: does a larger local-variance window make brightness+smoothness
per-card detection viable, scoped to the already-cropped search ROI?

Background: per-card detection via brightness+local-variance was tried and
abandoned in Task 5 (see tables_figures/eda/layout_notes.md) -- a small
variance window flags individual text strokes as "high variance", excluding
the exact card interior we want. This retests the idea with a wider
moving-average window (dilutes sparse handwriting against blank paper) to
see whether that changes the outcome, now that cards are found within the
finalized search ROI (src.detect_labels.get_search_crop) rather than the
whole image.

Usage:
    conda run -n museumscat-challenge python -m src.eda_card_variance_test

Output: tables_figures/eda/card_detection_variance_test/<image_file> --
for each sample, stacked panels: annotated crop (red = candidate card box)
for window=15 (original) and window=41 (best found), then the two raw
variance heatmaps (brighter = higher local variance) for direct comparison.
"""
import os

import cv2
import numpy as np

from src.detect_labels import get_search_crop

IMAGES_DIR = os.path.join("data", "raw", "images")
OUT_DIR = os.path.join("tables_figures", "eda", "card_detection_variance_test")

SAMPLES = [
    "1709431.jpeg",  # single card, long multi-line locality
    "1705903.jpeg",  # 2 cards
    "1706979.jpeg",  # 2 cards + 2 confound cards
    "1705485.jpeg",  # 2 cards + 2 confound cards
    "1706347.jpeg",  # 2 cards + confound card
    "1743909.jpeg",  # single card + confound card
]

BRIGHTNESS_THRESHOLD = 140
WINDOWS_AND_THRESHOLDS = [(15, 300), (41, 200)]
MIN_CARD_AREA = 3000


def local_variance(gray, window):
    """Local variance of `gray` over a window x window moving average."""
    mean = cv2.blur(gray.astype(np.float32), (window, window))
    sq_mean = cv2.blur((gray.astype(np.float32)) ** 2, (window, window))
    return sq_mean - mean ** 2


def find_card_boxes(gray, window, var_threshold):
    """Candidate card boxes: bright and locally smooth at the given window size."""
    bright = gray > BRIGHTNESS_THRESHOLD
    var = local_variance(gray, window)
    smooth = var < var_threshold
    mask = (bright & smooth).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [tuple(stats[i, :4]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_CARD_AREA]
    return boxes, var


def _heatmap(var, vmax=600):
    normalized = np.clip(var / vmax * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for image_file in SAMPLES:
        crop = get_search_crop(os.path.join(IMAGES_DIR, image_file))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        panels = [crop]
        for window, var_threshold in WINDOWS_AND_THRESHOLDS:
            boxes, var = find_card_boxes(gray, window, var_threshold)
            annotated = crop.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 6)
            cv2.putText(annotated, f"window={window} boxes={len(boxes)}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            panels.append(annotated)
            panels.append(_heatmap(var))

        h, w = crop.shape[:2]
        scale = 1200 / w
        panels = [cv2.resize(p, (int(w * scale), int(h * scale))) for p in panels]
        stacked = np.vstack(panels)

        out_path = os.path.join(OUT_DIR, image_file)
        cv2.imwrite(out_path, stacked)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
