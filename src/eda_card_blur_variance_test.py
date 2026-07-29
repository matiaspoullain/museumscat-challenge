"""EDA: blur-then-variance per-card detection (the operation that actually
dilutes the mount's speckle, unlike a bigger raw-variance window -- see
tables_figures/eda/card_detection_variance_test/notes.md's "Follow-up:
blur-then-variance" section for the numeric case that motivated this).

Usage:
    conda run -n museumscat-challenge python -m src.eda_card_blur_variance_test

Output: tables_figures/eda/card_detection_variance_test/blur_<sigma>_<image_file>
-- annotated crop (red = candidate card box) + raw variance heatmap
(viridis, brighter = higher variance), for two blur levels.
"""
import os

import cv2
import numpy as np

from src.detect_labels import get_search_crop

IMAGES_DIR = os.path.join("data", "raw", "images")
OUT_DIR = os.path.join("tables_figures", "eda", "card_detection_variance_test")

SAMPLES = [
    "1709431.jpeg",
    "1705903.jpeg",
    "1706979.jpeg",
    "1705485.jpeg",
    "1706347.jpeg",
    "1743909.jpeg",
]

BRIGHTNESS_THRESHOLD = 140
VARIANCE_WINDOW = 15  # window for the variance step itself, after blurring
SIGMAS_AND_THRESHOLDS = [(15, 50), (40, 10)]
MIN_CARD_AREA = 3000


def local_variance(gray, window):
    mean = cv2.blur(gray.astype(np.float32), (window, window))
    sq_mean = cv2.blur((gray.astype(np.float32)) ** 2, (window, window))
    return sq_mean - mean ** 2


def find_card_boxes(gray, sigma, var_threshold):
    ksize = int(sigma * 4) | 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma) if sigma > 0 else gray
    var = local_variance(blurred, VARIANCE_WINDOW)
    bright = gray > BRIGHTNESS_THRESHOLD
    smooth = var < var_threshold
    mask = (bright & smooth).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [tuple(stats[i, :4]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_CARD_AREA]
    return boxes, var


def _heatmap(var, vmax=200):
    normalized = np.clip(var / vmax * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for image_file in SAMPLES:
        crop = get_search_crop(os.path.join(IMAGES_DIR, image_file))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        panels = [crop]
        for sigma, var_threshold in SIGMAS_AND_THRESHOLDS:
            boxes, var = find_card_boxes(gray, sigma, var_threshold)
            annotated = crop.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 6)
            cv2.putText(annotated, f"blur_sigma={sigma} thr={var_threshold} boxes={len(boxes)}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
            panels.append(annotated)
            panels.append(_heatmap(var))

        h, w = crop.shape[:2]
        scale = 1200 / w
        panels = [cv2.resize(p, (int(w * scale), int(h * scale))) for p in panels]
        stacked = np.vstack(panels)

        out_path = os.path.join(OUT_DIR, f"blur_{image_file}")
        cv2.imwrite(out_path, stacked)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
