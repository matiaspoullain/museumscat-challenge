"""Visualize find_search_roi on a single image, for manual inspection.

Usage:
    conda run -n museumscat-challenge python -m src.inspect_detection <image_file_or_path> [output_path]

<image_file_or_path> can be just a filename (looked up under data/raw/images/)
or a full path. Output defaults to
tables_figures/eda/dassco_anchor_test/manual_inspection/<filename>.
"""
import os
import sys

import cv2

from src.detect_labels import _dassco_bbox, _deskew, find_search_roi

IMAGES_DIR = os.path.join("data", "raw", "images")
DEFAULT_OUT_DIR = os.path.join("tables_figures", "eda", "dassco_anchor_test", "manual_inspection")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]
    image_path = arg if os.path.exists(arg) else os.path.join(IMAGES_DIR, arg)
    if not os.path.exists(image_path):
        print(f"image not found: {arg} (looked at '{arg}' and '{image_path}')")
        sys.exit(1)

    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        out_path = os.path.join(DEFAULT_OUT_DIR, os.path.basename(image_path))

    angle, (x1, y1, x2, y2) = find_search_roi(image_path)

    raw = cv2.imread(image_path)
    deskewed, applied_angle = _deskew(raw)
    dassco = _dassco_bbox(deskewed)

    if dassco:
        dx, dy, dw, dh = dassco
        cv2.rectangle(deskewed, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 10)
    cv2.rectangle(deskewed, (x1, y1), (x2, y2), (0, 0, 255), 10)

    h, w = deskewed.shape[:2]
    cv2.imwrite(out_path, cv2.resize(deskewed, (w // 4, h // 4)))

    print(f"image: {image_path}")
    print(f"rotation applied: {applied_angle:.2f} degrees")
    print(f"dassco bbox: {dassco}")
    print(f"search roi: ({x1}, {y1}, {x2}, {y2})")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
