"""EDA: does Segment Anything (SAM, ViT-B checkpoint) segment individual
label cards against the mount's speckled background, zero-shot? Tested as
an alternative to the hand-crafted CV approaches already tried and
abandoned (see tables_figures/eda/card_detection_variance_test/notes.md --
brightness+small-window texture, Canny+rectangularity, and blur-then-variance
all failed because the card-vs-mount smoothness relationship flips sign
depending on how much ink a given card carries).

The actual detection logic (SAM loading, select_card_masks filtering) lives
in src/detect_cards.py, shared with the production script
src/extract_card_crops.py. This script is just the visual
validation/threshold-tuning harness -- see
tables_figures/eda/sam_card_detection_test/notes.md for the full writeup.

Setup: `pip install segment-anything` (done), checkpoint downloaded from
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth into
models/sam/ (gitignored, like every other model artifact in this repo).

Usage:
    conda run -n museumscat-challenge python -m src.old.eda_sam_card_detection
    (raw masks are cached to CACHE_DIR after the first run, gitignored, so
    re-running to tune select_card_masks's thresholds is fast -- add
    --regenerate to force re-running SAM itself)
"""
import os
import pickle  # safe here: cache is only ever written by this script and read back locally, never an untrusted/external source
import sys
import time

import cv2

from src.detect_cards import load_generator, select_card_masks, downsample
from src.detect_labels import get_search_crop

IMAGES_DIR = os.path.join("data", "raw", "images")
OUT_DIR = os.path.join("tables_figures", "eda", "sam_card_detection_test")
CACHE_DIR = os.path.join("data", "processed", "sam_mask_cache")

SAMPLES = [
    "1709431.jpeg",  # single card, long multi-line locality
    "1705903.jpeg",  # 2 cards
    "1706979.jpeg",  # 2 cards
    "1705485.jpeg",  # 2 cards
    "1706347.jpeg",  # 2 real cards + small confound tags (the hard case)
    "1743909.jpeg",  # single card + confound card
]


def _small_crop(image_file):
    crop = get_search_crop(os.path.join(IMAGES_DIR, image_file))
    small, _ = downsample(crop)
    return small


def generate_masks(image_file, generator):
    """Run SAM on image_file's search-ROI crop (downsampled), cache the raw
    mask list to CACHE_DIR, and return (small_bgr_crop, masks)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{os.path.splitext(image_file)[0]}.pkl")
    small = _small_crop(image_file)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            masks = pickle.load(f)
        return small, masks

    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    print(f"{image_file}: running SAM on {small.shape[1]}x{small.shape[0]}...")
    t0 = time.time()
    masks = generator.generate(small_rgb)
    print(f"  {len(masks)} raw masks in {time.time() - t0:.1f}s")
    with open(cache_path, "wb") as f:
        pickle.dump(masks, f)
    return small, masks


def draw_boxes(image, boxes, color=(0, 0, 255)):
    overlay = image.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
    return overlay


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    regenerate = "--regenerate" in sys.argv
    if regenerate:
        for image_file in SAMPLES:
            cache_path = os.path.join(CACHE_DIR, f"{os.path.splitext(image_file)[0]}.pkl")
            if os.path.exists(cache_path):
                os.remove(cache_path)

    print("loading SAM (ViT-B) on CPU...")
    generator = load_generator()

    for image_file in SAMPLES:
        small, masks = generate_masks(image_file, generator)
        boxes = select_card_masks(masks, small.shape[:2])
        print(f"{image_file}: {len(masks)} raw masks -> {len(boxes)} card candidates")

        overlay = draw_boxes(small, boxes)
        out_path = os.path.join(OUT_DIR, f"filtered_{image_file}")
        cv2.imwrite(out_path, overlay)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
