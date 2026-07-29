"""Extract per-card crops from every specimen photo's search ROI, via SAM
card detection (src.detect_cards). Saves each detected card as its own
full-resolution image, plus a manifest CSV linking each crop back to its
source image (and its card index within that image), so Task 6's OCR step
(doctr) can be run once per card crop instead of once per whole ROI.

This is a long-running, CPU-only job. On the 6-image validation set at
POINTS_PER_SIDE=32, SAM took ~130-230s/image (see
tables_figures/eda/sam_card_detection_test/notes.md's runtime table) --
extrapolated, the full 3500-image dataset is on the order of days of serial
CPU time. Resumable: images already recorded in processed.log are skipped
on a re-run, so it's safe to interrupt (Ctrl-C) and restart later. Processes
train.csv's 200 labeled rows first, then test.csv's 3300 rows, so the dev
set (usable to validate the OCR step against ground truth) finishes early.

Usage:
    conda run -n museumscat-challenge python -m src.extract_card_crops
    conda run -n museumscat-challenge python -m src.extract_card_crops --limit 20   # smoke test on the first 20 images
    conda run -n museumscat-challenge python -m src.extract_card_crops --workers 4  # cap CPU usage (default: uncapped, uses all cores)

Output:
    data/processed/card_crops/images/<source_stem>_card<NN>.jpeg -- one file
        per detected card, full resolution (not the downsampled copy SAM
        itself runs on).
    data/processed/card_crops/manifest.csv -- columns crop_filename,
        source_image_file, card_index, x1, y1, x2, y2 (box in the deskewed
        search-ROI crop's own pixel coordinates, from src.detect_labels).
        Load with pandas and group by source_image_file to gather all of
        one specimen's card crops back together.
    data/processed/card_crops/processed.log -- one source_image_file per
        line, written only after that image's crops+manifest rows are all
        saved; the resumability ledger for re-running this script.
"""
import argparse
import csv
import os

import cv2
import pandas as pd
from tqdm import tqdm

from src.detect_cards import configure_threads, detect_cards, load_generator
from src.detect_labels import get_search_crop

IMAGES_DIR = os.path.join("data", "raw", "images")
TRAIN_CSV = os.path.join("data", "raw", "train.csv")
TEST_CSV = os.path.join("data", "raw", "test.csv")
OUT_DIR = os.path.join("data", "processed", "card_crops")
CROPS_DIR = os.path.join(OUT_DIR, "images")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.csv")
PROCESSED_LOG_PATH = os.path.join(OUT_DIR, "processed.log")

MANIFEST_COLUMNS = ["crop_filename", "source_image_file", "card_index", "x1", "y1", "x2", "y2"]


def _ordered_image_files():
    """All images under IMAGES_DIR, train.csv rows first then test.csv rows
    -- the labeled 200-row dev set finishes early so the OCR step downstream
    can be validated against ground truth before the 3300-row test set is
    done. Any file on disk that's in neither CSV (shouldn't happen, but
    train.csv + test.csv should equal the full image set) is appended last,
    sorted, as a safety net."""
    train_files = pd.read_csv(TRAIN_CSV)["image_file"].tolist()
    test_files = pd.read_csv(TEST_CSV)["image_file"].tolist()
    on_disk = set(os.listdir(IMAGES_DIR))

    ordered = [f for f in train_files + test_files if f in on_disk]
    leftovers = sorted(on_disk - set(ordered))
    return ordered + leftovers


def _load_processed():
    if not os.path.exists(PROCESSED_LOG_PATH):
        return set()
    with open(PROCESSED_LOG_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _init_manifest():
    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(MANIFEST_COLUMNS)


def process_image(image_file, generator):
    """Detect and save every card crop for one source image. Returns the
    manifest rows to write (empty list if zero cards were detected)."""
    crop = get_search_crop(os.path.join(IMAGES_DIR, image_file))
    boxes = detect_cards(crop, generator)

    stem = os.path.splitext(image_file)[0]
    rows = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        card_crop = crop[y1:y2, x1:x2]
        crop_filename = f"{stem}_card{i:02d}.jpeg"
        cv2.imwrite(os.path.join(CROPS_DIR, crop_filename), card_crop)
        rows.append([crop_filename, image_file, i, x1, y1, x2, y2])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="process only the first N not-yet-done images (for smoke-testing)")
    parser.add_argument("--workers", type=int, default=None,
                         help="CPU threads/cores to use (torch + OpenCV intra-op parallelism); "
                              "default: unset, i.e. torch/OpenCV's own default (all cores)")
    args = parser.parse_args()

    if args.workers is not None:
        configure_threads(args.workers)
        print(f"using {args.workers} CPU threads (of {os.cpu_count()} available)")
    else:
        print(f"using default CPU threading (torch/OpenCV default -- all {os.cpu_count()} cores)")

    os.makedirs(CROPS_DIR, exist_ok=True)
    _init_manifest()
    processed = _load_processed()

    all_images = _ordered_image_files()
    todo = [f for f in all_images if f not in processed]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(processed)} images already done, {len(todo)} to process this run")
    if not todo:
        return

    print("loading SAM (ViT-B) on CPU...")
    generator = load_generator()

    for image_file in tqdm(todo, desc="detecting cards"):
        rows = process_image(image_file, generator)
        with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        with open(PROCESSED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(image_file + "\n")


if __name__ == "__main__":
    main()
