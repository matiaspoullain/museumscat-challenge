"""Extract word-level crops from dev images whose ground-truth text contains
a Danish letter, for manual transcription and doctr recognizer fine-tuning.
See docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md.

Usage:
    conda run -n museumscat-challenge python -m src.old.extract_reco_crops
"""
import json
import os
import re

import cv2
import pandas as pd

from src.old.benchmark_ocr import ENGINES
from src.detect_labels import get_search_crop

TRAIN_CSV = os.path.join("data", "raw", "train.csv")
IMAGES_DIR = os.path.join("data", "raw", "images")
DATASET_DIR = os.path.join("data", "processed", "reco_finetune", "dataset")
IMAGES_OUT_DIR = os.path.join(DATASET_DIR, "images")
CANDIDATES_PATH = os.path.join(DATASET_DIR, "candidates.json")

DANISH_LETTERS = re.compile(r"[æøåÆØÅ]")


def rows_with_danish_letters(df):
    """Rows whose verbatimDate or verbatimLocality contains æ/ø/å (either case)."""
    has_danish = (
        df["verbatimDate"].str.contains(DANISH_LETTERS, na=False)
        | df["verbatimLocality"].str.contains(DANISH_LETTERS, na=False)
    )
    return df[has_danish]


def extract_crops():
    """Run word detection on every Danish-candidate image, save each detected
    word crop as its own PNG, and write a manifest for the labeling app
    mapping {crop_filename: {source, doctr_text, doctr_confidence}} -- doctr's
    own recognition guess and confidence are kept so the labeling app can
    prioritize the least-confident (most likely wrong) crops first. Returns
    the number of crops written."""
    os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    rows = rows_with_danish_letters(df)
    run_engine = ENGINES["doctr"]

    candidates = {}
    for _, row in rows.iterrows():
        image_file = row["image_file"]
        crop = get_search_crop(os.path.join(IMAGES_DIR, image_file))
        words = run_engine(crop)

        stem = os.path.splitext(image_file)[0]
        for i, (text, confidence, (x1, y1, x2, y2)) in enumerate(words):
            if x2 <= x1 or y2 <= y1:
                continue
            word_crop = crop[y1:y2, x1:x2]
            crop_filename = f"{stem}_{i:03d}.png"
            cv2.imwrite(os.path.join(IMAGES_OUT_DIR, crop_filename), word_crop)
            candidates[crop_filename] = {
                "source": image_file,
                "doctr_text": text,
                "doctr_confidence": confidence,
            }

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    return len(candidates)


if __name__ == "__main__":
    n = extract_crops()
    print(f"Wrote {n} candidate crops to {IMAGES_OUT_DIR}")
    print(f"Manifest: {CANDIDATES_PATH}")
