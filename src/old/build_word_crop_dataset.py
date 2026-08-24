"""Build a word-level image+label training set for the doctr recognizer
fine-tune from data/processed/card_crops/word_boxes.json (produced by
src.old.box_card_words_app). See
docs/superpowers/specs/2026-07-29-card-word-boxes-danish-finetune-design.md.

Usage:
    conda run -n museumscat-challenge python -m src.old.build_word_crop_dataset
"""
import json
import os

import cv2
from doctr.datasets import VOCABS

WORD_BOXES_PATH = os.path.join("data", "processed", "card_crops", "word_boxes.json")
CARD_IMAGES_DIR = os.path.join("data", "processed", "card_crops", "images")
DATASET_DIR = os.path.join("data", "processed", "reco_finetune_v2", "dataset")
IMAGES_OUT_DIR = os.path.join(DATASET_DIR, "images")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.json")
DANISH_VOCAB = set(VOCABS["danish"])


def has_out_of_vocab_chars(text, vocab):
    """True if any character in text is missing from vocab (a set of
    allowed characters) -- doctr's CTC training crashes on such labels
    (e.g. archaic 'ÿ' spellings aren't in doctr's built-in danish vocab)."""
    return bool(set(text) - vocab)


def union_box(boxes):
    """boxes: list of (x1, y1, x2, y2). Returns the minimal box enclosing
    all of them -- used when a word was drawn with more than one box
    (e.g. split across two lines)."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def build_dataset():
    """Crop every boxed (non-skipped) word from word_boxes.json out of its
    source card image, save it under IMAGES_OUT_DIR, and write labels.json
    in doctr's {filename: text} training format. Words whose text contains
    a character outside doctr's danish vocab are skipped (they'd crash CTC
    training). Returns (crop_count, multi_box_count, out_of_vocab_count)."""
    os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
    with open(WORD_BOXES_PATH, encoding="utf-8") as f:
        word_boxes = json.load(f)

    labels = {}
    multi_box_count = 0
    out_of_vocab_count = 0
    for crop_filename, words in word_boxes.items():
        card_img = None
        for w in words:
            if not w["boxes"]:
                continue
            if has_out_of_vocab_chars(w["text"], DANISH_VOCAB):
                out_of_vocab_count += 1
                continue
            if len(w["boxes"]) > 1:
                multi_box_count += 1
            x1, y1, x2, y2 = union_box(w["boxes"])
            if card_img is None:
                card_img = cv2.imread(os.path.join(CARD_IMAGES_DIR, crop_filename))
            word_crop = card_img[y1:y2, x1:x2]
            out_filename = f"{os.path.splitext(crop_filename)[0]}_{w['field']}{w['index']}_{w['word_index']}.png"
            cv2.imwrite(os.path.join(IMAGES_OUT_DIR, out_filename), word_crop)
            labels[out_filename] = w["text"]

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    return len(labels), multi_box_count, out_of_vocab_count


if __name__ == "__main__":
    n, multi, out_of_vocab = build_dataset()
    print(f"Wrote {n} word crops to {IMAGES_OUT_DIR} ({multi} words had more than one box)")
    print(f"Skipped {out_of_vocab} words with characters outside doctr's danish vocab")
    print(f"Labels: {LABELS_PATH}")
