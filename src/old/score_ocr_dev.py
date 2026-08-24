"""Score OCR engine/recognizer choices on real dev-set images: per-field NED + AURC.

Usage:
    conda run -n museumscat-challenge python -m src.old.score_ocr_dev <engine> [n_images]

This is an approximation, not the final Task 7 submission pipeline -- just
enough signal to compare OCR engine/recognizer configs quantitatively
instead of by eyeballing a handful of images. Pipeline: detect ROI (Task 5)
-> run OCR engine (word/phrase-level snippets) -> classify_field (Task 3)
each snippet -> join same-field snippets in reading order with a plain
space (not the production " | " multi-card join from Task 4, which isn't
needed for this single-recognizer comparison) -> best_pipe_ned vs ground
truth per field -> aurc using the mean snippet confidence per field.
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

from src.old.benchmark_ocr import ENGINES
from src.classify_field import classify_field
from src.detect_labels import _deskew, find_search_roi
from src.metrics import aurc, best_pipe_ned

IMAGES_DIR = os.path.join("data", "raw", "images")
TRAIN_CSV = os.path.join("data", "raw", "train.csv")


def _sample_rows(n, seed=7):
    df = pd.read_csv(TRAIN_CSV)
    sub = df[(df.verbatimDate != "MISSING") & (df.verbatimLocality != "MISSING")]
    return sub.sample(n, random_state=seed)


def _assemble_by_field(snippets):
    """snippets: list of (text, confidence, (x1,y1,x2,y2)).
    Returns {"date": (text, conf), "locality": (text, conf)}."""
    buckets = {"date": [], "locality": []}
    for text, conf, (x1, y1, x2, y2) in snippets:
        label = classify_field(text)
        if label in buckets:
            buckets[label].append((x1, y1, text, conf))

    out = {}
    for field, items in buckets.items():
        if not items:
            out[field] = ("MISSING", 0.5)
            continue
        items.sort(key=lambda t: (t[1] // 50, t[0]))
        text = " ".join(t[2] for t in items)
        conf = float(np.mean([t[3] for t in items]))
        out[field] = (text, conf)
    return out


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ENGINES:
        print(__doc__)
        sys.exit(1)

    engine = sys.argv[1]
    n_images = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    run_engine = ENGINES[engine]

    rows = _sample_rows(n_images)

    results = {"date": [], "locality": []}
    for _, row in rows.iterrows():
        image_path = os.path.join(IMAGES_DIR, row["image_file"])
        raw = cv2.imread(image_path)
        deskewed, _ = _deskew(raw)
        _, roi_box = find_search_roi(image_path)
        x1, y1, x2, y2 = roi_box
        crop = deskewed[y1:y2, x1:x2]

        snippets = run_engine(crop)
        fields = _assemble_by_field(snippets)

        for field in ("date", "locality"):
            pred_text, pred_conf = fields[field]
            truth = row[f"verbatim{field.capitalize()}"]
            err = best_pipe_ned(pred_text, truth, is_date=(field == "date"))
            results[field].append((err, pred_conf))
            print(f"{row['image_file']} [{field}] pred={pred_text!r} truth={truth!r} ned={err:.3f} conf={pred_conf:.2f}")

    print()
    for field, pairs in results.items():
        errors = [e for e, _ in pairs]
        confs = [c for _, c in pairs]
        mean_ned = float(np.mean(errors))
        field_aurc = aurc(errors, confs)
        print(f"{field}: mean_NED={mean_ned:.3f} AURC={field_aurc:.3f} (n={len(errors)})")


if __name__ == "__main__":
    main()
