"""Score a hybrid OCR strategy on real dev-set images: per-field NED + AURC.

Usage:
    conda run -n museumscat-challenge python -m src.old.score_hybrid_dev [n_images]

Hybrid rule (see docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md
"Fine-tune result" section for why): always use the baseline doctr engine
for verbatimDate (clear numeric win in every comparison so far). For
verbatimLocality, run both the baseline doctr engine and
doctr-finetuned-danish-stage2 on the same crop and keep whichever
assembled prediction has the higher mean snippet confidence. A prior
version selected by "does the fine-tuned prediction contain a Danish
letter" instead -- that performed no better than just using the
fine-tuned model outright, since garbled noise can still contain a stray
aeoeaa/AEOEAA. Caveat worth watching: the two engines' confidence scores
aren't independently calibrated against each other, so "higher raw
confidence" isn't guaranteed to mean "more likely correct" across engines
-- this is an empirical test of whether it works in practice, not an
assumption that it must.

Same approximation as score_ocr_dev.py (word-level classify_field,
space-joined snippets, mean-confidence AURC) -- comparable numbers, same
25-row sample (seed=7).
"""
import os
import sys

import cv2
import pandas as pd

from src.old.benchmark_ocr import ENGINES
from src.detect_labels import _deskew, find_search_roi
from src.metrics import aurc, best_pipe_ned
from src.old.score_ocr_dev import _assemble_by_field, _sample_rows

IMAGES_DIR = os.path.join("data", "raw", "images")

DATE_ENGINE = "doctr"
LOCALITY_FALLBACK_ENGINE = "doctr"
LOCALITY_DANISH_ENGINE = "doctr-finetuned-danish-stage2"


def main():
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 25

    rows = _sample_rows(n_images)
    run_date_engine = ENGINES[DATE_ENGINE]
    run_locality_fallback = ENGINES[LOCALITY_FALLBACK_ENGINE]
    run_locality_danish = ENGINES[LOCALITY_DANISH_ENGINE]

    results = {"date": [], "locality": []}
    for _, row in rows.iterrows():
        image_path = os.path.join(IMAGES_DIR, row["image_file"])
        raw = cv2.imread(image_path)
        deskewed, _ = _deskew(raw)
        _, roi_box = find_search_roi(image_path)
        x1, y1, x2, y2 = roi_box
        crop = deskewed[y1:y2, x1:x2]

        date_fields = _assemble_by_field(run_date_engine(crop))
        fallback_fields = _assemble_by_field(run_locality_fallback(crop))
        danish_fields = _assemble_by_field(run_locality_danish(crop))

        date_text, date_conf = date_fields["date"]

        danish_text, danish_conf = danish_fields["locality"]
        fallback_text, fallback_conf = fallback_fields["locality"]
        if danish_conf >= fallback_conf:
            locality_text, locality_conf = danish_text, danish_conf
            source = "finetuned-danish"
        else:
            locality_text, locality_conf = fallback_text, fallback_conf
            source = "baseline"

        fields = {"date": (date_text, date_conf), "locality": (locality_text, locality_conf)}

        for field in ("date", "locality"):
            pred_text, pred_conf = fields[field]
            truth = row[f"verbatim{field.capitalize()}"]
            err = best_pipe_ned(pred_text, truth, is_date=(field == "date"))
            results[field].append((err, pred_conf))
            extra = f" source={source}" if field == "locality" else ""
            print(f"{row['image_file']} [{field}] pred={pred_text!r} truth={truth!r} ned={err:.3f} conf={pred_conf:.2f}{extra}")

    print()
    for field, pairs in results.items():
        errors = [e for e, _ in pairs]
        confs = [c for _, c in pairs]
        mean_ned = sum(errors) / len(errors)
        field_aurc = aurc(errors, confs)
        print(f"{field}: mean_NED={mean_ned:.3f} AURC={field_aurc:.3f} (n={len(errors)})")


if __name__ == "__main__":
    main()
