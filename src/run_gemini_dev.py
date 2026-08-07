"""Batch-transcribes card crops belonging to the 200 dev (train.csv) specimens
through the Gemini API, for comparison against the classical doctr pipeline's
0.576 local AURC baseline (docs/superpowers/plans/2026-07-29-submission-pipeline.md).
Confound cards, per the leakage-free dev relevance flags, are skipped entirely
to save quota. See src.gemini_batch_runner for the resumable batch logic shared
with src.run_gemini_test.

Phase 2 (after the per-card pass): specimens still both-fields-MISSING after
the per-card merge (failed card detection, an over-eager confound filter, or
a crop the model itself couldn't read) get one whole-image fallback call each
-- see docs/superpowers/specs/2026-08-07-gemini-wholeimage-fallback-design.md.

Usage:
    conda run -n museumscat-challenge python -m src.run_gemini_dev \
        [--n-samples 3] [--rpm 10] [--limit 5] [--models gemini-3.5-flash-lite,gemini-3.1-flash-lite]

Requires GEMINI_API_KEY in the environment (or --api-key). Run
src.evaluate_gemini_dev afterward to check the resulting dev-set AURC before
spending quota on the full test set via src.run_gemini_test.
"""
import argparse
import os

import pandas as pd

from src.calibrate_gemini_confidence import load_predictions_by_crop, merge_predictions_by_specimen
from src.card_relevance_visual import compute_dev_relevance_flags
from src.gemini_batch_runner import run_batch, run_wholeimage_batch
from src.gemini_transcribe import MODEL_FALLBACK_ORDER
from src.pipeline import load_manifest

TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
OUTPUT_PATH = os.path.join("data", "processed", "gemini", "dev_card_predictions.csv")
OUTPUT_PATH_WHOLEIMAGE = os.path.join("data", "processed", "gemini", "dev_wholeimage_predictions.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=3,
                         help="self-consistency samples per card (default 3)")
    parser.add_argument("--rpm", type=int, default=10,
                         help="max Gemini calls per minute (default 10, conservative for free tier)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process this many pending crops (for smoke testing)")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"),
                         help="defaults to the GEMINI_API_KEY environment variable")
    parser.add_argument("--models", default=",".join(MODEL_FALLBACK_ORDER),
                         help="comma-separated, tried in order, falling forward to the next "
                              "one once the current model's free-tier quota is exhausted "
                              f"(default: {','.join(MODEL_FALLBACK_ORDER)})")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("GEMINI_API_KEY is not set (env var or --api-key)")
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"[setup] reading dev specimens from {TRAIN_CSV_PATH}...", flush=True)
    dev_image_files = pd.read_csv(TRAIN_CSV_PATH)["image_file"].tolist()
    dev_specimens = {os.path.splitext(f)[0] for f in dev_image_files}
    print(f"[setup] {len(dev_specimens)} dev specimens loaded", flush=True)

    print("[setup] computing card relevance flags (visual classifier, cross-validated)...", flush=True)
    relevance_flags = compute_dev_relevance_flags(dev_specimens)
    print(f"[setup] relevance flags computed for {len(relevance_flags)} card crops", flush=True)

    run_batch(dev_image_files, relevance_flags, OUTPUT_PATH,
              args.n_samples, args.rpm, args.limit, args.api_key, models)

    print("[setup] checking per-card results for both-MISSING specimens...", flush=True)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    predictions_by_crop = load_predictions_by_crop(OUTPUT_PATH)
    merged = merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, dev_image_files)

    run_wholeimage_batch(dev_image_files, merged, OUTPUT_PATH_WHOLEIMAGE,
                          args.n_samples, args.rpm, args.limit, args.api_key, models)


if __name__ == "__main__":
    main()
