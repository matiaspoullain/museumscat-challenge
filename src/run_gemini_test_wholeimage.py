"""Whole-image fallback pass (phase 2) for the ~3300 test.csv specimens --
src.run_gemini_dev bundles both phases in one script since the 200-specimen
dev set finishes within a single quota budget, but the test set's per-card
pass (src.run_gemini_test) spans multiple days, so this is deliberately a
separate script, meant to be run once that pass is completely finished.

Refuses to run while any relevant test card crop is still untranscribed:
src.calibrate_gemini_confidence.merge_predictions_by_specimen can't tell "this
card was never sent to Gemini yet" from "this card was sent and had no
value" -- both look like zero contribution to a specimen's merged fields. Running
the fallback pass early would misclassify still-pending specimens as
genuinely both-fields-MISSING and spend fallback quota on them before their
per-card answer had a chance to land. Pass --force to override this check
(e.g. for a smoke test against a handful of specimens via --limit).

Usage (only after `conda run -n museumscat-challenge python -m
src.run_gemini_test` prints "[done] nothing to do" -- i.e. every relevant test
card crop has been transcribed):
    conda run -n museumscat-challenge python -m src.run_gemini_test_wholeimage \
        [--n-samples 3] [--rpm 10] [--limit 200] [--models gemini-3.5-flash-lite,gemini-3.1-flash-lite]

Requires GEMINI_API_KEY in the environment (or --api-key). Resumable, same
pattern as src.run_gemini_test -- already transcribed specimens are skipped
automatically, so re-run the same command daily/as needed after a daily quota
429.
"""
import argparse
import os

import pandas as pd

from src.calibrate_gemini_confidence import load_predictions_by_crop, merge_predictions_by_specimen
from src.card_relevance_visual import TEST_RELEVANCE_CACHE_PATH, load_or_compute_relevance_flags
from src.gemini_batch_runner import pending_crop_count, run_wholeimage_batch
from src.gemini_transcribe import MODEL_FALLBACK_ORDER
from src.pipeline import load_manifest

TEST_CSV_PATH = os.path.join("data", "raw", "test.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
CARD_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "test_card_predictions.csv")
OUTPUT_PATH = os.path.join("data", "processed", "gemini", "test_wholeimage_predictions.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=3,
                         help="self-consistency samples per specimen (default 3)")
    parser.add_argument("--rpm", type=int, default=10,
                         help="max Gemini calls per minute (default 10, conservative for free tier; "
                              "hard-capped at src.gemini_transcribe.MAX_SAFE_RPM=14, since the "
                              "free-tier quota is 15/min per model)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process this many pending specimens in this invocation "
                              "(useful to cap a single day's run against daily quota)")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"),
                         help="defaults to the GEMINI_API_KEY environment variable")
    parser.add_argument("--models", default=",".join(MODEL_FALLBACK_ORDER),
                         help="comma-separated, tried in order, falling forward to the next "
                              "one once the current model's free-tier quota is exhausted "
                              f"(default: {','.join(MODEL_FALLBACK_ORDER)})")
    parser.add_argument("--force", action="store_true",
                         help="skip the per-card-pass-complete check (e.g. for a smoke test)")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("GEMINI_API_KEY is not set (env var or --api-key)")
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"[setup] reading test specimens from {TEST_CSV_PATH}...", flush=True)
    test_image_files = pd.read_csv(TEST_CSV_PATH)["image_file"].tolist()
    test_specimens = {os.path.splitext(f)[0] for f in test_image_files}
    print(f"[setup] {len(test_specimens)} test specimens loaded", flush=True)

    relevance_flags = load_or_compute_relevance_flags(test_specimens, TEST_RELEVANCE_CACHE_PATH)
    print(f"[setup] relevance flags ready for {len(relevance_flags)} card crops", flush=True)

    n_pending = pending_crop_count(test_image_files, relevance_flags, CARD_PREDICTIONS_PATH,
                                    manifest_path=MANIFEST_PATH)
    if n_pending and not args.force:
        raise SystemExit(
            f"refusing to start the whole-image fallback pass: {n_pending} test card crop(s) "
            "still haven't been transcribed by src.run_gemini_test. Running the fallback now "
            "would misclassify still-pending specimens as both-fields-MISSING. Finish "
            "`python -m src.run_gemini_test` first (it prints \"[done] nothing to do\" when "
            "complete), or pass --force to override.")

    print("[setup] merging per-card predictions...", flush=True)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    predictions_by_crop = load_predictions_by_crop(CARD_PREDICTIONS_PATH)
    merged = merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, test_image_files)

    run_wholeimage_batch(test_image_files, merged, OUTPUT_PATH,
                          args.n_samples, args.rpm, args.limit, args.api_key, models)


if __name__ == "__main__":
    main()
