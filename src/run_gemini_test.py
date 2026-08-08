"""Batch-transcribes card crops belonging to the ~3300 test.csv specimens
through the Gemini API. Same resumable pattern as src.run_gemini_dev (see
src.gemini_batch_runner) -- meant to be run/resumed across multiple days to
respect Gemini free-tier daily quota, not completed in one sitting.

Only run this after src.evaluate_gemini_dev shows the per-card Gemini approach
beating the classical pipeline's 0.576 dev-set AURC baseline -- otherwise this
burns days of free-tier quota on an approach that isn't actually better.

Usage:
    conda run -n museumscat-challenge python -m src.run_gemini_test \
        [--n-samples 3] [--rpm 10] [--limit 200] [--models gemini-3.5-flash-lite,gemini-3.1-flash-lite]

Requires GEMINI_API_KEY in the environment (or --api-key). Re-run the same
command daily/as needed to resume after hitting the daily quota -- already
transcribed crops are skipped automatically.
"""
import argparse
import os

import pandas as pd

from src.card_relevance_visual import compute_relevance_flags_for_specimens
from src.gemini_batch_runner import run_batch
from src.gemini_transcribe import MODEL_FALLBACK_ORDER

TEST_CSV_PATH = os.path.join("data", "raw", "test.csv")
OUTPUT_PATH = os.path.join("data", "processed", "gemini", "test_card_predictions.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=3,
                         help="self-consistency samples per card (default 3)")
    parser.add_argument("--rpm", type=int, default=10,
                         help="max Gemini calls per minute (default 10, conservative for free tier; "
                              "hard-capped at src.gemini_transcribe.MAX_SAFE_RPM=14, since the "
                              "free-tier quota is 15/min per model)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process this many pending crops in this invocation "
                              "(useful to cap a single day's run against daily quota)")
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

    print(f"[setup] reading test specimens from {TEST_CSV_PATH}...", flush=True)
    test_image_files = pd.read_csv(TEST_CSV_PATH)["image_file"].tolist()
    test_specimens = {os.path.splitext(f)[0] for f in test_image_files}
    print(f"[setup] {len(test_specimens)} test specimens loaded", flush=True)

    print("[setup] computing card relevance flags (visual classifier)...", flush=True)
    relevance_flags = compute_relevance_flags_for_specimens(test_specimens)
    print(f"[setup] relevance flags computed for {len(relevance_flags)} card crops", flush=True)

    run_batch(test_image_files, relevance_flags, OUTPUT_PATH,
              args.n_samples, args.rpm, args.limit, args.api_key, models)


if __name__ == "__main__":
    main()
