"""Shared resumable batch-transcription runner used by both src.run_gemini_dev
(200 labeled dev specimens) and src.run_gemini_test (3300 test specimens) --
only the specimen list, relevance flags, and output path differ between them.

Resumable by construction: appends to output_path and skips crop_filenames
already present there, so a run interrupted by a daily free-tier quota limit
picks up where it left off on the next invocation.
"""
import csv
import json
import os

from google.genai import errors as genai_errors

from src.gemini_transcribe import RateLimiter, build_client, call_gemini_once, transcribe_card_with_confidence
from src.pipeline import load_manifest

MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
CROPS_IMAGES_DIR = os.path.join("data", "processed", "card_crops", "images")

OUTPUT_COLUMNS = [
    "crop_filename",
    "source_image_file",
    "verbatimDate",
    "verbatimDate_confidence",
    "verbatimLocality",
    "verbatimLocality_confidence",
    "models_used",
    "raw_samples_json",
]


def pending_crops(image_files, relevance_flags, manifest_path=MANIFEST_PATH):
    """Card crops belonging to image_files, skipping any the relevance
    classifier flags as confound (saves quota; a card missing entirely from
    the output is treated downstream exactly like an empty card list)."""
    manifest_by_image = load_manifest(manifest_path)
    crops = []
    for image_file in image_files:
        for crop_filename in manifest_by_image.get(image_file, []):
            if relevance_flags.get(crop_filename, True):
                crops.append((crop_filename, image_file))
    return crops


def already_done(output_path):
    if not os.path.exists(output_path):
        return set()
    with open(output_path, encoding="utf-8") as f:
        return {row["crop_filename"] for row in csv.DictReader(f)}


def is_quota_exhausted(error):
    return isinstance(error, genai_errors.ClientError) and error.code == 429


class ModelRotation:
    """Tries models in order (most capable first) and falls forward -- never
    back -- to the next one once the current model's free-tier quota is
    exhausted. Shared across a whole batch run, so once a model is exhausted
    every later card skips straight past it instead of re-discovering that on
    every single call."""

    def __init__(self, models):
        models = list(models)
        if not models:
            raise ValueError("models must not be empty")
        self.models = models
        self.index = 0

    @property
    def current(self):
        return self.models[self.index]

    def advance(self):
        if self.index >= len(self.models) - 1:
            return False
        self.index += 1
        print(f"[model] {self.models[self.index - 1]} quota exhausted, "
              f"switching to {self.models[self.index]}", flush=True)
        return True


def _call_with_fallback(client, image_bytes, model_rotation, rate_limiter):
    """Same return shape as call_gemini_once, plus "_model" recording which
    model in the rotation actually produced this sample -- self_consistency_scores
    only reads the field keys it knows about, so this rides along harmlessly
    into transcribe_card_with_confidence's raw samples list."""
    while True:
        model = model_rotation.current
        try:
            result = call_gemini_once(client, image_bytes, model=model, rate_limiter=rate_limiter)
            return {**result, "_model": model}
        except genai_errors.ClientError as e:
            if is_quota_exhausted(e) and model_rotation.advance():
                continue
            raise


def run_batch(image_files, relevance_flags, output_path, n_samples, rpm, limit, api_key, models,
              crops_images_dir=CROPS_IMAGES_DIR):
    if isinstance(models, str):
        models = [models]
    model_rotation = ModelRotation(models)

    print(f"[setup] {len(image_files)} specimens given", flush=True)
    all_pending = pending_crops(image_files, relevance_flags)
    n_confound = sum(1 for is_relevant in relevance_flags.values() if not is_relevant)
    print(f"[setup] {len(all_pending)} relevant card crops found "
          f"({n_confound} flagged confound and skipped)", flush=True)

    done = already_done(output_path)
    pending = [(crop, image_file) for crop, image_file in all_pending if crop not in done]
    print(f"[setup] {len(done)} crops already transcribed in {output_path} (resuming, will skip)", flush=True)
    if limit is not None:
        pending = pending[:limit]
        print(f"[setup] --limit {limit} applied: processing {len(pending)} crops this run", flush=True)
    if not pending:
        print("[done] nothing to do (all relevant crops already transcribed)", flush=True)
        return

    print(f"[setup] building Gemini client (models={model_rotation.models}, "
          f"n_samples={n_samples}, rpm={rpm})...", flush=True)
    client = build_client(api_key)
    rate_limiter = RateLimiter(max_calls_per_minute=rpm)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_header = not os.path.exists(output_path)
    print(f"[run] starting: {len(pending)} crops to transcribe", flush=True)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()

        for i, (crop_filename, source_image_file) in enumerate(pending, start=1):
            print(f"[{i}/{len(pending)}] {crop_filename}: requesting {n_samples} sample(s) "
                  f"from {model_rotation.current}...", flush=True)
            with open(os.path.join(crops_images_dir, crop_filename), "rb") as img_f:
                image_bytes = img_f.read()

            def call_fn(image_bytes=image_bytes):
                return _call_with_fallback(client, image_bytes, model_rotation, rate_limiter)

            scores, samples = transcribe_card_with_confidence(n_samples=n_samples, call_fn=call_fn)
            models_used = sorted({sample.get("_model", "") for sample in samples} - {""})
            row = {
                "crop_filename": crop_filename,
                "source_image_file": source_image_file,
                "verbatimDate": scores["verbatimDate"][0],
                "verbatimDate_confidence": scores["verbatimDate"][1],
                "verbatimLocality": scores["verbatimLocality"][0],
                "verbatimLocality_confidence": scores["verbatimLocality"][1],
                "models_used": ",".join(models_used),
                "raw_samples_json": json.dumps(samples),
            }
            writer.writerow(row)
            f.flush()
            print(f"[{i}/{len(pending)}] {crop_filename}: "
                  f"date={row['verbatimDate']!r} ({row['verbatimDate_confidence']:.2f}) "
                  f"locality={row['verbatimLocality']!r} ({row['verbatimLocality_confidence']:.2f}) "
                  f"models={row['models_used']}",
                  flush=True)

    print(f"[done] wrote {len(pending)} new rows to {output_path}", flush=True)
