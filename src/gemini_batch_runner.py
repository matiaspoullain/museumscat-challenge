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
import time
from collections.abc import Iterable
from typing import cast

import httpx
from google import genai
from google.genai import errors as genai_errors

from src.gemini_transcribe import (
    MAX_SAFE_RPM,
    PROMPT,
    RateLimiter,
    WHOLE_IMAGE_PROMPT,
    FieldValues,
    build_client,
    call_gemini_once,
    transcribe_card_with_confidence,
)
from src.merge_records import FieldScores
from src.pipeline import load_manifest

# How long to back off before retrying the same model after a per-minute
# (RPM) 429 -- long enough for the 60s sliding window Gemini enforces to
# clear, even though src.gemini_transcribe.RateLimiter's preemptive throttle
# should mostly prevent this from happening at all.
RPM_BACKOFF_SECONDS = 60

# How long to back off before retrying the same model after a 5xx
# ServerError (observed 2026-08-09: intermittent "503 UNAVAILABLE / Unable
# to process input image", which Gemini's own error message says to retry).
# Much shorter than RPM_BACKOFF_SECONDS since this isn't a quota limit --
# just a transient server-side hiccup.
SERVER_ERROR_BACKOFF_SECONDS = 5

MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
CROPS_IMAGES_DIR = os.path.join("data", "processed", "card_crops", "images")
WHOLEIMAGE_IMAGES_DIR = os.path.join("data", "raw", "images")

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

OUTPUT_COLUMNS_WHOLEIMAGE = [
    "source_image_file",
    "verbatimDate",
    "verbatimDate_confidence",
    "verbatimLocality",
    "verbatimLocality_confidence",
    "models_used",
    "raw_samples_json",
]

FIELDS = ("verbatimDate", "verbatimLocality")


def pending_crops(image_files: list[str], relevance_flags: dict[str, bool],
                  manifest_path: str = MANIFEST_PATH) -> list[tuple[str, str]]:
    """Card crops belonging to image_files, skipping any the relevance
    classifier flags as confound (saves quota; a card missing entirely from
    the output is treated downstream exactly like an empty card list)."""
    manifest_by_image = load_manifest(manifest_path)
    crops: list[tuple[str, str]] = []
    for image_file in image_files:
        for crop_filename in manifest_by_image.get(image_file, []):
            if relevance_flags.get(crop_filename, True):
                crops.append((crop_filename, image_file))
    return crops


def already_done(output_path: str, key_column: str = "crop_filename") -> set[str]:
    if not os.path.exists(output_path):
        return set()
    with open(output_path, encoding="utf-8") as f:
        return {row[key_column] for row in csv.DictReader(f)}


def pending_crop_count(image_files: list[str], relevance_flags: dict[str, bool], output_path: str,
                       manifest_path: str = MANIFEST_PATH) -> int:
    """How many relevant card crops for image_files still have no row in
    output_path -- used to gate a whole-image fallback pass on the per-card
    pass being fully finished first. merge_predictions_by_specimen can't tell
    "never sent to Gemini yet" from "sent, and this card had no value" (both
    just look like no contribution), so running the fallback pass while any
    crop is still pending would misclassify those specimens as genuinely
    both-fields-MISSING and spend fallback quota on them prematurely."""
    all_pending = pending_crops(image_files, relevance_flags, manifest_path=manifest_path)
    done = already_done(output_path)
    return sum(1 for crop_filename, _ in all_pending if crop_filename not in done)


def wholeimage_pending(image_files: list[str], merged_predictions: dict[str, FieldScores]) -> list[str]:
    """image_files whose merged per-card prediction (src.calibrate_gemini_confidence.
    merge_predictions_by_specimen's output shape) has both fields "MISSING" --
    the per-card path found nothing for these, so a whole-image fallback call
    is worth the extra quota. A specimen absent from merged_predictions is
    skipped (nothing to compare against)."""
    pending: list[str] = []
    for image_file in image_files:
        fields = merged_predictions.get(image_file)
        if fields is None:
            continue
        if all(fields[field][0] == "MISSING" for field in FIELDS):
            pending.append(image_file)
    return pending


def is_quota_exhausted(error: Exception) -> bool:
    return isinstance(error, genai_errors.ClientError) and error.code == 429


def _quota_violation_ids(error: Exception) -> list[str]:
    """quotaId strings (e.g. "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    from a 429's structured google.rpc.QuotaFailure detail, if the response body
    included one. Empty list if not present/parseable."""
    body = getattr(error, "details", None)
    if not isinstance(body, dict):
        return []
    inner = body.get("error", body)
    if not isinstance(inner, dict):
        return []
    quota_ids: list[str] = []
    for detail in inner.get("details") or []:
        for violation in detail.get("violations") or []:
            quota_id = violation.get("quotaId")
            if quota_id:
                quota_ids.append(quota_id)
    return quota_ids


def is_daily_quota_exhausted(error: Exception) -> bool:
    """True only for a 429 that's actually a per-day quota limit -- as opposed
    to a per-minute (RPM) 429, which src.gemini_transcribe.RateLimiter's
    preemptive throttle should mostly prevent but can still slip through
    (clock skew against the API's window, a burst right at a minute
    boundary). Only a genuine daily-quota 429 should make ModelRotation fall
    forward to the next model -- a per-minute 429 means the current model
    still has quota today, it just needs a brief pause before retrying."""
    if not is_quota_exhausted(error):
        return False
    quota_ids = _quota_violation_ids(error)
    if quota_ids:
        return any("day" in quota_id.lower() for quota_id in quota_ids)
    # No structured QuotaFailure detail to inspect -- fall back to the
    # message text, which free-tier responses also spell out in words.
    # Narrowed to a ClientError by is_quota_exhausted above (cast is a runtime no-op).
    message = (cast(genai_errors.ClientError, error).message or "").lower()
    if "per day" in message or "daily" in message:
        return True
    if "per minute" in message:
        return False
    # Genuinely ambiguous: treat as a transient per-minute condition rather
    # than burning through the model rotation on an error we can't positively
    # identify as a daily-quota exhaustion.
    return False


class AllModelsExhaustedError(Exception):
    """Raised by _call_with_fallback when every model in the rotation has hit
    its free-tier daily quota -- distinct from a bare ClientError so
    run_batch/run_wholeimage_batch can stop the batch cleanly (rows already
    written are flushed per-row, so nothing is lost) instead of letting an
    unhandled traceback crash the whole run."""


class ModelRotation:
    """Tries models in order (most capable first) and falls forward -- never
    back -- to the next one once the current model's free-tier DAILY quota is
    exhausted (see is_daily_quota_exhausted -- a per-minute/RPM 429 does not
    advance this). Shared across a whole batch run, so once a model is
    exhausted every later card skips straight past it instead of
    re-discovering that on every single call."""

    def __init__(self, models: Iterable[str]) -> None:
        models = list(models)
        if not models:
            raise ValueError("models must not be empty")
        self.models = models
        self.index = 0

    @property
    def current(self) -> str:
        return self.models[self.index]

    def advance(self) -> bool:
        if self.index >= len(self.models) - 1:
            return False
        self.index += 1
        print(f"[model] {self.models[self.index - 1]} quota exhausted, "
              f"switching to {self.models[self.index]}", flush=True)
        return True


def _call_with_fallback(client: genai.Client, image_bytes: bytes, model_rotation: ModelRotation,
                        rate_limiter: RateLimiter | None, prompt: str = PROMPT) -> FieldValues:
    """Same return shape as call_gemini_once, plus "_model" recording which
    model in the rotation actually produced this sample -- self_consistency_scores
    only reads the field keys it knows about, so this rides along harmlessly
    into transcribe_card_with_confidence's raw samples list.

    Model rotation only advances on a genuine daily-quota 429
    (is_daily_quota_exhausted) -- a per-minute (RPM) 429 retries the SAME
    model after RPM_BACKOFF_SECONDS instead, since the model isn't actually
    out of quota for the day. A hung request (src.gemini_transcribe.
    REQUEST_TIMEOUT_MS exceeded, or a transport-level connect failure) is
    also just retried on the same model, after build_client's own
    retry_options budget is exhausted -- it's not a quota signal at all. A
    5xx ServerError (e.g. intermittent "503 UNAVAILABLE / Unable to process
    input image") is likewise transient and not a quota signal, so it's
    retried on the same model too, after a short SERVER_ERROR_BACKOFF_SECONDS
    pause."""
    while True:
        model = model_rotation.current
        try:
            result = call_gemini_once(client, image_bytes, model=model, rate_limiter=rate_limiter,
                                       prompt=prompt)
            return {**result, "_model": model}
        except (httpx.TimeoutException, httpx.ConnectError):
            print(f"[timeout] {model} request stalled past the retry budget -- "
                  "cancelling and retrying the same model...", flush=True)
            continue
        except genai_errors.ServerError as e:
            print(f"[server-error] {model} returned {e.code} ({e.message}) -- backing off "
                  f"{SERVER_ERROR_BACKOFF_SECONDS}s and retrying the same model...", flush=True)
            time.sleep(SERVER_ERROR_BACKOFF_SECONDS)
            continue
        except genai_errors.ClientError as e:
            if not is_quota_exhausted(e):
                raise
            if is_daily_quota_exhausted(e):
                if model_rotation.advance():
                    continue
                raise AllModelsExhaustedError(str(e)) from e
            print(f"[rate-limit] {model} hit a per-minute 429 -- backing off "
                  f"{RPM_BACKOFF_SECONDS}s and retrying the same model (not switching)...",
                  flush=True)
            time.sleep(RPM_BACKOFF_SECONDS)


def _validate_rpm(rpm: int) -> None:
    if rpm > MAX_SAFE_RPM:
        raise ValueError(f"rpm={rpm} exceeds MAX_SAFE_RPM={MAX_SAFE_RPM} -- Gemini free-tier "
                          "quota is 15 requests/minute per model, and this stays a safety "
                          "margin below that rather than equal to it")


def run_batch(image_files: list[str], relevance_flags: dict[str, bool], output_path: str,
              n_samples: int, rpm: int, limit: int | None, api_key: str | None,
              models: str | Iterable[str], crops_images_dir: str = CROPS_IMAGES_DIR) -> None:
    _validate_rpm(rpm)
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

            def call_fn(image_bytes: bytes = image_bytes) -> FieldValues:
                return _call_with_fallback(client, image_bytes, model_rotation, rate_limiter)

            try:
                scores, samples = transcribe_card_with_confidence(n_samples=n_samples, call_fn=call_fn)
            except AllModelsExhaustedError:
                print(f"[done] every model in the rotation hit its daily quota after "
                      f"{i - 1}/{len(pending)} crops this run -- stopping here, resume by "
                      "rerunning later", flush=True)
                return
            models_used = sorted({sample.get("_model", "") for sample in samples} - {""})
            row: dict[str, str | float] = {
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


def run_wholeimage_batch(image_files: list[str], merged_predictions: dict[str, FieldScores],
                         output_path: str, n_samples: int, rpm: int, limit: int | None,
                         api_key: str | None, models: str | Iterable[str],
                         images_dir: str = WHOLEIMAGE_IMAGES_DIR) -> None:
    """Whole-image fallback for specimens the per-card pass left both-fields
    MISSING -- same resumable shape as run_batch (skip-on-resume via
    already_done, flush-per-row, model rotation), but reads the original
    uncropped photo and uses WHOLE_IMAGE_PROMPT instead of a card crop."""
    _validate_rpm(rpm)
    if isinstance(models, str):
        models = [models]
    model_rotation = ModelRotation(models)

    print(f"[setup] {len(image_files)} specimens given", flush=True)
    all_pending = wholeimage_pending(image_files, merged_predictions)
    print(f"[setup] {len(all_pending)} specimens still both-MISSING after the per-card pass", flush=True)

    done = already_done(output_path, key_column="source_image_file")
    pending = [image_file for image_file in all_pending if image_file not in done]
    print(f"[setup] {len(done)} specimens already transcribed in {output_path} (resuming, will skip)",
          flush=True)
    if limit is not None:
        pending = pending[:limit]
        print(f"[setup] --limit {limit} applied: processing {len(pending)} specimens this run", flush=True)
    if not pending:
        print("[done] nothing to do (all both-MISSING specimens already transcribed)", flush=True)
        return

    print(f"[setup] building Gemini client (models={model_rotation.models}, "
          f"n_samples={n_samples}, rpm={rpm})...", flush=True)
    client = build_client(api_key)
    rate_limiter = RateLimiter(max_calls_per_minute=rpm)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_header = not os.path.exists(output_path)
    print(f"[run] starting: {len(pending)} specimens to transcribe (whole image)", flush=True)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS_WHOLEIMAGE)
        if write_header:
            writer.writeheader()

        for i, source_image_file in enumerate(pending, start=1):
            print(f"[{i}/{len(pending)}] {source_image_file}: requesting {n_samples} sample(s) "
                  f"from {model_rotation.current}...", flush=True)
            with open(os.path.join(images_dir, source_image_file), "rb") as img_f:
                image_bytes = img_f.read()

            def call_fn(image_bytes: bytes = image_bytes) -> FieldValues:
                return _call_with_fallback(client, image_bytes, model_rotation, rate_limiter,
                                            prompt=WHOLE_IMAGE_PROMPT)

            try:
                scores, samples = transcribe_card_with_confidence(n_samples=n_samples, call_fn=call_fn)
            except AllModelsExhaustedError:
                print(f"[done] every model in the rotation hit its daily quota after "
                      f"{i - 1}/{len(pending)} specimens this run -- stopping here, resume by "
                      "rerunning later", flush=True)
                return
            models_used = sorted({sample.get("_model", "") for sample in samples} - {""})
            row: dict[str, str | float] = {
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
            print(f"[{i}/{len(pending)}] {source_image_file}: "
                  f"date={row['verbatimDate']!r} ({row['verbatimDate_confidence']:.2f}) "
                  f"locality={row['verbatimLocality']!r} ({row['verbatimLocality_confidence']:.2f}) "
                  f"models={row['models_used']}",
                  flush=True)

    print(f"[done] wrote {len(pending)} new rows to {output_path}", flush=True)
