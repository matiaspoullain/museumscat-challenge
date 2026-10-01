"""Per-card VLM transcription via the Gemini API, as an alternative to the
doctr + classify_field OCR path. Each call transcribes a single already-cropped
card image (from data/processed/card_crops/, produced by the existing SAM
detection pipeline) -- multi-card joining still happens downstream via
src.merge_records.assemble_field, unchanged.

Confidence isn't taken from the model's own self-reported number (LLMs are
reliably overconfident there); instead each card is sampled N times at
temperature > 0 and scored by self-consistency -- the fraction of samples that
agree with the modal answer. That raw score is intentionally uncalibrated here;
src.calibrate_gemini_confidence fits it against real dev-set correctness.

Network access (call_gemini_once) is the only function that touches the
google.genai client, so tests can exercise everything else -- prompt content
aside -- via an injected call_fn, no mocking framework needed.
"""
import collections
import json
import re
import time
from collections.abc import Callable

from google import genai
from google.genai import types

from src.merge_records import FieldScores
from src.metrics import ned

# "gemini-2.5-flash" (this project's original choice) returns 404 for new API
# keys, and "gemini-flash-latest" resolved to "gemini-3.6-flash", whose free
# tier is only 20 requests/day -- unusably small. "gemini-2.5-flash-lite" was
# a third fallback here too, but also 404s for new API keys ("no longer
# available to new users", confirmed 2026-08-08) -- kept in the rotation it
# would silently turn a graceful quota-exhausted stop into a confusing crash
# once the two models below are both exhausted, so it's removed rather than
# kept as a broken placeholder. The "-lite" variants have much larger
# free-tier daily quotas; src.gemini_batch_runner.ModelRotation tries these in
# order (most capable first) and falls forward to the next one once the
# current model's daily quota is exhausted (HTTP 429).
MODEL_FALLBACK_ORDER = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
DEFAULT_MODEL = MODEL_FALLBACK_ORDER[0]
FIELDS = ("verbatimDate", "verbatimLocality")

# One Gemini response: {field: text} for each of FIELDS.
FieldValues = dict[str, str]

# One card crop per call (multi-card joining happens downstream via
# src.merge_records), so the prompt only needs single-card confound rules --
# no pipe/multi-card instructions here.
PROMPT = """You are transcribing a single card from a pinned Danish Dung Beetle \
specimen label (1800s-present handwriting and print, sometimes faded or historical \
spelling). Transcribe exactly two fields from THIS card image only.

verbatimDate: the collection date on this card, transcribed exactly as written \
(digits, roman numerals, and separators as they appear -- do not reformat or \
normalize punctuation). If this card has no date, use "MISSING".

verbatimLocality: the collection locality (place name) on this card, transcribed \
exactly as written, preserving original spelling and abbreviations. If this card \
has no locality, use "MISSING".

Danish special letters are æ, ø, and å -- NOT ö, ä, or ü. If a handwritten or \
printed letter looks like it could be ø, transcribe it as ø, never as ö.

Do NOT transcribe the following as either field, even if it's the only text on \
this card -- if that's all this card has, both fields are "MISSING":
- Collector names, often prefixed "Coll." (e.g. "Coll. J. P. Johansen")
- Species determination info, often prefixed "det." or "Tilg."
- "Dania" -- this is the name of the specimen collection itself, not a locality
- Collection substrate/habitat notes like "i kogøding" ("in cow dung") or \
"Løv."/"Mus. Løv." ("i løv", i.e. "in leaf litter") -- not a locality
- Museum/institution boilerplate (e.g. "Statens Naturhistoriske Museum", \
"Københavns Universitet", "DaSSCo", "NHMD")
- Catalog numbers, page numbers, or a collector's monogram (only when it's \
clearly a person's initials, e.g. accompanying or styled like a "Coll." \
attribution) -- not a locality
- Do NOT treat every short 1-3 letter label as a bare initial: Danish locality \
labels are routinely abbreviated to just 1-3 letters on their own small card \
(e.g. "Kb" for København). A short capitalized label on its own card, with no \
"Coll." attribution nearby, is a place abbreviation and belongs in \
verbatimLocality exactly as written -- even if you don't recognize the \
specific abbreviation (e.g. "Ti", "Al"). This does NOT apply to "Løv."/"Mus. \
Løv." or "i kogøding" above -- those are excluded regardless of how short or \
standalone they look, even though they're written in the same short-label \
style as a real place abbreviation

Respond with ONLY a JSON object: {"verbatimDate": "...", "verbatimLocality": "..."}"""

# Used as a fallback for specimens where the per-card path found nothing at
# all (failed card detection, an over-eager confound filter, or a crop the
# model itself couldn't read) -- src.gemini_batch_runner.run_wholeimage_batch.
# Shares PROMPT's confound-exclusion rules verbatim (kept in sync by
# tests/test_gemini_transcribe.py), but this is the full photo, which may
# show more than one pinned specimen card.
WHOLE_IMAGE_PROMPT = """You are transcribing a photo of a pinned Danish Dung \
Beetle specimen (1800s-present handwriting and print, sometimes faded or \
historical spelling). This is the FULL photo, not a single card -- it may \
show one label, several separate labels on the same pin, or stray tags. \
Transcribe verbatimDate and verbatimLocality from ALL cards/labels visible.

verbatimDate: the collection date(s), transcribed exactly as written \
(digits, roman numerals, and separators as they appear -- do not reformat or \
normalize punctuation). If no card has a date, use "MISSING".

verbatimLocality: the collection locality/localities (place name), \
transcribed exactly as written, preserving original spelling and \
abbreviations. If no card has a locality, use "MISSING".

Danish special letters are æ, ø, and å -- NOT ö, ä, or ü. If a handwritten or \
printed letter looks like it could be ø, transcribe it as ø, never as ö.

If MULTIPLE cards each carry their own date or locality, join those values \
with " | " in reading order, e.g. "Ti | Kulhuset Jægerspris" -- do not merge \
them into one string or pick only one card.

Do NOT transcribe the following as either field, even if it's the only text \
visible -- if that's all there is, both fields are "MISSING":
- Collector names, often prefixed "Coll." (e.g. "Coll. J. P. Johansen")
- Species determination info, often prefixed "det." or "Tilg."
- "Dania" -- this is the name of the specimen collection itself, not a locality
- Collection substrate/habitat notes like "i kogøding" ("in cow dung") or \
"Løv."/"Mus. Løv." ("i løv", i.e. "in leaf litter") -- not a locality, even \
when it looks like a second pipe-joined value
- Museum/institution boilerplate (e.g. "Statens Naturhistoriske Museum", \
"Københavns Universitet", "DaSSCo", "NHMD"), including stamps or labels that \
repeat this on multiple cards in the same photo
- Catalog numbers, page numbers, or a collector's monogram (only when it's \
clearly a person's initials, e.g. accompanying or styled like a "Coll." \
attribution) -- not a locality
- Do NOT treat every short 1-3 letter label as a bare initial: Danish locality \
labels are routinely abbreviated to just 1-3 letters on their own small card \
(e.g. "Kb" for København). A short capitalized label on its own card, with no \
"Coll." attribution nearby, is a place abbreviation and belongs in \
verbatimLocality exactly as written -- even if you don't recognize the \
specific abbreviation (e.g. "Ti", "Al"). This does NOT apply to "Løv."/"Mus. \
Løv." or "i kogøding" above -- those are excluded regardless of how short or \
standalone they look, even though they're written in the same short-label \
style as a real place abbreviation

Respond with ONLY a JSON object: {"verbatimDate": "...", "verbatimLocality": "..."}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verbatimDate": {"type": "string"},
        "verbatimLocality": {"type": "string"},
    },
    "required": list(FIELDS),
}


# Free-tier RPM quota is 15 requests/minute for both MODEL_FALLBACK_ORDER
# models. Cap configured rpm below that (not equal to it) so clock skew
# between our local minute-window and the API's doesn't tip us over --
# src.gemini_batch_runner.run_batch/run_wholeimage_batch enforce this.
MAX_SAFE_RPM = 14


class RateLimiter:
    """Preemptive requests-per-minute throttle. Complements (doesn't replace)
    the Gemini client's own reactive retry-on-429 -- this avoids tripping rate
    limits in the first place, which matters when free-tier daily quota is the
    scarce resource."""

    def __init__(self, max_calls_per_minute: int) -> None:
        self.max_calls_per_minute = max_calls_per_minute
        self._call_times: collections.deque[float] = collections.deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._call_times and now - self._call_times[0] >= 60:
            self._call_times.popleft()
        if len(self._call_times) >= self.max_calls_per_minute:
            time.sleep(60 - (now - self._call_times[0]))
        self._call_times.append(time.monotonic())


# Some requests observed to hang far longer than a normal Gemini call
# (transport-level stall, not a 429) and never return -- REQUEST_TIMEOUT_MS
# bounds how long httpx waits before raising httpx.TimeoutException, so a
# stuck request gets cancelled instead of blocking the batch run forever.
# src.gemini_batch_runner._call_with_fallback retries the same model when
# that happens.
REQUEST_TIMEOUT_MS = 120_000


def build_client(api_key: str | None) -> genai.Client:
    # Short retry budget: a daily-quota 429 won't resolve by retrying the same
    # model within seconds, so we want it to surface quickly and let
    # ModelRotation switch models rather than burn a minute of backoff first.
    # The same retry_options also covers REQUEST_TIMEOUT_MS timeouts (the SDK
    # retries httpx.TimeoutException/ConnectError, not just retriable HTTP
    # status codes) before _call_with_fallback ever sees one.
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1.0, max_delay=5.0),
        ),
    )


def _parse_response_json(text: str) -> FieldValues:
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in Gemini response: {text!r}")
    parsed = json.loads(match.group(0))
    return {
        field: (str(parsed.get(field, "MISSING")).strip() or "MISSING")
        for field in FIELDS
    }


def call_gemini_once(client: genai.Client, image_bytes: bytes, mime_type: str = "image/jpeg",
                      model: str = DEFAULT_MODEL, temperature: float = 0.7,
                      rate_limiter: RateLimiter | None = None, prompt: str = PROMPT) -> FieldValues:
    if rate_limiter is not None:
        rate_limiter.wait()
    response = client.models.generate_content(
        model=model,
        # camelCase "mimeType" isn't a key of the SDK's BlobDict stub, but its pydantic model
        # accepts the alias at runtime.
        contents=[  # type: ignore[arg-type]
            {"inline_data": {"data": image_bytes, "mimeType": mime_type}},
        ],
        config=types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=temperature,
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
        ),
    )
    # response.text is None when Gemini returns no text part; not guarded here.
    return _parse_response_json(response.text)  # type: ignore[arg-type]


def _normalize_for_matching(text: str) -> str:
    return " ".join(text.strip().lower().split())


def self_consistency_scores(samples: list[FieldValues]) -> FieldScores:
    """samples: list of {field: text} dicts from repeated calls at temperature > 0.
    Returns {field: (modal_text, agreement_fraction)} -- agreement_fraction is the
    share of samples matching the modal (case/whitespace-normalized) answer, used
    as raw (uncalibrated) confidence."""
    scores: FieldScores = {}
    for field in FIELDS:
        texts = [sample[field] for sample in samples]
        counts = collections.Counter(_normalize_for_matching(t) for t in texts)
        modal_normalized, modal_count = counts.most_common(1)[0]
        modal_text = next(t for t in texts if _normalize_for_matching(t) == modal_normalized)
        scores[field] = (modal_text, modal_count / len(texts))
    return scores


def soft_consistency_scores(samples: list[FieldValues]) -> FieldScores:
    """Like self_consistency_scores, but each field's confidence is the average
    similarity (1 - normalized edit distance) between every sample and the modal
    answer, instead of the fraction of samples that match it exactly. A sample
    that's a one-character OCR slip away from the modal answer ("1897" vs
    "1887") still counts as partial agreement here, rather than the zero credit
    self_consistency_scores' exact-match count would give it."""
    scores: FieldScores = {}
    for field in FIELDS:
        texts = [sample[field] for sample in samples]
        counts = collections.Counter(_normalize_for_matching(t) for t in texts)
        modal_normalized, _ = counts.most_common(1)[0]
        modal_text = next(t for t in texts if _normalize_for_matching(t) == modal_normalized)
        is_date = field == "verbatimDate"
        similarities = [1.0 - ned(t, modal_text, is_date=is_date) for t in texts]
        scores[field] = (modal_text, sum(similarities) / len(similarities))
    return scores


def transcribe_card_with_confidence(image_bytes: bytes | None = None, api_key: str | None = None,
                                     n_samples: int = 3, model: str = DEFAULT_MODEL,
                                     temperature: float = 0.7, mime_type: str = "image/jpeg",
                                     rate_limiter: RateLimiter | None = None,
                                     call_fn: Callable[[], FieldValues] | None = None,
                                     ) -> tuple[FieldScores, list[FieldValues]]:
    """Returns (scores, samples) where scores is {field: (text, raw_confidence)}
    (the same shape src.merge_records.assemble_field expects a list of) and
    samples is the raw list of per-call dicts, kept for later recalibration
    without re-querying the API.

    call_fn, if given, is a zero-arg callable returning a {field: text} dict --
    used by tests to bypass the network entirely."""
    if call_fn is None:
        client = build_client(api_key)

        def call_fn() -> FieldValues:
            # image_bytes is only None when the caller also passes call_fn.
            return call_gemini_once(client, image_bytes, mime_type=mime_type, model=model,  # type: ignore[arg-type]
                                     temperature=temperature, rate_limiter=rate_limiter)

    samples = [call_fn() for _ in range(n_samples)]
    return self_consistency_scores(samples), samples
