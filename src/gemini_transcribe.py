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

from google import genai
from google.genai import types

# "gemini-2.5-flash" (this project's original choice) returns 404 for new API
# keys, and "gemini-flash-latest" resolved to "gemini-3.6-flash", whose free
# tier is only 20 requests/day -- unusably small. The "-lite" variants have
# much larger free-tier daily quotas; src.gemini_batch_runner.ModelRotation
# tries these in order (most capable first) and falls forward to the next one
# once the current model's daily quota is exhausted (HTTP 429).
MODEL_FALLBACK_ORDER = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]
DEFAULT_MODEL = MODEL_FALLBACK_ORDER[0]
FIELDS = ("verbatimDate", "verbatimLocality")

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

Do NOT transcribe the following as either field, even if it's the only text on \
this card -- if that's all this card has, both fields are "MISSING":
- Collector names, often prefixed "Coll." (e.g. "Coll. J. P. Johansen")
- Species determination info, often prefixed "det." or "Tilg."
- "Dania" -- this is the name of the specimen collection itself, not a locality
- Collection substrate notes like "i kogøding" ("in cow dung") -- not a locality
- Museum/institution boilerplate (e.g. "Statens Naturhistoriske Museum", \
"Københavns Universitet", "DaSSCo", "NHMD")
- Catalog numbers, page numbers, or bare initials with no place/date meaning

Respond with ONLY a JSON object: {"verbatimDate": "...", "verbatimLocality": "..."}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verbatimDate": {"type": "string"},
        "verbatimLocality": {"type": "string"},
    },
    "required": list(FIELDS),
}


class RateLimiter:
    """Preemptive requests-per-minute throttle. Complements (doesn't replace)
    the Gemini client's own reactive retry-on-429 -- this avoids tripping rate
    limits in the first place, which matters when free-tier daily quota is the
    scarce resource."""

    def __init__(self, max_calls_per_minute):
        self.max_calls_per_minute = max_calls_per_minute
        self._call_times = collections.deque()

    def wait(self):
        now = time.monotonic()
        while self._call_times and now - self._call_times[0] >= 60:
            self._call_times.popleft()
        if len(self._call_times) >= self.max_calls_per_minute:
            time.sleep(60 - (now - self._call_times[0]))
        self._call_times.append(time.monotonic())


def build_client(api_key):
    # Short retry budget: a daily-quota 429 won't resolve by retrying the same
    # model within seconds, so we want it to surface quickly and let
    # ModelRotation switch models rather than burn a minute of backoff first.
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1.0, max_delay=5.0)
        ),
    )


def _parse_response_json(text):
    match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in Gemini response: {text!r}")
    parsed = json.loads(match.group(0))
    return {
        field: (str(parsed.get(field, "MISSING")).strip() or "MISSING")
        for field in FIELDS
    }


def call_gemini_once(client, image_bytes, mime_type="image/jpeg", model=DEFAULT_MODEL,
                      temperature=0.7, rate_limiter=None):
    if rate_limiter is not None:
        rate_limiter.wait()
    response = client.models.generate_content(
        model=model,
        contents=[
            PROMPT,
            {"inline_data": {"data": image_bytes, "mimeType": mime_type}},
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
        ),
    )
    return _parse_response_json(response.text)


def _normalize_for_matching(text):
    return " ".join(text.strip().lower().split())


def self_consistency_scores(samples):
    """samples: list of {field: text} dicts from repeated calls at temperature > 0.
    Returns {field: (modal_text, agreement_fraction)} -- agreement_fraction is the
    share of samples matching the modal (case/whitespace-normalized) answer, used
    as raw (uncalibrated) confidence."""
    scores = {}
    for field in FIELDS:
        texts = [sample[field] for sample in samples]
        counts = collections.Counter(_normalize_for_matching(t) for t in texts)
        modal_normalized, modal_count = counts.most_common(1)[0]
        modal_text = next(t for t in texts if _normalize_for_matching(t) == modal_normalized)
        scores[field] = (modal_text, modal_count / len(texts))
    return scores


def transcribe_card_with_confidence(image_bytes=None, api_key=None, n_samples=3,
                                     model=DEFAULT_MODEL, temperature=0.7, mime_type="image/jpeg",
                                     rate_limiter=None, call_fn=None):
    """Returns (scores, samples) where scores is {field: (text, raw_confidence)}
    (the same shape src.merge_records.assemble_field expects a list of) and
    samples is the raw list of per-call dicts, kept for later recalibration
    without re-querying the API.

    call_fn, if given, is a zero-arg callable returning a {field: text} dict --
    used by tests to bypass the network entirely."""
    if call_fn is None:
        client = build_client(api_key)

        def call_fn():
            return call_gemini_once(client, image_bytes, mime_type=mime_type, model=model,
                                     temperature=temperature, rate_limiter=rate_limiter)

    samples = [call_fn() for _ in range(n_samples)]
    return self_consistency_scores(samples), samples
