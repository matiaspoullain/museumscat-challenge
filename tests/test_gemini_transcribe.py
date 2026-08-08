import time

import pytest

from src.gemini_transcribe import (
    PROMPT,
    WHOLE_IMAGE_PROMPT,
    RateLimiter,
    _normalize_for_matching,
    _parse_response_json,
    call_gemini_once,
    self_consistency_scores,
    soft_consistency_scores,
    transcribe_card_with_confidence,
)


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return _FakeResponse('{"verbatimDate": "MISSING", "verbatimLocality": "MISSING"}')


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_call_gemini_once_defaults_to_card_prompt():
    client = _FakeClient()
    call_gemini_once(client, b"fake-bytes")
    assert client.models.calls[0]["config"].system_instruction == PROMPT


def test_call_gemini_once_forwards_custom_prompt():
    client = _FakeClient()
    call_gemini_once(client, b"fake-bytes", prompt=WHOLE_IMAGE_PROMPT)
    assert client.models.calls[0]["config"].system_instruction == WHOLE_IMAGE_PROMPT


# Confound-exclusion rules that WHOLE_IMAGE_PROMPT must keep in sync with
# PROMPT -- a divergence here would mean the whole-image fallback stops
# excluding a category the per-card prompt already knows to drop.
_CONFOUND_RULE_SUBSTRINGS = [
    "Coll.", "det.", "Tilg.", "Dania", "i kogøding", "Løv.",
    "Statens Naturhistoriske Museum", "Københavns Universitet", "DaSSCo", "NHMD",
]


def test_whole_image_prompt_keeps_confound_rules_in_sync_with_card_prompt():
    for substring in _CONFOUND_RULE_SUBSTRINGS:
        assert substring in PROMPT, f"{substring!r} missing from PROMPT itself"
        assert substring in WHOLE_IMAGE_PROMPT, f"{substring!r} missing from WHOLE_IMAGE_PROMPT"


def test_parse_response_json_plain():
    result = _parse_response_json('{"verbatimDate": "27.5.2022", "verbatimLocality": "Kb"}')
    assert result == {"verbatimDate": "27.5.2022", "verbatimLocality": "Kb"}


def test_parse_response_json_strips_markdown_fence():
    text = '```json\n{"verbatimDate": "MISSING", "verbatimLocality": "Dyrehaven"}\n```'
    result = _parse_response_json(text)
    assert result == {"verbatimDate": "MISSING", "verbatimLocality": "Dyrehaven"}


def test_parse_response_json_missing_key_defaults_to_missing():
    result = _parse_response_json('{"verbatimDate": "1.7.2000"}')
    assert result == {"verbatimDate": "1.7.2000", "verbatimLocality": "MISSING"}


def test_parse_response_json_empty_string_becomes_missing():
    result = _parse_response_json('{"verbatimDate": "", "verbatimLocality": "  "}')
    assert result == {"verbatimDate": "MISSING", "verbatimLocality": "MISSING"}


def test_normalize_for_matching_case_and_whitespace_insensitive():
    assert _normalize_for_matching("  Kb  ") == _normalize_for_matching("kb")
    assert _normalize_for_matching("Ti  Kulhuset") == _normalize_for_matching("ti kulhuset")


def test_self_consistency_scores_unanimous_is_full_confidence():
    samples = [{"verbatimDate": "27.5.2022", "verbatimLocality": "Kb"}] * 3
    scores = self_consistency_scores(samples)
    assert scores["verbatimDate"] == ("27.5.2022", 1.0)
    assert scores["verbatimLocality"] == ("Kb", 1.0)


def test_self_consistency_scores_majority_wins():
    samples = [
        {"verbatimDate": "27.5.2022", "verbatimLocality": "Kb"},
        {"verbatimDate": "27.5.2022", "verbatimLocality": "Kb"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Kob"},
    ]
    scores = self_consistency_scores(samples)
    assert scores["verbatimDate"] == ("27.5.2022", 2 / 3)
    assert scores["verbatimLocality"] == ("Kb", 2 / 3)


def test_self_consistency_scores_case_insensitive_agreement():
    samples = [
        {"verbatimDate": "MISSING", "verbatimLocality": "Kb"},
        {"verbatimDate": "MISSING", "verbatimLocality": "kb"},
    ]
    scores = self_consistency_scores(samples)
    # both samples agree once normalized; the returned text keeps original casing
    # of whichever sample matched the modal form first
    assert scores["verbatimLocality"] == ("Kb", 1.0)


def test_self_consistency_scores_all_disagree_is_low_confidence():
    samples = [
        {"verbatimDate": "MISSING", "verbatimLocality": "Kb"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Ti"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Rude Skov"},
    ]
    scores = self_consistency_scores(samples)
    assert scores["verbatimDate"] == ("MISSING", 1.0)
    _, confidence = scores["verbatimLocality"]
    assert confidence == 1 / 3


def test_soft_consistency_scores_unanimous_is_full_confidence():
    samples = [{"verbatimDate": "27.5.2022", "verbatimLocality": "Kb"}] * 3
    scores = soft_consistency_scores(samples)
    assert scores["verbatimDate"] == ("27.5.2022", 1.0)
    assert scores["verbatimLocality"] == ("Kb", 1.0)


def test_soft_consistency_scores_gives_partial_credit_for_near_miss():
    # "Kob" is one edit from the modal "Kb" (ned=1/3 -> similarity=2/3), so the
    # average similarity across all 3 samples is (1 + 1 + 2/3) / 3 = 8/9 --
    # strictly higher than self_consistency_scores' exact-match fraction (2/3)
    # would give the same samples, since that scores "Kob" as zero credit.
    samples = [
        {"verbatimDate": "MISSING", "verbatimLocality": "Kb"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Kb"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Kob"},
    ]
    scores = soft_consistency_scores(samples)
    text, confidence = scores["verbatimLocality"]
    assert text == "Kb"
    assert confidence == pytest.approx(8 / 9)


def test_soft_consistency_scores_all_disagree_is_still_above_zero():
    samples = [
        {"verbatimDate": "MISSING", "verbatimLocality": "Kb"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Ti"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Rude Skov"},
    ]
    scores = soft_consistency_scores(samples)
    _, confidence = scores["verbatimLocality"]
    # "Kb" vs itself is similarity 1.0; "Ti" shares no characters and gets
    # similarity 0.0; "Rude Skov" shares 'k' so gets partial similarity.
    # Average is (1.0 + 0.0 + similarity) / 3, which is above 1/3 due to
    # partial overlap with "Rude Skov".
    assert confidence > 1 / 3
    assert confidence == pytest.approx(0.3703703703703704)


def test_transcribe_card_with_confidence_uses_injected_call_fn():
    responses = iter([
        {"verbatimDate": "27.5.2022", "verbatimLocality": "Kb"},
        {"verbatimDate": "27.5.2022", "verbatimLocality": "Kb"},
    ])
    scores, samples = transcribe_card_with_confidence(n_samples=2, call_fn=lambda: next(responses))
    assert scores["verbatimDate"] == ("27.5.2022", 1.0)
    assert scores["verbatimLocality"] == ("Kb", 1.0)
    assert len(samples) == 2


def test_rate_limiter_does_not_block_under_the_limit():
    limiter = RateLimiter(max_calls_per_minute=1000)
    start = time.perf_counter()
    for _ in range(5):
        limiter.wait()
    assert time.perf_counter() - start < 0.5
    assert len(limiter._call_times) == 5
