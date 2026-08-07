import time

from src.gemini_transcribe import (
    RateLimiter,
    _normalize_for_matching,
    _parse_response_json,
    self_consistency_scores,
    transcribe_card_with_confidence,
)


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
