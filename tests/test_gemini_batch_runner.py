import csv
import os
import tempfile

import httpx
import pytest
from google.genai import errors as genai_errors

from src.gemini_batch_runner import (
    MAX_SAFE_RPM,
    ModelRotation,
    _call_with_fallback,
    _validate_rpm,
    already_done,
    is_daily_quota_exhausted,
    is_quota_exhausted,
    wholeimage_pending,
)


def _client_error(code):
    return genai_errors.ClientError(code, {"error": {"message": "boom"}}, response=None)


def _quota_client_error(quota_id=None, message="boom"):
    """A 429 shaped like a real Gemini free-tier response: a structured
    google.rpc.QuotaFailure detail carrying the quotaId that distinguishes a
    per-day limit (e.g. "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    from a per-minute/RPM one ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")."""
    details = []
    if quota_id is not None:
        details.append({
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id}],
        })
    return genai_errors.ClientError(
        429, {"error": {"message": message, "details": details}}, response=None)


def test_model_rotation_starts_at_first_model():
    rotation = ModelRotation(["a", "b", "c"])
    assert rotation.current == "a"


def test_model_rotation_advance_moves_forward():
    rotation = ModelRotation(["a", "b", "c"])
    assert rotation.advance() is True
    assert rotation.current == "b"
    assert rotation.advance() is True
    assert rotation.current == "c"


def test_model_rotation_advance_returns_false_at_last_model():
    rotation = ModelRotation(["a", "b"])
    rotation.advance()
    assert rotation.current == "b"
    assert rotation.advance() is False
    assert rotation.current == "b"


def test_model_rotation_rejects_empty_list():
    with pytest.raises(ValueError):
        ModelRotation([])


def test_is_quota_exhausted_true_for_429():
    assert is_quota_exhausted(_client_error(429)) is True


def test_is_quota_exhausted_false_for_other_codes():
    assert is_quota_exhausted(_client_error(404)) is False


def test_is_quota_exhausted_false_for_other_exception_types():
    assert is_quota_exhausted(ValueError("not an api error")) is False


def test_wholeimage_pending_includes_both_missing_specimens():
    merged_predictions = {
        "a.jpeg": {"verbatimDate": ("MISSING", 0.5), "verbatimLocality": ("MISSING", 0.5)},
    }
    assert wholeimage_pending(["a.jpeg"], merged_predictions) == ["a.jpeg"]


def test_wholeimage_pending_excludes_specimen_with_one_field_present():
    merged_predictions = {
        "a.jpeg": {"verbatimDate": ("22.5.1977", 0.9), "verbatimLocality": ("MISSING", 0.5)},
    }
    assert wholeimage_pending(["a.jpeg"], merged_predictions) == []


def test_wholeimage_pending_excludes_specimen_absent_from_merged_predictions():
    assert wholeimage_pending(["a.jpeg"], {}) == []


def test_already_done_reads_custom_key_column():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "wholeimage.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source_image_file", "verbatimDate"])
            writer.writeheader()
            writer.writerow({"source_image_file": "a.jpeg", "verbatimDate": "MISSING"})
        assert already_done(path, key_column="source_image_file") == {"a.jpeg"}


def test_already_done_missing_file_returns_empty_set():
    assert already_done("/nonexistent/path.csv", key_column="source_image_file") == set()


def test_is_daily_quota_exhausted_true_for_per_day_quota_id():
    error = _quota_client_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert is_daily_quota_exhausted(error) is True


def test_is_daily_quota_exhausted_false_for_per_minute_quota_id():
    error = _quota_client_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    assert is_daily_quota_exhausted(error) is False


def test_is_daily_quota_exhausted_false_for_non_429():
    assert is_daily_quota_exhausted(_client_error(404)) is False


def test_is_daily_quota_exhausted_false_for_other_exception_types():
    assert is_daily_quota_exhausted(ValueError("not an api error")) is False


def test_is_daily_quota_exhausted_falls_back_to_message_text_for_daily():
    error = _quota_client_error(quota_id=None, message="You exceeded your per day quota")
    assert is_daily_quota_exhausted(error) is True


def test_is_daily_quota_exhausted_falls_back_to_message_text_for_minute():
    error = _quota_client_error(quota_id=None, message="You exceeded your per minute quota")
    assert is_daily_quota_exhausted(error) is False


def test_is_daily_quota_exhausted_defaults_false_when_ambiguous():
    error = _quota_client_error(quota_id=None, message="quota exceeded")
    assert is_daily_quota_exhausted(error) is False


def test_validate_rpm_accepts_max_safe_rpm():
    _validate_rpm(MAX_SAFE_RPM)


def test_validate_rpm_rejects_above_max_safe_rpm():
    with pytest.raises(ValueError):
        _validate_rpm(MAX_SAFE_RPM + 1)


class _FlakyModels:
    """Fake google.genai client.models that raises a scripted sequence of
    errors/responses in order, recording which model each call used."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class _Response:
            text = outcome

        return _Response()


class _FlakyClient:
    def __init__(self, script):
        self.models = _FlakyModels(script)


_SUCCESS_TEXT = '{"verbatimDate": "MISSING", "verbatimLocality": "MISSING"}'


def test_call_with_fallback_retries_same_model_on_per_minute_429(monkeypatch):
    slept = []
    monkeypatch.setattr("src.gemini_batch_runner.time.sleep", lambda seconds: slept.append(seconds))

    per_minute_error = _quota_client_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
    client = _FlakyClient([per_minute_error, _SUCCESS_TEXT])
    rotation = ModelRotation(["a", "b"])

    result = _call_with_fallback(client, b"fake-bytes", rotation, rate_limiter=None)

    assert result["_model"] == "a"
    assert client.models.calls == ["a", "a"]
    assert rotation.current == "a"
    assert slept == [60]


def test_call_with_fallback_advances_model_on_daily_429():
    daily_error = _quota_client_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    client = _FlakyClient([daily_error, _SUCCESS_TEXT])
    rotation = ModelRotation(["a", "b"])

    result = _call_with_fallback(client, b"fake-bytes", rotation, rate_limiter=None)

    assert result["_model"] == "b"
    assert client.models.calls == ["a", "b"]
    assert rotation.current == "b"


def test_call_with_fallback_reraises_non_quota_errors():
    other_error = _client_error(404)
    client = _FlakyClient([other_error])
    rotation = ModelRotation(["a", "b"])

    with pytest.raises(genai_errors.ClientError):
        _call_with_fallback(client, b"fake-bytes", rotation, rate_limiter=None)


def test_call_with_fallback_retries_same_model_on_timeout():
    client = _FlakyClient([httpx.TimeoutException("stalled"), _SUCCESS_TEXT])
    rotation = ModelRotation(["a", "b"])

    result = _call_with_fallback(client, b"fake-bytes", rotation, rate_limiter=None)

    assert result["_model"] == "a"
    assert client.models.calls == ["a", "a"]
    assert rotation.current == "a"


def test_call_with_fallback_retries_same_model_on_connect_error():
    client = _FlakyClient([httpx.ConnectError("refused"), _SUCCESS_TEXT])
    rotation = ModelRotation(["a", "b"])

    result = _call_with_fallback(client, b"fake-bytes", rotation, rate_limiter=None)

    assert result["_model"] == "a"
    assert client.models.calls == ["a", "a"]
    assert rotation.current == "a"
