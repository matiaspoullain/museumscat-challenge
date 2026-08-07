import pytest
from google.genai import errors as genai_errors

from src.gemini_batch_runner import ModelRotation, is_quota_exhausted


def _client_error(code):
    return genai_errors.ClientError(code, {"error": {"message": "boom"}}, response=None)


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
