import csv
import os
import tempfile

import pytest
from google.genai import errors as genai_errors

from src.gemini_batch_runner import ModelRotation, already_done, is_quota_exhausted, wholeimage_pending


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
