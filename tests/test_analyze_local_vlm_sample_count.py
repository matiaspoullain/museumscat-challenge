import csv
import json
import os

from src.old.analyze_local_vlm_sample_count import (
    N_SAMPLE_COUNTS,
    agreement_gradient_for_n,
    truncated_local_vlm_predictions_by_crop,
)


def _write_raw_results_csv(path, rows):
    fieldnames = ["crop_filename", "source_image_file", "local_vlm_verbatimDate",
                  "local_vlm_verbatimDate_confidence", "local_vlm_verbatimLocality",
                  "local_vlm_verbatimLocality_confidence", "raw_samples_json"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_truncated_local_vlm_predictions_by_crop_uses_only_first_n_samples(tmp_path):
    samples = [
        {"verbatimDate": "22.5.1977", "verbatimLocality": "Kb"},
        {"verbatimDate": "22.5.1977", "verbatimLocality": "Kb"},
        {"verbatimDate": "1.1.1900", "verbatimLocality": "Ti"},  # only the first 2 matter for n=2
    ]
    path = tmp_path / "raw.csv"
    _write_raw_results_csv(path, [{
        "crop_filename": "a_card00.jpeg", "source_image_file": "a.jpeg",
        "local_vlm_verbatimDate": "unused", "local_vlm_verbatimDate_confidence": "0.0",
        "local_vlm_verbatimLocality": "unused", "local_vlm_verbatimLocality_confidence": "0.0",
        "raw_samples_json": json.dumps(samples),
    }])

    by_crop = truncated_local_vlm_predictions_by_crop(str(path), n_samples=2)

    assert by_crop["a_card00.jpeg"]["verbatimDate"] == ("22.5.1977", 1.0)
    assert by_crop["a_card00.jpeg"]["verbatimLocality"] == ("Kb", 1.0)


def test_truncated_local_vlm_predictions_by_crop_n_equals_1_is_the_first_sample_verbatim(tmp_path):
    samples = [
        {"verbatimDate": "1.1.1900", "verbatimLocality": "Ti"},
        {"verbatimDate": "22.5.1977", "verbatimLocality": "Kb"},
    ]
    path = tmp_path / "raw.csv"
    _write_raw_results_csv(path, [{
        "crop_filename": "a_card00.jpeg", "source_image_file": "a.jpeg",
        "local_vlm_verbatimDate": "unused", "local_vlm_verbatimDate_confidence": "0.0",
        "local_vlm_verbatimLocality": "unused", "local_vlm_verbatimLocality_confidence": "0.0",
        "raw_samples_json": json.dumps(samples),
    }])

    by_crop = truncated_local_vlm_predictions_by_crop(str(path), n_samples=1)

    assert by_crop["a_card00.jpeg"]["verbatimDate"] == ("1.1.1900", 1.0)


def test_truncated_local_vlm_predictions_by_crop_missing_file_returns_empty(tmp_path):
    assert truncated_local_vlm_predictions_by_crop(str(tmp_path / "nope.csv"), n_samples=3) == {}


def test_n_sample_counts_includes_the_full_pilot_count():
    # The dev pilot collected 10 samples per crop -- any subsampling analysis
    # must include 10 itself as the "no change" control.
    assert 10 in N_SAMPLE_COUNTS
    assert N_SAMPLE_COUNTS == sorted(N_SAMPLE_COUNTS)


def test_agreement_gradient_for_n_splits_top_tier_by_agreement():
    # Two dev specimens tied at Gemini's top confidence (1.0): one where the
    # local VLM agrees with Gemini's text (low error), one where it disagrees
    # (high error) -- the gradient row must separate them correctly.
    image_files = ["a.jpeg", "b.jpeg"]
    gemini_confidences = [1.0, 1.0]
    gemini_texts = ["Kb", "Kb"]
    errors = [0.0, 0.8]
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"], "b.jpeg": ["b_card00.jpeg"]}
    local_vlm_predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 1.0), "verbatimLocality": ("Kb", 1.0)},
        "b_card00.jpeg": {"verbatimDate": ("22.5.1977", 1.0), "verbatimLocality": ("Ti", 1.0)},
    }

    row = agreement_gradient_for_n(
        field="verbatimLocality", image_files=image_files, gemini_confidences=gemini_confidences,
        gemini_texts=gemini_texts, errors=errors, manifest_by_image=manifest_by_image,
        local_vlm_predictions_by_crop=local_vlm_predictions_by_crop,
    )

    assert row["agree_n"] == 1
    assert row["agree_mean_ned"] == 0.0
    assert row["disagree_n"] == 1
    assert row["disagree_mean_ned"] == 0.8
