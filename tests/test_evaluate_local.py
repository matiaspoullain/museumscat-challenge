import pandas as pd

from src.old.evaluate_local import score_predictions


def test_score_predictions_perfect_match_is_zero_ned():
    predicted = {"a.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("Kb", 0.8)}}
    ground_truth = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
    ])
    report = score_predictions(predicted, ground_truth)
    assert report["verbatimDate"]["mean_ned"] == 0.0
    assert report["verbatimLocality"]["mean_ned"] == 0.0


def test_score_predictions_date_normalizes_punctuation_locality_does_not():
    predicted = {"a.jpeg": {"verbatimDate": ("1-7-2000", 0.9), "verbatimLocality": ("kb ", 0.8)}}
    ground_truth = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
    ])
    report = score_predictions(predicted, ground_truth)
    assert report["verbatimDate"]["mean_ned"] == 0.0
    assert report["verbatimLocality"]["mean_ned"] == 0.0  # case/whitespace-insensitive per src.metrics.ned


def test_score_predictions_returns_aurc_key_per_field():
    predicted = {"a.jpeg": {"verbatimDate": ("MISSING", 0.5), "verbatimLocality": ("MISSING", 0.5)}}
    ground_truth = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "MISSING", "verbatimLocality": "MISSING"},
    ])
    report = score_predictions(predicted, ground_truth)
    assert "aurc" in report["verbatimDate"]
    assert "aurc" in report["verbatimLocality"]
