import pandas as pd

from src.analyze_confidence_errors import high_confidence_errors


def test_high_confidence_errors_filters_and_sorts_by_confidence_descending():
    merged = {
        "a.jpeg": {"verbatimDate": ("1.7.2000", 0.95), "verbatimLocality": ("Kb", 0.95)},   # correct, excluded
        "b.jpeg": {"verbatimDate": ("2.7.2000", 0.99), "verbatimLocality": ("Kb", 0.99)},   # date wrong+confident
        "c.jpeg": {"verbatimDate": ("1.7.2000", 0.92), "verbatimLocality": ("Ti", 0.92)},   # locality wrong+confident
        "d.jpeg": {"verbatimDate": ("9.9.9999", 0.5), "verbatimLocality": ("Rude Skov", 0.5)},  # wrong but not confident
    }
    ground_truth_df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
        {"image_file": "b.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
        {"image_file": "c.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Rude Skov"},
        {"image_file": "d.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Rude Skov"},
    ])
    rows = high_confidence_errors(merged, ground_truth_df, confidence_threshold=0.9)
    assert [(r["image_file"], r["field"]) for r in rows] == [
        ("b.jpeg", "verbatimDate"),
        ("c.jpeg", "verbatimLocality"),
    ]
    assert rows[0]["confidence"] >= rows[1]["confidence"]
