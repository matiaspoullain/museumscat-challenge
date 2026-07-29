"""Classify whether a card crop's recognized text actually carries a real
verbatimDate/verbatimLocality value, or is confound content (collector,
determination, museum boilerplate) -- reuses src.classify_field's
keyword-based date/locality/confound classifier per word, applied across
all the words doctr recognized on one card. See
docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md.

Usage:
    conda run -n museumscat-challenge python -m src.classify_card_relevance
"""
import csv
import json
import os

from src.classify_field import classify_field

MATCHES_PATH = os.path.join("data", "processed", "card_crops", "card_text_matches.json")
PREDICTIONS_PATH = os.path.join("data", "processed", "card_crops", "predictions.csv")
OUTPUT_PATH = os.path.join("data", "processed", "card_crops", "predictions_with_relevance.csv")


def is_relevant_card(words):
    """words: list of recognized text snippets on one card. True if any of
    them classifies as date or locality (i.e. not confound) -- an empty
    list is never relevant."""
    return any(classify_field(word) != "confound" for word in words)


def evaluate(predictions_by_card, matches):
    """predictions_by_card: crop_filename -> list of recognized words.
    matches: card_text_matches.json dict. Returns (precision, recall,
    accuracy) of is_relevant_card against ground truth over every card
    whose status is 'matched' or 'no_relevant_text' (labeled ground truth;
    'skipped' cards are excluded since they're bad detections, not a
    relevance judgment)."""
    tp = fp = tn = fn = 0
    for crop_filename, entry in matches.items():
        if entry["status"] not in ("matched", "no_relevant_text"):
            continue
        truth = entry["status"] == "matched"
        predicted = is_relevant_card(predictions_by_card.get(crop_filename, []))
        if predicted and truth:
            tp += 1
        elif predicted and not truth:
            fp += 1
        elif not predicted and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return precision, recall, accuracy


def _load_predictions_by_card():
    """Group predictions.csv's per-word rows into crop_filename -> list of
    recognized word strings."""
    predictions_by_card = {}
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            predictions_by_card.setdefault(row["crop_filename"], []).append(row["predicted_text"])
    return predictions_by_card


def apply_to_all():
    """Add a 'relevant' column to every row of predictions.csv, writing
    OUTPUT_PATH. Returns the number of rows written."""
    predictions_by_card = _load_predictions_by_card()
    with open(PREDICTIONS_PATH, encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = reader.fieldnames + ["relevant"]

    for row in rows:
        row["relevant"] = is_relevant_card(predictions_by_card.get(row["crop_filename"], []))

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    with open(MATCHES_PATH, encoding="utf-8") as f:
        matches = json.load(f)
    predictions_by_card = _load_predictions_by_card()
    precision, recall, accuracy = evaluate(predictions_by_card, matches)
    print(f"precision={precision:.3f} recall={recall:.3f} accuracy={accuracy:.3f}")

    n = apply_to_all()
    print(f"wrote {n} rows with relevance to {OUTPUT_PATH}")
