"""Local web app for manually matching each detected card crop under
data/processed/card_crops/ against the piece(s) of data/raw/train.csv
ground-truth text actually written on that card, in reading order. Produces
training pairs for a future doctr fine-tune. See
docs/superpowers/specs/2026-07-28-card-label-matching-design.md.

Usage:
    conda run -n museumscat-challenge python -m src.match_card_labels_app
Then open http://localhost:8766 in a browser (WSL2 forwards localhost to
Windows automatically).
"""
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
TRAIN_CSV = os.path.join("data", "raw", "train.csv")
IMAGES_DIR = os.path.join("data", "processed", "card_crops", "images")
MATCHES_PATH = os.path.join("data", "processed", "card_crops", "card_text_matches.json")
PORT = 8766

QUEUE = []


def build_candidates(row):
    """row has verbatimDate/verbatimLocality string values (as read from
    train.csv). Returns an ordered list of {field, index, text} dicts, one
    per non-"MISSING" pipe-separated piece -- all date pieces first, then
    all locality pieces, each piece stripped of surrounding whitespace."""
    candidates = []
    for field, column in (("date", "verbatimDate"), ("locality", "verbatimLocality")):
        pieces = [p.strip() for p in row[column].split("|")]
        for index, text in enumerate(pieces):
            if text == "MISSING":
                continue
            candidates.append({"field": field, "index": index, "text": text})
    return candidates
