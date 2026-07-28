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


def build_queue(manifest_rows, train_rows):
    """manifest_rows: dicts with crop_filename/source_image_file/card_index
    (as read from manifest.csv -- card_index is a string). train_rows: dicts
    with image_file/verbatimDate/verbatimLocality (as read from train.csv).
    Returns an ordered list of {crop_filename, source_image_file, card_index,
    candidates} -- one entry per card crop, grouped by specimen in train.csv's
    own row order (then by card_index ascending within a specimen), skipping
    any specimen whose candidate list is empty (both fields "MISSING")."""
    by_source = {}
    for row in manifest_rows:
        by_source.setdefault(row["source_image_file"], []).append(row)

    queue = []
    for train_row in train_rows:
        image_file = train_row["image_file"]
        candidates = build_candidates(train_row)
        if not candidates:
            continue
        cards = sorted(by_source.get(image_file, []), key=lambda r: int(r["card_index"]))
        for card in cards:
            queue.append({
                "crop_filename": card["crop_filename"],
                "source_image_file": image_file,
                "card_index": int(card["card_index"]),
                "candidates": candidates,
            })
    return queue


def next_unmatched(queue, matches):
    """First queue entry whose crop_filename isn't a key in matches, or None
    if every entry has already been matched."""
    for entry in queue:
        if entry["crop_filename"] not in matches:
            return entry
    return None


def unused_candidates(candidates, selections_by_card):
    """candidates: one specimen's full candidate list. selections_by_card: a
    list of that specimen's cards' already-saved selections (each itself a
    list of {field, index, text} dicts). Returns the candidates that don't
    appear in any card's selections, identified by (field, index)."""
    used = {(sel["field"], sel["index"]) for selections in selections_by_card for sel in selections}
    return [c for c in candidates if (c["field"], c["index"]) not in used]
