"""Local web app for manually drawing one bounding box per word on every
'matched' card in data/processed/card_crops/card_text_matches.json, to
build a word-level training set for a Danish-vocab doctr recognizer
fine-tune. See
docs/superpowers/specs/2026-07-29-card-word-boxes-danish-finetune-design.md.

Usage:
    conda run -n museumscat-challenge python -m src.box_card_words_app
Then open http://localhost:8767 in a browser (WSL2 forwards localhost to
Windows automatically).
"""
import json
import os

MATCHES_PATH = os.path.join("data", "processed", "card_crops", "card_text_matches.json")
IMAGES_DIR = os.path.join("data", "processed", "card_crops", "images")
WORD_BOXES_PATH = os.path.join("data", "processed", "card_crops", "word_boxes.json")
PORT = 8767


def build_word_queue(matches):
    """matches: the card_text_matches.json dict (crop_filename -> {status,
    selections}). Returns an ordered list of {crop_filename, field, index,
    word_index, text} -- one entry per whitespace-split word of every
    selection on every 'matched' card, in matches' own iteration order
    (already grouped by specimen/card from the matching app), each card's
    selections in their saved reading order, each selection's words
    left-to-right."""
    queue = []
    for crop_filename, entry in matches.items():
        if entry["status"] != "matched":
            continue
        for selection in entry["selections"]:
            words = selection["text"].split()
            for word_index, word in enumerate(words):
                queue.append({
                    "crop_filename": crop_filename,
                    "field": selection["field"],
                    "index": selection["index"],
                    "word_index": word_index,
                    "text": word,
                })
    return queue


def _word_key(entry):
    """The (crop_filename, field, index, word_index) tuple identifying a
    queue entry, matched against word_boxes.json's own per-word records."""
    return (entry["crop_filename"], entry["field"], entry["index"], entry["word_index"])


def next_unboxed(queue, word_boxes):
    """First queue entry not yet present (boxed or explicitly skipped) in
    word_boxes -- word_boxes maps crop_filename -> list of {field, index,
    word_index, text, boxes} dicts (boxes == [] means explicitly skipped).
    Returns None if every entry has been handled."""
    done = set()
    for crop_filename, words in word_boxes.items():
        for w in words:
            done.add((crop_filename, w["field"], w["index"], w["word_index"]))
    for entry in queue:
        if _word_key(entry) not in done:
            return entry
    return None


def _load_json(path, default):
    """Read JSON from path, or return default if the file doesn't exist yet."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    """Write JSON to path atomically (write to a temp file, then rename)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    pass
