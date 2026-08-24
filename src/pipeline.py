"""Pipeline clasico de extremo a extremo: palabras OCR -> lineas -> campos -> especimen.

Agrupa las palabras reconocidas en lineas (`src.reconstruct_lines`), clasifica
cada linea en date/locality (`src.classify_field`) y fusiona las tarjetas de un
especimen (`src.merge_records`). Sigue en la ruta activa porque los cargadores de
manifiesto/predicciones y `process_specimen` los reutiliza la ruta Gemini.
"""

import csv

from src.classify_field import classify_field
from src.merge_records import assemble_field
from src.reconstruct_lines import group_into_lines

FIELD_TO_COLUMN = {"date": "verbatimDate", "locality": "verbatimLocality"}


def classify_card_field_values(lines):
    grouped = {"date": [], "locality": []}
    for line in lines:
        field = classify_field(line["text"])
        if field in grouped:
            grouped[field].append(line)

    values = {}
    for field, field_lines in grouped.items():
        if not field_lines:
            continue
        text = " ".join(line["text"] for line in field_lines)
        confidence = sum(line["confidence"] for line in field_lines) / len(field_lines)
        values[field] = (text, confidence)
    return values


def process_specimen(crop_filenames, lines_by_crop, is_relevant=None):
    """is_relevant: optional crop_filename -> bool predicate. When given, cards
    it flags as not relevant (e.g. confound/collector labels) are dropped
    before their lines are classified, instead of relying solely on
    classify_field to filter out confound text line-by-line."""
    if is_relevant is not None:
        crop_filenames = [c for c in crop_filenames if is_relevant(c)]
    per_field_cards = {"date": [], "locality": []}
    for crop_filename in crop_filenames:
        card_values = classify_card_field_values(lines_by_crop.get(crop_filename, []))
        for field, value in card_values.items():
            per_field_cards[field].append(value)
    return {
        FIELD_TO_COLUMN[field]: assemble_field(cards)
        for field, cards in per_field_cards.items()
    }


def run_pipeline(image_files, manifest_by_image, lines_by_crop, is_relevant=None):
    return {
        image_file: process_specimen(manifest_by_image.get(image_file, []), lines_by_crop, is_relevant)
        for image_file in image_files
    }


def load_manifest(path):
    by_image = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_image.setdefault(row["source_image_file"], []).append(
                (int(row["card_index"]), row["crop_filename"])
            )
    return {
        image_file: [crop_filename for _, crop_filename in sorted(entries)]
        for image_file, entries in by_image.items()
    }


def load_predictions(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "crop_filename": row["crop_filename"],
                "block_index": int(row["block_index"]),
                "line_index": int(row["line_index"]),
                "word_index": int(row["word_index"]),
                "predicted_text": row["predicted_text"],
                "confidence": float(row["confidence"]),
            })
    return group_into_lines(rows)
