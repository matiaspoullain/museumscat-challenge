"""Classical end-to-end pipeline: OCR words -> lines -> fields -> specimen.

Groups recognized words into lines (src.reconstruct_lines), classifies each line
as date/locality (src.classify_field), and merges a specimen's cards
(src.merge_records). Still on the active path: the Gemini route reuses its
manifest/prediction loaders and `process_specimen`.
"""

import csv
from collections.abc import Callable

from src.classify_field import classify_field
from src.merge_records import CardValue, assemble_field
from src.reconstruct_lines import Line, WordRow, group_into_lines

FIELD_TO_COLUMN = {"date": "verbatimDate", "locality": "verbatimLocality"}


def classify_card_field_values(lines: list[Line]) -> dict[str, CardValue]:
    grouped: dict[str, list[Line]] = {"date": [], "locality": []}
    for line in lines:
        field = classify_field(line["text"])
        if field in grouped:
            grouped[field].append(line)

    values: dict[str, CardValue] = {}
    for field, field_lines in grouped.items():
        if not field_lines:
            continue
        text = " ".join(line["text"] for line in field_lines)
        confidence = sum(line["confidence"] for line in field_lines) / len(field_lines)
        values[field] = (text, confidence)
    return values


def process_specimen(crop_filenames: list[str], lines_by_crop: dict[str, list[Line]],
                     is_relevant: Callable[[str], bool] | None = None) -> dict[str, CardValue]:
    """is_relevant: optional crop_filename -> bool predicate. When given, cards
    it flags as not relevant (e.g. confound/collector labels) are dropped
    before their lines are classified, instead of relying solely on
    classify_field to filter out confound text line-by-line."""
    if is_relevant is not None:
        crop_filenames = [c for c in crop_filenames if is_relevant(c)]
    per_field_cards: dict[str, list[CardValue]] = {"date": [], "locality": []}
    for crop_filename in crop_filenames:
        card_values = classify_card_field_values(lines_by_crop.get(crop_filename, []))
        for field, value in card_values.items():
            per_field_cards[field].append(value)
    return {
        FIELD_TO_COLUMN[field]: assemble_field(cards)
        for field, cards in per_field_cards.items()
    }


def run_pipeline(image_files: list[str], manifest_by_image: dict[str, list[str]],
                 lines_by_crop: dict[str, list[Line]],
                 is_relevant: Callable[[str], bool] | None = None) -> dict[str, dict[str, CardValue]]:
    return {
        image_file: process_specimen(manifest_by_image.get(image_file, []), lines_by_crop, is_relevant)
        for image_file in image_files
    }


def load_manifest(path: str) -> dict[str, list[str]]:
    by_image: dict[str, list[tuple[int, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_image.setdefault(row["source_image_file"], []).append(
                (int(row["card_index"]), row["crop_filename"])
            )
    return {
        image_file: [crop_filename for _, crop_filename in sorted(entries)]
        for image_file, entries in by_image.items()
    }


def load_predictions(path: str) -> dict[str, list[Line]]:
    rows: list[WordRow] = []
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
