"""Regroup recognized words back into text lines, per card.

Orders by word index within each (crop, block, line) and concatenates; the step
between src.old.predict_cards and src.classify_field inside src.pipeline.
"""

from typing import TypedDict


class WordRow(TypedDict):
    """One recognized word, as loaded by src.pipeline.load_predictions."""

    crop_filename: str
    block_index: int
    line_index: int
    word_index: int
    predicted_text: str
    confidence: float


class Line(TypedDict):
    """One reconstructed text line of a card."""

    text: str
    confidence: float


def group_into_lines(word_rows: list[WordRow]) -> dict[str, list[Line]]:
    by_crop_line: dict[tuple[str, int, int], list[WordRow]] = {}
    for row in word_rows:
        key = (row["crop_filename"], row["block_index"], row["line_index"])
        by_crop_line.setdefault(key, []).append(row)

    lines_by_crop: dict[str, list[Line]] = {}
    for (crop_filename, _block_index, _line_index), rows in sorted(by_crop_line.items()):
        rows = sorted(rows, key=lambda r: r["word_index"])
        words = [r["predicted_text"] for r in rows if r["predicted_text"]]
        if not words:
            continue
        text = " ".join(words)
        confidence = sum(r["confidence"] for r in rows) / len(rows)
        lines_by_crop.setdefault(crop_filename, []).append({"text": text, "confidence": confidence})
    return lines_by_crop
