"""Regroup recognized words back into text lines, per card.

Orders by word index within each (crop, block, line) and concatenates; the step
between src.old.predict_cards and src.classify_field inside src.pipeline.
"""

def group_into_lines(word_rows):
    by_crop_line = {}
    for row in word_rows:
        key = (row["crop_filename"], row["block_index"], row["line_index"])
        by_crop_line.setdefault(key, []).append(row)

    lines_by_crop = {}
    for (crop_filename, _block_index, _line_index), rows in sorted(by_crop_line.items()):
        rows = sorted(rows, key=lambda r: r["word_index"])
        words = [r["predicted_text"] for r in rows if r["predicted_text"]]
        if not words:
            continue
        text = " ".join(words)
        confidence = sum(r["confidence"] for r in rows) / len(rows)
        lines_by_crop.setdefault(crop_filename, []).append({"text": text, "confidence": confidence})
    return lines_by_crop
