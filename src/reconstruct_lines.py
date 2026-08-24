"""Reagrupa las palabras reconocidas en lineas de texto por tarjeta.

Ordena por indice de palabra dentro de cada (crop, bloque, linea) y concatena;
es el paso entre `src.old.predict_cards` y `src.classify_field` dentro de
`src.pipeline`.
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
