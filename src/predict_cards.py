def rows_from_result(result, crop_filename):
    rows = []
    for page in result.pages:
        for block_index, block in enumerate(page.blocks):
            for line_index, line in enumerate(block.lines):
                for word_index, word in enumerate(line.words):
                    (x1, y1), (x2, y2) = word.geometry
                    rows.append({
                        "crop_filename": crop_filename,
                        "block_index": block_index,
                        "line_index": line_index,
                        "word_index": word_index,
                        "predicted_text": word.value,
                        "confidence": word.confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    })
    return rows
