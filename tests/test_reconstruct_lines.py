import pytest

from src.reconstruct_lines import group_into_lines


def _row(crop_filename, block_index, line_index, word_index, text, confidence):
    return {
        "crop_filename": crop_filename, "block_index": block_index, "line_index": line_index,
        "word_index": word_index, "predicted_text": text, "confidence": confidence,
    }


def test_orders_words_within_a_line_and_averages_confidence():
    rows = [
        _row("a.jpeg", 0, 0, 1, "strand", 0.8),
        _row("a.jpeg", 0, 0, 0, "Svinø", 0.9),
    ]
    result = group_into_lines(rows)
    assert result["a.jpeg"][0]["text"] == "Svinø strand"
    assert result["a.jpeg"][0]["confidence"] == pytest.approx(0.85)


def test_orders_lines_by_block_then_line_index():
    rows = [
        _row("a.jpeg", 1, 0, 0, "second", 0.5),
        _row("a.jpeg", 0, 0, 0, "first", 0.5),
    ]
    result = group_into_lines(rows)
    assert [line["text"] for line in result["a.jpeg"]] == ["first", "second"]


def test_skips_lines_with_no_nonempty_words():
    rows = [_row("a.jpeg", 0, 0, 0, "", 0.1)]
    assert group_into_lines(rows) == {}


def test_separates_different_crops():
    rows = [_row("a.jpeg", 0, 0, 0, "Kb", 0.9), _row("b.jpeg", 0, 0, 0, "Ti", 0.7)]
    result = group_into_lines(rows)
    assert set(result.keys()) == {"a.jpeg", "b.jpeg"}
