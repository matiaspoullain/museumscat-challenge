from src.predict_cards import rows_from_result


class FakeWord:
    def __init__(self, value, confidence, geometry):
        self.value = value
        self.confidence = confidence
        self.geometry = geometry


class FakeLine:
    def __init__(self, words):
        self.words = words


class FakeBlock:
    def __init__(self, lines):
        self.lines = lines


class FakePage:
    def __init__(self, blocks):
        self.blocks = blocks


class FakeResult:
    def __init__(self, pages):
        self.pages = pages


def test_rows_from_result_flattens_one_word():
    result = FakeResult([FakePage([FakeBlock([FakeLine([
        FakeWord("Kb", 0.9, ((0.1, 0.2), (0.3, 0.4))),
    ])])])])
    rows = rows_from_result(result, "a.jpeg")
    assert rows == [{
        "crop_filename": "a.jpeg", "block_index": 0, "line_index": 0, "word_index": 0,
        "predicted_text": "Kb", "confidence": 0.9,
        "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4,
    }]


def test_rows_from_result_preserves_word_order_within_a_line():
    result = FakeResult([FakePage([FakeBlock([FakeLine([
        FakeWord("Svinø", 0.9, ((0.0, 0.0), (0.1, 0.1))),
        FakeWord("strand", 0.8, ((0.1, 0.0), (0.2, 0.1))),
    ])])])])
    rows = rows_from_result(result, "a.jpeg")
    assert [r["predicted_text"] for r in rows] == ["Svinø", "strand"]
    assert [r["word_index"] for r in rows] == [0, 1]


def test_rows_from_result_indexes_multiple_blocks_and_lines():
    result = FakeResult([FakePage([
        FakeBlock([FakeLine([FakeWord("A", 0.5, ((0, 0), (1, 1)))])]),
        FakeBlock([
            FakeLine([FakeWord("B", 0.5, ((0, 0), (1, 1)))]),
            FakeLine([FakeWord("C", 0.5, ((0, 0), (1, 1)))]),
        ]),
    ])])
    rows = rows_from_result(result, "a.jpeg")
    assert [(r["block_index"], r["line_index"], r["predicted_text"]) for r in rows] == [
        (0, 0, "A"), (1, 0, "B"), (1, 1, "C"),
    ]
