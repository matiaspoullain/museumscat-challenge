from src.box_card_words_app import build_word_queue, next_unboxed


def test_build_word_queue_splits_selections_into_words_in_order():
    matches = {
        "a_card00.jpeg": {
            "status": "matched",
            "selections": [
                {"field": "date", "index": 0, "text": "6/1862"},
                {"field": "locality", "index": 1, "text": "Kulhuset Jaegerspris"},
            ],
        },
        "a_card01.jpeg": {"status": "no_relevant_text", "selections": []},
    }
    queue = build_word_queue(matches)
    assert queue == [
        {"crop_filename": "a_card00.jpeg", "field": "date", "index": 0, "word_index": 0, "text": "6/1862"},
        {"crop_filename": "a_card00.jpeg", "field": "locality", "index": 1, "word_index": 0, "text": "Kulhuset"},
        {"crop_filename": "a_card00.jpeg", "field": "locality", "index": 1, "word_index": 1, "text": "Jaegerspris"},
    ]


def test_build_word_queue_excludes_skipped_and_no_relevant_text():
    matches = {
        "b_card00.jpeg": {"status": "skipped", "selections": []},
        "c_card00.jpeg": {"status": "no_relevant_text", "selections": []},
    }
    assert build_word_queue(matches) == []


def test_next_unboxed_skips_already_handled_words():
    queue = [
        {"crop_filename": "a.jpeg", "field": "date", "index": 0, "word_index": 0, "text": "6/1862"},
        {"crop_filename": "a.jpeg", "field": "locality", "index": 0, "word_index": 0, "text": "Kb"},
    ]
    word_boxes = {
        "a.jpeg": [{"field": "date", "index": 0, "word_index": 0, "text": "6/1862", "boxes": [[1, 2, 3, 4]]}],
    }
    assert next_unboxed(queue, word_boxes) == queue[1]


def test_next_unboxed_treats_explicit_skip_as_handled():
    queue = [{"crop_filename": "a.jpeg", "field": "date", "index": 0, "word_index": 0, "text": "6/1862"}]
    word_boxes = {"a.jpeg": [{"field": "date", "index": 0, "word_index": 0, "text": "6/1862", "boxes": []}]}
    assert next_unboxed(queue, word_boxes) is None


def test_next_unboxed_none_when_all_handled():
    assert next_unboxed([], {}) is None
