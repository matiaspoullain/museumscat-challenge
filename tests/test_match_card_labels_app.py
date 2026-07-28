from src.match_card_labels_app import build_candidates


def test_single_value_each_field():
    row = {"verbatimDate": "22.5.1977", "verbatimLocality": "Svinø strand"}
    assert build_candidates(row) == [
        {"field": "date", "index": 0, "text": "22.5.1977"},
        {"field": "locality", "index": 0, "text": "Svinø strand"},
    ]


def test_both_fields_missing_returns_empty():
    row = {"verbatimDate": "MISSING", "verbatimLocality": "MISSING"}
    assert build_candidates(row) == []


def test_one_field_missing_only_other_included():
    row = {"verbatimDate": "MISSING", "verbatimLocality": "Ti"}
    assert build_candidates(row) == [{"field": "locality", "index": 0, "text": "Ti"}]


def test_multi_pipe_field_indexed_in_order():
    row = {"verbatimDate": "6/1862", "verbatimLocality": "Ti | Kulhuset Jægerspris"}
    assert build_candidates(row) == [
        {"field": "date", "index": 0, "text": "6/1862"},
        {"field": "locality", "index": 0, "text": "Ti"},
        {"field": "locality", "index": 1, "text": "Kulhuset Jægerspris"},
    ]


def test_pieces_are_stripped_of_whitespace():
    row = {"verbatimDate": "MISSING", "verbatimLocality": " Al | Nord for Storskoven Vendsyssel"}
    assert build_candidates(row) == [
        {"field": "locality", "index": 0, "text": "Al"},
        {"field": "locality", "index": 1, "text": "Nord for Storskoven Vendsyssel"},
    ]


def test_missing_piece_within_a_multi_pipe_field_is_skipped():
    row = {"verbatimDate": "MISSING", "verbatimLocality": "Ti | MISSING | Rørvig"}
    assert build_candidates(row) == [
        {"field": "locality", "index": 0, "text": "Ti"},
        {"field": "locality", "index": 2, "text": "Rørvig"},
    ]
