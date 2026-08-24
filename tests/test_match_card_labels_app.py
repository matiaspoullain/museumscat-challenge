from src.old.match_card_labels_app import build_candidates, build_queue, next_unmatched, unused_candidates


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


def test_groups_cards_by_specimen_in_card_index_order():
    manifest_rows = [
        {"crop_filename": "a_card01.jpeg", "source_image_file": "a.jpeg", "card_index": "1"},
        {"crop_filename": "a_card00.jpeg", "source_image_file": "a.jpeg", "card_index": "0"},
    ]
    train_rows = [{"image_file": "a.jpeg", "verbatimDate": "1901", "verbatimLocality": "Odense"}]
    queue = build_queue(manifest_rows, train_rows)
    assert [q["crop_filename"] for q in queue] == ["a_card00.jpeg", "a_card01.jpeg"]
    assert queue[0]["candidates"] == [
        {"field": "date", "index": 0, "text": "1901"},
        {"field": "locality", "index": 0, "text": "Odense"},
    ]


def test_excludes_specimens_with_no_candidates():
    manifest_rows = [{"crop_filename": "b_card00.jpeg", "source_image_file": "b.jpeg", "card_index": "0"}]
    train_rows = [{"image_file": "b.jpeg", "verbatimDate": "MISSING", "verbatimLocality": "MISSING"}]
    assert build_queue(manifest_rows, train_rows) == []


def test_specimen_with_candidates_but_no_manifest_rows_contributes_nothing():
    train_rows = [{"image_file": "c.jpeg", "verbatimDate": "1901", "verbatimLocality": "Odense"}]
    assert build_queue([], train_rows) == []


def test_preserves_train_csv_specimen_order():
    manifest_rows = [
        {"crop_filename": "y_card00.jpeg", "source_image_file": "y.jpeg", "card_index": "0"},
        {"crop_filename": "x_card00.jpeg", "source_image_file": "x.jpeg", "card_index": "0"},
    ]
    train_rows = [
        {"image_file": "x.jpeg", "verbatimDate": "1901", "verbatimLocality": "Odense"},
        {"image_file": "y.jpeg", "verbatimDate": "1902", "verbatimLocality": "Aarhus"},
    ]
    queue = build_queue(manifest_rows, train_rows)
    assert [q["source_image_file"] for q in queue] == ["x.jpeg", "y.jpeg"]


def test_next_unmatched_returns_first_not_in_matches():
    queue = [{"crop_filename": "a.jpeg"}, {"crop_filename": "b.jpeg"}]
    assert next_unmatched(queue, matches={"a.jpeg": {}}) == {"crop_filename": "b.jpeg"}


def test_next_unmatched_returns_none_when_all_done():
    queue = [{"crop_filename": "a.jpeg"}]
    assert next_unmatched(queue, matches={"a.jpeg": {}}) is None


def test_unused_candidates_finds_candidates_in_no_cards_selections():
    candidates = [
        {"field": "locality", "index": 0, "text": "Ti"},
        {"field": "locality", "index": 1, "text": "Rørvig"},
    ]
    selections_by_card = [[{"field": "locality", "index": 0, "text": "Ti"}]]
    assert unused_candidates(candidates, selections_by_card) == [
        {"field": "locality", "index": 1, "text": "Rørvig"}
    ]


def test_unused_candidates_empty_when_all_used_across_cards():
    candidates = [
        {"field": "locality", "index": 0, "text": "Ti"},
        {"field": "locality", "index": 1, "text": "Rørvig"},
    ]
    selections_by_card = [
        [{"field": "locality", "index": 0, "text": "Ti"}],
        [{"field": "locality", "index": 1, "text": "Rørvig"}],
    ]
    assert unused_candidates(candidates, selections_by_card) == []
