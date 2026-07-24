from src.merge_records import assemble_field


def test_empty_cards_is_missing():
    text, conf = assemble_field([])
    assert text == "MISSING"
    assert 0.0 <= conf <= 1.0


def test_single_card_passthrough():
    text, conf = assemble_field([("Kb", 0.9)])
    assert text == "Kb"
    assert conf == 0.9


def test_multiple_cards_joined_with_pipe_in_order():
    text, conf = assemble_field([("Ti", 0.8), ("Kulhuset Jaegerspris", 0.7)])
    assert text == "Ti | Kulhuset Jaegerspris"
    assert conf == 0.75
