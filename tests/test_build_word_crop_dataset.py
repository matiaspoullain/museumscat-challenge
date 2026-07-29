from src.build_word_crop_dataset import has_out_of_vocab_chars, union_box


def test_union_box_single_box_passthrough():
    assert union_box([(10, 20, 30, 40)]) == (10, 20, 30, 40)


def test_union_box_two_boxes():
    assert union_box([(10, 20, 30, 40), (5, 50, 60, 70)]) == (5, 20, 60, 70)


def test_union_box_three_boxes():
    assert union_box([(0, 0, 10, 10), (20, 20, 30, 30), (5, 5, 8, 8)]) == (0, 0, 30, 30)


def test_has_out_of_vocab_chars_true_when_char_missing():
    assert has_out_of_vocab_chars("Vendsÿssel", set("Vendsyssel")) is True


def test_has_out_of_vocab_chars_false_when_all_chars_present():
    assert has_out_of_vocab_chars("Kb", set("abcdefghijklmnopqrstuvwxyzKb")) is False
