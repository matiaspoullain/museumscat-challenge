from src.build_word_crop_dataset import union_box


def test_union_box_single_box_passthrough():
    assert union_box([(10, 20, 30, 40)]) == (10, 20, 30, 40)


def test_union_box_two_boxes():
    assert union_box([(10, 20, 30, 40), (5, 50, 60, 70)]) == (5, 20, 60, 70)


def test_union_box_three_boxes():
    assert union_box([(0, 0, 10, 10), (20, 20, 30, 30), (5, 5, 8, 8)]) == (0, 0, 30, 30)
