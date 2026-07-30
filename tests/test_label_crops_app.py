from src.label_crops_app import is_valid_confidence, next_unlabeled


def test_returns_least_confident_candidate_not_yet_labeled_or_skipped():
    candidates = {
        "a.png": {"source": "img1.jpg", "doctr_text": "Foo", "doctr_confidence": 0.9},
        "b.png": {"source": "img1.jpg", "doctr_text": "Bar", "doctr_confidence": 0.2},
        "c.png": {"source": "img2.jpg", "doctr_text": "Baz", "doctr_confidence": 0.5},
    }
    assert next_unlabeled(candidates, labels={}, skipped=[]) == "b.png"


def test_next_unlabeled_skips_labeled_and_skipped_candidates():
    candidates = {
        "a.png": {"source": "img1.jpg", "doctr_text": "Foo", "doctr_confidence": 0.1},
        "b.png": {"source": "img1.jpg", "doctr_text": "Bar", "doctr_confidence": 0.9},
        "c.png": {"source": "img2.jpg", "doctr_text": "Baz", "doctr_confidence": 0.5},
    }
    assert next_unlabeled(candidates, labels={"a.png": "Foo"}, skipped=["c.png"]) == "b.png"


def test_returns_none_when_all_candidates_handled():
    candidates = {"a.png": {"source": "img1.jpg", "doctr_text": "Foo", "doctr_confidence": 0.5}}
    assert next_unlabeled(candidates, labels={"a.png": "Aabenraa"}, skipped=[]) is None


def test_confidence_accepts_numbers_in_range():
    assert is_valid_confidence(0) is True
    assert is_valid_confidence(1) is True
    assert is_valid_confidence(0.3) is True


def test_confidence_rejects_out_of_range_or_wrong_type():
    assert is_valid_confidence(1.1) is False
    assert is_valid_confidence(-0.1) is False
    assert is_valid_confidence("0.5") is False
    assert is_valid_confidence(True) is False
