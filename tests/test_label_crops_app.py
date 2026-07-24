from src.label_crops_app import next_unlabeled


def test_returns_first_candidate_not_yet_labeled_or_skipped():
    candidates = {"a.png": "img1.jpg", "b.png": "img1.jpg", "c.png": "img2.jpg"}
    assert next_unlabeled(candidates, labels={"a.png": "Aabenraa"}, skipped=["b.png"]) == "c.png"


def test_returns_none_when_all_candidates_handled():
    candidates = {"a.png": "img1.jpg"}
    assert next_unlabeled(candidates, labels={"a.png": "Aabenraa"}, skipped=[]) is None
