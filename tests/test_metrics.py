from src.metrics import CORRECT_THRESHOLD, aurc, best_pipe_ned, edit_distance, ned, normalize_date_punctuation


def test_edit_distance_identical():
    assert edit_distance("kb", "kb") == 0


def test_edit_distance_substitution():
    assert edit_distance("kb", "kg") == 1


def test_edit_distance_insertion():
    assert edit_distance("kb", "kbx") == 1


def test_normalize_date_punctuation_variants_match():
    variants = ["27.5.2022", "27,5,2022", "27-5-2022", "27·5·2022", "27 5 2022"]
    normalized = {normalize_date_punctuation(v) for v in variants}
    assert len(normalized) == 1


def test_ned_exact_match_is_zero():
    assert ned("Kb", "kb") == 0.0


def test_ned_completely_wrong_is_one():
    assert ned("abc", "xyz") == 1.0


def test_ned_date_punctuation_normalized():
    assert ned("27-5-2022", "27.5.2022", is_date=True) == 0.0


def test_ned_locality_punctuation_not_normalized():
    assert ned("27-5-2022", "27.5.2022", is_date=False) > 0.0


def test_best_pipe_ned_single_card():
    assert best_pipe_ned("Kb", "Kb") == 0.0


def test_best_pipe_ned_tries_orderings():
    assert best_pipe_ned("Kulhuset | Ti", "Ti | Kulhuset") == 0.0


def test_best_pipe_ned_penalizes_wrong_content_even_if_reordered():
    assert best_pipe_ned("Wrong | Ti", "Ti | Kulhuset") > 0.0


def test_aurc_perfect_calibration_beats_inverted():
    errors = [0.0, 0.2, 0.5, 1.0]
    good_confidences = [1.0, 0.8, 0.5, 0.0]
    bad_confidences = [0.0, 0.2, 0.5, 1.0]
    assert aurc(errors, good_confidences) < aurc(errors, bad_confidences)


def test_aurc_all_zero_error_is_zero():
    assert aurc([0.0, 0.0, 0.0], [0.9, 0.1, 0.5]) == 0.0


def test_correct_threshold_is_a_small_tolerance():
    assert 0.0 < CORRECT_THRESHOLD < 0.2
