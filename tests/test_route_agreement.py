import pytest

from src.route_agreement import route_agreement_score, wholeimage_primary_confidence


def test_identical_texts_score_1():
    assert route_agreement_score("Lodskovvad", "Lodskovvad", is_date=False) == 1.0


def test_case_and_whitespace_insensitive():
    assert route_agreement_score("  lodskovvad ", "Lodskovvad", is_date=False) == 1.0


def test_near_miss_scores_between_0_and_1():
    # One character apart out of ten -- partial agreement, not zero.
    score = route_agreement_score("Lodskovvad", "Lodskovvod", is_date=False)
    assert 0.5 < score < 1.0


def test_both_routes_missing_counts_as_agreement():
    # Two independent routes each finding nothing is evidence, not absence of it:
    # the structural-missing tier finding is that such rows are unusually accurate.
    assert route_agreement_score("MISSING", "MISSING", is_date=False) == 1.0


def test_one_sided_missing_is_maximal_disagreement():
    assert route_agreement_score("Lodskovvad", "MISSING", is_date=False) == 0.0
    assert route_agreement_score("MISSING", "Lodskovvad", is_date=False) == 0.0


def test_date_punctuation_is_free():
    # src.metrics.normalize_date_punctuation collapses [.,-.\s] to "." at scoring
    # time, so those must not count as disagreement here either. "/" is NOT in that
    # set -- the scorer does treat it as a real edit, so agreement must too.
    assert route_agreement_score("22.5.1977", "22-5-1977", is_date=True) == 1.0
    assert route_agreement_score("22.5.1977", "22 5 1977", is_date=True) == 1.0
    assert route_agreement_score("22.5.1977", "22-5-1977", is_date=False) < 1.0
    assert route_agreement_score("22.5.1977", "22/5/1977", is_date=True) < 1.0


def test_confidence_is_agreement_alone_for_dates():
    # verbatimDate has no corpus component -- corpus support is locality-only.
    confidence = wholeimage_primary_confidence(
        "verbatimDate", "22.5.1977", "22.5.1977", locality_corpus=None, local_vlm_agreement=None)
    assert confidence == pytest.approx(1.0)


def test_confidence_stays_within_unit_range():
    # combine_tier_and_within_tier_score caps the top tier so a within-tier signal
    # cannot push a confidence above 1.0 -- validate_submission rejects that.
    for corpus in (None, {"lodskovvad": 40}):
        confidence = wholeimage_primary_confidence(
            "verbatimLocality", "Lodskovvad", "Lodskovvad", corpus, local_vlm_agreement=1.0)
        assert 0.0 <= confidence <= 1.0


def test_agreement_dominates_the_within_tier_signals():
    # The tie structure must survive: a specimen the two routes agree on outranks
    # one they disagree on, no matter how the corpus/VLM signals fall.
    agreeing = wholeimage_primary_confidence(
        "verbatimLocality", "Lodskovvad", "Lodskovvad", {}, local_vlm_agreement=0.0)
    disagreeing = wholeimage_primary_confidence(
        "verbatimLocality", "Lodskovvad", "Ravneholm", {"lodskovvad": 99}, local_vlm_agreement=1.0)
    assert agreeing > disagreeing


def test_corpus_support_breaks_ties_between_equally_agreeing_rows():
    supported = wholeimage_primary_confidence(
        "verbatimLocality", "Lodskovvad", "Lodskovvad", {"lodskovvad": 40}, None)
    unsupported = wholeimage_primary_confidence(
        "verbatimLocality", "Zzzzzzzzzz", "Zzzzzzzzzz", {"lodskovvad": 40}, None)
    assert supported > unsupported


def test_multi_card_order_is_not_disagreement():
    # Two independent routes have no reason to enumerate a specimen's cards in the
    # same order, and src.metrics.best_pipe_ned doesn't make the scorer care either.
    assert route_agreement_score("Grenaa | Brastrup", "Brastrup | Grenaa", is_date=False) == 1.0


def test_multi_card_still_penalizes_a_real_difference():
    score = route_agreement_score("Grenaa | Brastrup", "Grenaa | Ravneholm", is_date=False)
    assert 0.0 < score < 1.0
