import pytest

from src.confidence_rank_rules import (
    STRUCTURAL_MISSING_CONFIDENCE,
    TIER_EPSILON,
    apply_structural_missing_tier,
    combine_tier_and_within_tier_score,
    date_plausibility_score,
    is_confident_structural_missing,
    within_tier_score,
)


def test_is_confident_structural_missing_true_when_zero_cards_processed():
    assert is_confident_structural_missing("MISSING", n_processed=0, n_contributing=0)


def test_is_confident_structural_missing_true_when_one_card_processed_and_missed():
    assert is_confident_structural_missing("MISSING", n_processed=1, n_contributing=0)


def test_is_confident_structural_missing_false_when_two_plus_cards_processed():
    # This is exactly src.confidence_penalty_rules' own under-coverage risk
    # pattern (a second card's real value may have been missed) -- deliberately
    # NOT elevated here, left to the existing penalty instead.
    assert not is_confident_structural_missing("MISSING", n_processed=2, n_contributing=0)


def test_is_confident_structural_missing_false_when_text_present():
    assert not is_confident_structural_missing("Kb", n_processed=1, n_contributing=0)


def test_apply_structural_missing_tier_overrides_confidence_when_applicable():
    result = apply_structural_missing_tier(0.5, "MISSING", n_processed=0, n_contributing=0)
    assert result == STRUCTURAL_MISSING_CONFIDENCE


def test_apply_structural_missing_tier_leaves_confidence_when_not_applicable():
    result = apply_structural_missing_tier(0.9, "Kb", n_processed=1, n_contributing=1)
    assert result == 0.9


def test_apply_structural_missing_tier_leaves_flagged_missing_untouched():
    # 2 processed, 0 contributing -- already penalized 0.5 * 0.7 = 0.35 by
    # src.confidence_penalty_rules; this rule must not also touch it.
    result = apply_structural_missing_tier(0.35, "MISSING", n_processed=2, n_contributing=0)
    assert result == 0.35


def test_combine_tier_and_within_tier_score_stays_within_epsilon_band():
    lo = combine_tier_and_within_tier_score(0.7, within_tier_score=0.0)
    hi = combine_tier_and_within_tier_score(0.7, within_tier_score=1.0)
    assert lo == 0.7
    assert hi == 0.7 + TIER_EPSILON


def test_combine_tier_and_within_tier_score_never_crosses_into_next_tier():
    # The smallest real gap between adjacent production confidence tiers
    # (e.g. 0.667 vs 0.7) is ~0.033 -- TIER_EPSILON must stay well under that.
    smallest_observed_gap = 0.033
    assert TIER_EPSILON < smallest_observed_gap


def test_combine_tier_and_within_tier_score_monotonic_in_within_tier_score():
    a = combine_tier_and_within_tier_score(0.5, within_tier_score=0.2)
    b = combine_tier_and_within_tier_score(0.5, within_tier_score=0.8)
    assert a < b


def test_combine_tier_and_within_tier_score_clips_out_of_range_input():
    below = combine_tier_and_within_tier_score(0.5, within_tier_score=-1.0)
    above = combine_tier_and_within_tier_score(0.5, within_tier_score=2.0)
    assert below == 0.5
    assert above == 0.5 + TIER_EPSILON


def test_combine_tier_and_within_tier_score_never_exceeds_one_at_top_tier():
    # tier_confidence == 1.0 (raw self-consistency ceiling, ~45% of all rows)
    # has no tier above it to borrow headroom from -- must cap itself instead
    # of overflowing past 1.0, which src.generate_submission.
    # validate_submission would reject.
    result = combine_tier_and_within_tier_score(1.0, within_tier_score=1.0)
    assert result <= 1.0
    assert result == pytest.approx(1.0)


def test_combine_tier_and_within_tier_score_still_orders_top_tier_by_score():
    lo = combine_tier_and_within_tier_score(1.0, within_tier_score=0.0)
    hi = combine_tier_and_within_tier_score(1.0, within_tier_score=1.0)
    assert lo < hi <= 1.0


def test_date_plausibility_score_full_date_is_plausible():
    assert date_plausibility_score("22.5.1977") == 1.0


def test_date_plausibility_score_bare_catalog_number_is_implausible():
    # the local-VLM spike's documented failure mode: a catalog number ("608.",
    # "2704") emitted as a date, with no date-like separators at all.
    assert date_plausibility_score("2704") == 0.0
    assert date_plausibility_score("608.") == 0.0


def test_date_plausibility_score_invalid_day_or_month_is_implausible():
    assert date_plausibility_score("40.13.1977") == 0.0


def test_date_plausibility_score_year_out_of_range_is_implausible():
    assert date_plausibility_score("1.1.2999") == 0.0


def test_date_plausibility_score_month_year_only_is_plausible():
    assert date_plausibility_score("5.1977") == 1.0


def test_date_plausibility_score_unrecognized_shape_is_neutral():
    assert date_plausibility_score("Maj 1977") == 0.5


def test_date_plausibility_score_two_digit_year_is_handled():
    assert date_plausibility_score("22.5.77") == 1.0


def test_within_tier_score_date_is_soft_confidence_alone():
    assert within_tier_score("verbatimDate", "22.5.1977", soft_confidence=0.8) == 0.8


def test_within_tier_score_locality_without_corpus_falls_back_to_soft_alone():
    assert within_tier_score("verbatimLocality", "Kb", soft_confidence=0.6, locality_corpus=None) == 0.6


def test_within_tier_score_locality_averages_soft_and_corpus_support():
    corpus = {"kb": 40}
    score = within_tier_score("verbatimLocality", "Kb", soft_confidence=0.6, locality_corpus=corpus)
    # corpus_support_score("Kb", {"kb": 40}, saturation=10) == 1.0 -> avg(0.6, 1.0) == 0.8
    assert score == 0.8


def test_within_tier_score_local_vlm_agreement_none_leaves_base_unchanged():
    score = within_tier_score("verbatimDate", "22.5.1977", soft_confidence=0.8, local_vlm_agreement=None)
    assert score == 0.8


def test_within_tier_score_local_vlm_agreement_blends_in_on_top_of_base():
    score = within_tier_score("verbatimDate", "22.5.1977", soft_confidence=0.8, local_vlm_agreement=0.0)
    assert score == pytest.approx(0.4)


def test_within_tier_score_local_vlm_agreement_blends_on_top_of_locality_combo():
    corpus = {"kb": 40}
    # base = avg(0.6, corpus_support=1.0) = 0.8; then avg(0.8, agreement=1.0) = 0.9
    score = within_tier_score(
        "verbatimLocality", "Kb", soft_confidence=0.6, locality_corpus=corpus, local_vlm_agreement=1.0
    )
    assert score == pytest.approx(0.9)
