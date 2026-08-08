from src.confidence_penalty_rules import apply_confidence_penalty, count_processed_and_contributing


def test_count_processed_and_contributing_counts_only_gemini_processed_crops():
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.9)},
        "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.7), "verbatimLocality": ("Kb", 0.7)},
    }
    # card02 is in the manifest (a detected crop) but never sent to Gemini
    # (filtered as a confound by the relevance classifier) -- must not count
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg", "a_card02.jpeg"]}
    n_processed, n_contributing = count_processed_and_contributing(
        "a.jpeg", "verbatimDate", manifest_by_image, predictions_by_crop
    )
    assert (n_processed, n_contributing) == (2, 1)


def test_apply_confidence_penalty_flags_incomplete_multi_card_coverage():
    # 2 processed cards, only 1 contributed to this field -- flagged
    penalized = apply_confidence_penalty(1.0, n_processed=2, n_contributing=1)
    assert penalized == 0.7


def test_apply_confidence_penalty_all_cards_contributed_not_flagged():
    unpenalized = apply_confidence_penalty(1.0, n_processed=2, n_contributing=2)
    assert unpenalized == 1.0


def test_apply_confidence_penalty_single_card_missing_field_not_flagged():
    # this is the ordinary "this card just has no date" case, not a coverage risk
    unpenalized = apply_confidence_penalty(0.9, n_processed=1, n_contributing=0)
    assert unpenalized == 0.9
