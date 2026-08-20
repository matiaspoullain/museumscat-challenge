import pytest

from src.generate_submission import REQUIRED_COLUMNS
from src.generate_submission_gemini import build_submission_df


def test_build_submission_df_merges_and_cleans_text():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"]}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("Bötö.", 0.6)},
    }

    df = build_submission_df(["a.jpeg"], manifest_by_image, predictions_by_crop)

    assert list(df.columns) == REQUIRED_COLUMNS
    row = df.set_index("image_file").loc["a.jpeg"]
    assert row["verbatimDate"] == "1.7.2000"
    assert row["verbatimDate_confidence"] == 0.9
    assert row["verbatimLocality"] == "Bøtø"  # cleaned: ö->ø, trailing period stripped
    assert row["verbatimLocality_confidence"] == 0.6


def test_build_submission_df_card_missing_a_field_does_not_join_missing_in():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 0.9), "verbatimLocality": ("Kb", 0.9)},
        "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.7), "verbatimLocality": ("MISSING", 0.7)},
    }
    df = build_submission_df(["a.jpeg"], manifest_by_image, predictions_by_crop)
    row = df.set_index("image_file").loc["a.jpeg"]
    assert row["verbatimDate"] == "22.5.1977"
    assert row["verbatimLocality"] == "Kb"


def test_build_submission_df_defaults_absent_specimen_to_missing():
    df = build_submission_df(["a.jpeg"], {}, {})
    row = df.set_index("image_file").loc["a.jpeg"]
    assert row["verbatimDate"] == "MISSING"
    assert row["verbatimLocality"] == "MISSING"


def test_build_submission_df_applies_wholeimage_fallback_and_cleans_it():
    wholeimage_predictions_by_specimen = {
        "a.jpeg": {"verbatimDate": ("22.5.1977", 0.8), "verbatimLocality": ("Bötö.", 0.8)},
    }
    df = build_submission_df(["a.jpeg"], {}, {}, wholeimage_predictions_by_specimen)
    row = df.set_index("image_file").loc["a.jpeg"]
    assert row["verbatimDate"] == "22.5.1977"
    assert row["verbatimLocality"] == "Bøtø"  # fallback value also gets ö->ø + trailing-period cleanup


def test_build_submission_df_applies_undercoverage_confidence_penalty():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 0.9), "verbatimLocality": ("Kb", 0.9)},
        "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.7), "verbatimLocality": ("MISSING", 0.7)},
    }
    df = build_submission_df(["a.jpeg"], manifest_by_image, predictions_by_crop)
    row = df.set_index("image_file").loc["a.jpeg"]
    # 2 cards processed but only 1 contributed a value -> flagged, penalized by 0.7x
    assert row["verbatimDate_confidence"] == pytest.approx(0.9 * 0.7)
    assert row["verbatimLocality_confidence"] == pytest.approx(0.9 * 0.7)


def test_build_submission_df_rank_refinement_off_by_default_matches_unrefined():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"]}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 1.0), "verbatimLocality": ("Kb", 1.0)},
    }
    without_flag = build_submission_df(["a.jpeg"], manifest_by_image, predictions_by_crop)
    with_flag_false = build_submission_df(
        ["a.jpeg"], manifest_by_image, predictions_by_crop, apply_rank_refinement=False
    )
    assert without_flag.equals(with_flag_false)


def test_build_submission_df_rank_refinement_text_stays_byte_identical():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"], "b.jpeg": []}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 1.0), "verbatimLocality": ("Kb", 1.0)},
    }
    base = build_submission_df(["a.jpeg", "b.jpeg"], manifest_by_image, predictions_by_crop)
    refined = build_submission_df(
        ["a.jpeg", "b.jpeg"], manifest_by_image, predictions_by_crop, apply_rank_refinement=True
    )
    assert list(refined["verbatimDate"]) == list(base["verbatimDate"])
    assert list(refined["verbatimLocality"]) == list(base["verbatimLocality"])


def test_build_submission_df_rank_refinement_elevates_structural_missing():
    # 0 crops processed for "b.jpeg" at all -- structural MISSING, should be
    # elevated well above merge_records.MISSING_CONFIDENCE's default 0.5.
    df = build_submission_df(
        ["b.jpeg"], {"b.jpeg": []}, {}, apply_rank_refinement=True,
    )
    row = df.set_index("image_file").loc["b.jpeg"]
    assert row["verbatimDate_confidence"] > 0.5
    assert row["verbatimLocality_confidence"] > 0.5


def test_build_submission_df_rank_refinement_within_tier_score_breaks_ties():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"], "b.jpeg": ["b_card00.jpeg"]}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 0.667), "verbatimLocality": ("Kb", 0.667)},
        "b_card00.jpeg": {"verbatimDate": ("22.5.1977", 0.667), "verbatimLocality": ("Kb", 0.667)},
    }
    soft_predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 0.9), "verbatimLocality": ("Kb", 0.9)},
        "b_card00.jpeg": {"verbatimDate": ("22.5.1977", 0.3), "verbatimLocality": ("Kb", 0.3)},
    }
    df = build_submission_df(
        ["a.jpeg", "b.jpeg"], manifest_by_image, predictions_by_crop,
        apply_rank_refinement=True, soft_predictions_by_crop=soft_predictions_by_crop,
    )
    a_conf = df.set_index("image_file").loc["a.jpeg", "verbatimDate_confidence"]
    b_conf = df.set_index("image_file").loc["b.jpeg", "verbatimDate_confidence"]
    # both tied at 0.667 before refinement; higher soft consistency (a) must
    # now rank strictly above the lower one (b), still within the same tier.
    assert a_conf > b_conf
    assert 0.667 <= b_conf < a_conf < 0.667 + 0.033  # stays inside the 0.667/0.7 gap


def test_build_submission_df_local_vlm_agreement_breaks_ties_further():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"], "b.jpeg": ["b_card00.jpeg"]}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 1.0), "verbatimLocality": ("Kb", 1.0)},
        "b_card00.jpeg": {"verbatimDate": ("22.5.1977", 1.0), "verbatimLocality": ("Kb", 1.0)},
    }
    # both tied at raw confidence 1.0, same soft consistency too (no
    # soft_predictions_by_crop passed) -- local VLM agreement should be the
    # only thing distinguishing them.
    local_vlm_predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 0.8), "verbatimLocality": ("Kb", 0.8)},
        "b_card00.jpeg": {"verbatimDate": ("1.1.1900", 0.8), "verbatimLocality": ("Ti", 0.8)},
    }
    df = build_submission_df(
        ["a.jpeg", "b.jpeg"], manifest_by_image, predictions_by_crop,
        apply_rank_refinement=True, soft_predictions_by_crop=predictions_by_crop,
        local_vlm_predictions_by_crop=local_vlm_predictions_by_crop,
    )
    a_conf = df.set_index("image_file").loc["a.jpeg", "verbatimDate_confidence"]
    b_conf = df.set_index("image_file").loc["b.jpeg", "verbatimDate_confidence"]
    assert a_conf > b_conf  # a: local VLM agrees; b: local VLM disagrees


def test_build_submission_df_local_vlm_predictions_optional():
    # Omitting local_vlm_predictions_by_crop entirely must not raise and must
    # behave identically to passing an empty dict (no local VLM run available
    # for this split yet).
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"]}
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("22.5.1977", 1.0), "verbatimLocality": ("Kb", 1.0)},
    }
    without = build_submission_df(
        ["a.jpeg"], manifest_by_image, predictions_by_crop,
        apply_rank_refinement=True, soft_predictions_by_crop=predictions_by_crop,
    )
    with_empty = build_submission_df(
        ["a.jpeg"], manifest_by_image, predictions_by_crop,
        apply_rank_refinement=True, soft_predictions_by_crop=predictions_by_crop,
        local_vlm_predictions_by_crop={},
    )
    assert without.equals(with_empty)
