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


# --- whole-image-primary route (docs/.../2026-08-21-transcription-quality-push.md, D4) ---

from src.generate_submission_gemini import build_wholeimage_primary_submission_df


def _wholeimage(text_date, text_locality):
    return {"verbatimDate": (text_date, 1.0), "verbatimLocality": (text_locality, 1.0)}


def _per_card(text_date, text_locality):
    return {"verbatimDate": (text_date, 1.0), "verbatimLocality": (text_locality, 1.0)}


def test_wholeimage_primary_takes_its_text_from_the_whole_image_pass():
    df = build_wholeimage_primary_submission_df(
        ["a.jpeg"],
        {"a.jpeg": _wholeimage("22.5.1977", "Lodskovvad")},
        {"a.jpeg": _per_card("MISSING", "Ravneholm")},
    )
    assert df.loc[0, "verbatimDate"] == "22.5.1977"
    assert df.loc[0, "verbatimLocality"] == "Lodskovvad"


def test_wholeimage_primary_keeps_a_whole_image_MISSING_rather_than_filling_it():
    # A whole-image MISSING is a real reading of the whole specimen, not a coverage
    # gap -- overriding it with per-card text is the partial-MISSING fallback design
    # that regressed the leaderboard twice.
    df = build_wholeimage_primary_submission_df(
        ["a.jpeg"],
        {"a.jpeg": _wholeimage("MISSING", "MISSING")},
        {"a.jpeg": _per_card("22.5.1977", "Lodskovvad")},
    )
    assert df.loc[0, "verbatimDate"] == "MISSING"
    assert df.loc[0, "verbatimLocality"] == "MISSING"


def test_wholeimage_primary_falls_back_to_per_card_when_the_pass_never_covered_a_specimen():
    # Different case entirely: no whole-image row at all (an incomplete pass). Emitting
    # MISSING there would throw away a real per-card reading for no reason.
    df = build_wholeimage_primary_submission_df(
        ["a.jpeg"], {}, {"a.jpeg": _per_card("22.5.1977", "Lodskovvad")})
    assert df.loc[0, "verbatimDate"] == "22.5.1977"
    assert df.loc[0, "verbatimLocality"] == "Lodskovvad"


def test_wholeimage_primary_confidence_is_higher_when_the_routes_agree():
    agreeing = build_wholeimage_primary_submission_df(
        ["a.jpeg"], {"a.jpeg": _wholeimage("22.5.1977", "Lodskovvad")},
        {"a.jpeg": _per_card("22.5.1977", "Lodskovvad")})
    disagreeing = build_wholeimage_primary_submission_df(
        ["a.jpeg"], {"a.jpeg": _wholeimage("22.5.1977", "Lodskovvad")},
        {"a.jpeg": _per_card("11.9.1954", "Ravneholm")})
    for field in ("verbatimDate", "verbatimLocality"):
        assert agreeing.loc[0, f"{field}_confidence"] > disagreeing.loc[0, f"{field}_confidence"]


def test_wholeimage_primary_output_passes_submission_validation():
    from src.generate_submission import validate_submission

    image_files = ["a.jpeg", "b.jpeg"]
    df = build_wholeimage_primary_submission_df(
        image_files,
        {"a.jpeg": _wholeimage("22.5.1977", "Lodskovvad"),
         "b.jpeg": _wholeimage("MISSING", "MISSING")},
        {"a.jpeg": _per_card("22.5.1977", "Lodskovvad"),
         "b.jpeg": _per_card("MISSING", "MISSING")},
    )
    assert validate_submission(df, set(image_files)) == []


def test_wholeimage_primary_applies_text_cleanup():
    # sanitize_text / clean_locality must still run: this route changed where the
    # text comes from, not whether it gets cleaned.
    df = build_wholeimage_primary_submission_df(
        ["a.jpeg"],
        {"a.jpeg": _wholeimage("22.5.1977", "B&#248;rstinge Mose.")},
        {"a.jpeg": _per_card("22.5.1977", "Børstinge Mose")},
    )
    assert df.loc[0, "verbatimLocality"] == "Børstinge Mose"


def test_uncovered_specimen_does_not_get_trivial_maximum_confidence():
    # Regression: for a specimen the whole-image pass never covered, the text falls
    # back to per-card -- so scoring agreement between per-card text and itself would
    # return 1.0 and rank an unverified row above genuinely cross-confirmed ones.
    uncovered = build_wholeimage_primary_submission_df(
        ["a.jpeg"], {},
        {"a.jpeg": {"verbatimDate": ("22.5.1977", 0.34),
                    "verbatimLocality": ("Lodskovvad", 0.34)}},
    )
    cross_confirmed = build_wholeimage_primary_submission_df(
        ["a.jpeg"], {"a.jpeg": _wholeimage("22.5.1977", "Lodskovvad")},
        {"a.jpeg": _per_card("22.5.1977", "Lodskovvad")},
    )
    for field in ("verbatimDate", "verbatimLocality"):
        assert uncovered.loc[0, f"{field}_confidence"] < cross_confirmed.loc[0, f"{field}_confidence"]


def test_uncovered_specimen_confidence_tracks_per_card_self_consistency():
    def confidence_at(per_card_confidence):
        df = build_wholeimage_primary_submission_df(
            ["a.jpeg"], {},
            {"a.jpeg": {"verbatimDate": ("22.5.1977", per_card_confidence),
                        "verbatimLocality": ("Lodskovvad", per_card_confidence)}})
        return df.loc[0, "verbatimDate_confidence"]

    assert confidence_at(1.0) > confidence_at(0.67) > confidence_at(0.34)
