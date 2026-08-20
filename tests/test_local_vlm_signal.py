from src.local_vlm_signal import build_local_vlm_predictions_by_crop, local_vlm_agreement_score


def test_local_vlm_agreement_score_agrees():
    local_merged = {"a.jpeg": {"verbatimDate": ("22.5.1977", 0.8)}}
    covered_crops = {"a.jpeg": ["a_card00.jpeg"]}
    score = local_vlm_agreement_score("a.jpeg", "verbatimDate", "22.5.1977", local_merged, covered_crops)
    assert score == 1.0


def test_local_vlm_agreement_score_disagrees():
    local_merged = {"a.jpeg": {"verbatimDate": ("22.5.1978", 0.8)}}
    covered_crops = {"a.jpeg": ["a_card00.jpeg"]}
    score = local_vlm_agreement_score("a.jpeg", "verbatimDate", "22.5.1977", local_merged, covered_crops)
    assert score == 0.0


def test_local_vlm_agreement_score_case_and_whitespace_insensitive():
    local_merged = {"a.jpeg": {"verbatimLocality": ("  Kb  ", 0.8)}}
    covered_crops = {"a.jpeg": ["a_card00.jpeg"]}
    score = local_vlm_agreement_score("a.jpeg", "verbatimLocality", "kb", local_merged, covered_crops)
    assert score == 1.0


def test_local_vlm_agreement_score_neutral_when_specimen_not_covered():
    score = local_vlm_agreement_score("b.jpeg", "verbatimDate", "22.5.1977", {}, {})
    assert score == 0.5


def test_local_vlm_agreement_score_neutral_when_gemini_text_is_missing():
    local_merged = {"a.jpeg": {"verbatimDate": ("22.5.1977", 0.8)}}
    covered_crops = {"a.jpeg": ["a_card00.jpeg"]}
    score = local_vlm_agreement_score("a.jpeg", "verbatimDate", "MISSING", local_merged, covered_crops)
    assert score == 0.5


def test_local_vlm_agreement_score_neutral_when_local_vlm_text_is_missing():
    local_merged = {"a.jpeg": {"verbatimDate": ("MISSING", 0.8)}}
    covered_crops = {"a.jpeg": ["a_card00.jpeg"]}
    score = local_vlm_agreement_score("a.jpeg", "verbatimDate", "22.5.1977", local_merged, covered_crops)
    assert score == 0.5


def test_build_local_vlm_predictions_by_crop_reads_csv(tmp_path):
    path = tmp_path / "local.csv"
    path.write_text(
        "crop_filename,source_image_file,local_vlm_verbatimDate,local_vlm_verbatimDate_confidence,"
        "local_vlm_verbatimLocality,local_vlm_verbatimLocality_confidence,raw_samples_json\n"
        'a_card00.jpeg,a.jpeg,22.5.1977,0.8,Kb,0.6,"[]"\n',
        encoding="utf-8",
    )
    by_crop = build_local_vlm_predictions_by_crop(str(path))
    assert by_crop["a_card00.jpeg"]["verbatimDate"] == ("22.5.1977", 0.8)
    assert by_crop["a_card00.jpeg"]["verbatimLocality"] == ("Kb", 0.6)


def test_build_local_vlm_predictions_by_crop_missing_file_returns_empty(tmp_path):
    by_crop = build_local_vlm_predictions_by_crop(str(tmp_path / "does_not_exist.csv"))
    assert by_crop == {}
