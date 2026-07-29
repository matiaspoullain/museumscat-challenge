from src.pipeline import classify_card_field_values, process_specimen, run_pipeline


def test_classify_card_field_values_joins_same_field_lines():
    lines = [{"text": "Svinø", "confidence": 0.9}, {"text": "strand", "confidence": 0.7}]
    assert classify_card_field_values(lines) == {"locality": ("Svinø strand", 0.8)}


def test_classify_card_field_values_drops_confound_lines():
    lines = [{"text": "Coll. J. P. Johansen", "confidence": 0.9}]
    assert classify_card_field_values(lines) == {}


def test_classify_card_field_values_separates_date_and_locality():
    lines = [{"text": "27.5.2022", "confidence": 0.9}, {"text": "Kb", "confidence": 0.8}]
    result = classify_card_field_values(lines)
    assert result == {"date": ("27.5.2022", 0.9), "locality": ("Kb", 0.8)}


def test_classify_card_field_values_empty_lines_is_empty_dict():
    assert classify_card_field_values([]) == {}


def test_process_specimen_joins_two_cards_for_the_same_field():
    lines_by_crop = {
        "a_card00.jpeg": [{"text": "Ti", "confidence": 0.8}],
        "a_card01.jpeg": [{"text": "Kulhuset Jaegerspris", "confidence": 0.6}],
    }
    result = process_specimen(["a_card00.jpeg", "a_card01.jpeg"], lines_by_crop)
    assert result["verbatimLocality"] == ("Ti | Kulhuset Jaegerspris", 0.7)
    assert result["verbatimDate"] == ("MISSING", 0.5)


def test_process_specimen_no_cards_is_missing_both_fields():
    result = process_specimen([], {})
    assert result["verbatimDate"] == ("MISSING", 0.5)
    assert result["verbatimLocality"] == ("MISSING", 0.5)


def test_process_specimen_ignores_crops_with_no_recorded_lines():
    result = process_specimen(["missing_from_predictions.jpeg"], {})
    assert result["verbatimDate"] == ("MISSING", 0.5)


def test_run_pipeline_defaults_absent_specimens_to_missing():
    lines_by_crop = {"a_card00.jpeg": [{"text": "Kb", "confidence": 0.9}]}
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"]}
    result = run_pipeline(["a.jpeg", "b.jpeg"], manifest_by_image, lines_by_crop)
    assert result["a.jpeg"]["verbatimLocality"] == ("Kb", 0.9)
    assert result["b.jpeg"]["verbatimLocality"] == ("MISSING", 0.5)
