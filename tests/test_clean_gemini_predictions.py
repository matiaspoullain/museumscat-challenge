from src.clean_gemini_predictions import clean_locality, clean_merged_predictions


def test_clean_locality_replaces_o_umlaut_with_o_slash():
    assert clean_locality("Bötö plv") == "Bøtø plv"


def test_clean_locality_replaces_uppercase_o_umlaut():
    assert clean_locality("ÖSTERBY") == "ØSTERBY"


def test_clean_locality_strips_trailing_period():
    assert clean_locality("Dyrehaven.") == "Dyrehaven"


def test_clean_locality_strips_trailing_period_and_whitespace():
    assert clean_locality("Rude Skov. ") == "Rude Skov"


def test_clean_locality_leaves_missing_untouched():
    assert clean_locality("MISSING") == "MISSING"


def test_clean_locality_leaves_clean_text_untouched():
    assert clean_locality("Kb") == "Kb"


def test_clean_locality_does_not_strip_internal_periods():
    assert clean_locality("SKIBBY, S.J.") == "SKIBBY, S.J"


def test_clean_merged_predictions_only_touches_locality():
    merged = {
        "a.jpeg": {"verbatimDate": ("22.5.1977.", 0.9), "verbatimLocality": ("Bötö.", 0.8)},
    }
    cleaned = clean_merged_predictions(merged)
    assert cleaned["a.jpeg"]["verbatimDate"] == ("22.5.1977.", 0.9)
    assert cleaned["a.jpeg"]["verbatimLocality"] == ("Bøtø", 0.8)
