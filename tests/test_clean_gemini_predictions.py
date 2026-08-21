from src.clean_gemini_predictions import clean_locality, clean_merged_predictions, sanitize_text


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


def test_clean_locality_strips_trailing_loev_segment():
    assert clean_locality("Tj | Løv.") == "Tj"


def test_clean_locality_strips_trailing_mus_loev_segment():
    assert clean_locality("Al | Mus. Løv.") == "Al"


def test_clean_locality_strips_trailing_loev_segment_case_insensitive():
    assert clean_locality("Kb | mus. lov") == "Kb"


def test_clean_locality_leaves_other_pipe_joined_localities_untouched():
    assert clean_locality("Grenaa | Brastrup") == "Grenaa | Brastrup"


def test_clean_merged_predictions_only_touches_locality():
    merged = {
        "a.jpeg": {"verbatimDate": ("22.5.1977.", 0.9), "verbatimLocality": ("Bötö.", 0.8)},
    }
    cleaned = clean_merged_predictions(merged)
    assert cleaned["a.jpeg"]["verbatimDate"] == ("22.5.1977.", 0.9)
    assert cleaned["a.jpeg"]["verbatimLocality"] == ("Bøtø", 0.8)


def test_sanitize_text_decodes_html_entities():
    assert sanitize_text("B&#248;rstinge Mose") == "Børstinge Mose"


def test_sanitize_text_normalizes_decomposed_danish_letters_to_nfc():
    # "Nivå" written as "Niva" + U+030A COMBINING RING ABOVE scores a nonzero
    # edit distance against the NFC ground truth despite looking identical --
    # all 200 dev ground-truth values are NFC (verified 2026-08-21).
    decomposed = "Nivå"
    assert decomposed != "Nivå"
    assert sanitize_text(decomposed) == "Nivå"


def test_sanitize_text_strips_control_characters():
    assert sanitize_text("Neksel\x00 S.") == "Neksel S."


def test_sanitize_text_collapses_embedded_newline_into_a_space():
    assert sanitize_text("Gislev kloster\n Vestermose") == "Gislev kloster Vestermose"


def test_sanitize_text_leaves_clean_text_untouched():
    assert sanitize_text("Jydske Aas Vendsyssel") == "Jydske Aas Vendsyssel"


def test_sanitize_text_leaves_missing_untouched():
    assert sanitize_text("MISSING") == "MISSING"


def test_sanitize_text_returns_missing_when_nothing_survives():
    assert sanitize_text("\x00 \x01") == "MISSING"


def test_clean_locality_sanitizes_before_applying_its_own_rules():
    # &#246; is o-umlaut, which clean_locality's own rule maps to ø -- so the
    # entity has to be decoded first for that rule to see it at all.
    assert clean_locality("B&#246;t&#246;") == "Bøtø"


def test_clean_merged_predictions_sanitizes_the_date_field_too():
    merged = {"a.jpeg": {"verbatimDate": ("12&#46;5&#46;1947", 0.9),
                          "verbatimLocality": ("Kb", 0.8)}}
    cleaned = clean_merged_predictions(merged)
    assert cleaned["a.jpeg"]["verbatimDate"] == ("12.5.1947", 0.9)


def test_clean_merged_predictions_does_not_strip_a_dates_trailing_period():
    # clean_locality's trailing-punctuation rule is locality-only; dates keep
    # theirs (src.metrics.normalize_date_punctuation makes it free anyway).
    merged = {"a.jpeg": {"verbatimDate": ("22.5.1977.", 0.9),
                          "verbatimLocality": ("Kb", 0.8)}}
    cleaned = clean_merged_predictions(merged)
    assert cleaned["a.jpeg"]["verbatimDate"] == ("22.5.1977.", 0.9)
