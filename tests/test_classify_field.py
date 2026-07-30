from src.classify_field import classify_field


def test_classifies_standard_date():
    assert classify_field("27.5.2022") == "date"


def test_classifies_roman_numeral_date():
    assert classify_field("27.IV.2022") == "date"


def test_classifies_partial_date():
    assert classify_field("IV.91") == "date"


def test_classifies_slash_date():
    assert classify_field("4/5 70") == "date"


def test_classifies_locality_kb():
    assert classify_field("Kb") == "locality"


def test_classifies_locality_with_coords():
    assert classify_field("DENMARK: NEZ: Helsingor, Gronne Vestkile, 56.0375N, 12.5598E") == "locality"


def test_classifies_confound_collector():
    assert classify_field("Coll. J. P. Johansen") == "confound"


def test_classifies_confound_determination():
    assert classify_field("det. Hansen") == "confound"


def test_classifies_confound_museum_boilerplate():
    assert classify_field("STATENS NATURHISTORISKE MUSEUM") == "confound"


def test_classifies_confound_dania():
    assert classify_field("Dania coll.") == "confound"


def test_classifies_empty_string_as_confound():
    assert classify_field("") == "confound"


def test_classifies_bare_year_as_date():
    assert classify_field("1994") == "date"
    assert classify_field("1922") == "date"


def test_classifies_short_number_noise_as_confound():
    assert classify_field("1") == "confound"
    assert classify_field("18") == "confound"
    assert classify_field("166") == "confound"


def test_classifies_ocr_misread_coll_as_confound():
    assert classify_field("Loll.") == "confound"
    assert classify_field("Løll.") == "confound"


def test_classifies_spelled_out_danish_month_as_date():
    assert classify_field("Juli 1930") == "date"
    assert classify_field("Juni 98") == "date"


def test_classifies_garbled_digit_dominant_fragment_as_date():
    # real reconstructed-line fragments of "22.5.1977" and "1.7.2000",
    # split/duplicated by the line detector -- punctuation-mangled but
    # still mostly digits (>=3 digit chars), so treating them as noisy
    # date guesses beats defaulting to MISSING (verified: NED ~0.44 vs 1.0)
    assert classify_field("22 5 1 1") == "date"
    assert classify_field("19 77. 1") == "date"
    assert classify_field("1l 200 0") == "date"


def test_digit_dominant_rule_does_not_override_short_number_confound():
    assert classify_field("18") == "confound"
    assert classify_field("166") == "confound"


def test_digit_dominant_rule_does_not_catch_alpha_dominant_locality():
    assert classify_field(
        "DENMARK: NEZ: Helsingor, Gronne Vestkile, 56.0375N, 12.5598E"
    ) == "locality"


def test_weak_digit_noise_below_date_threshold_is_confound_not_locality():
    # short digit-dominant scraps (<3 digit chars) are too weak to guess
    # as a date, but still shouldn't contaminate locality -- drop them
    assert classify_field("6t 1") == "confound"
    assert classify_field("1 1") == "confound"
    assert classify_field("K 8") == "confound"
