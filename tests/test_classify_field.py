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
