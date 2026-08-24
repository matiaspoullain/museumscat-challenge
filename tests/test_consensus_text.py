from src.old.consensus_text import decode_html_entities, mbr_consensus_scores, snap_to_corpus


def _samples(date_texts, locality_texts):
    return [{"verbatimDate": d, "verbatimLocality": l} for d, l in zip(date_texts, locality_texts)]


def test_mbr_consensus_scores_keeps_majority_text_unchanged():
    samples = _samples(["22.5.1977", "22.5.1977", "22.5.1978"], ["Kb", "Kb", "Ti"])
    scores = mbr_consensus_scores(samples)
    assert scores["verbatimDate"] == ("22.5.1977", 2 / 3)
    assert scores["verbatimLocality"] == ("Kb", 2 / 3)


def test_mbr_consensus_scores_picks_medoid_when_all_three_disagree():
    # "22.5.1977" is a 1-char slip from both others; "1.1.1900" and "9.9.1999"
    # are far from everything -- MBR should pick the central one, not
    # whichever sample happened to come first.
    samples = _samples(["1.1.1900", "22.5.1977", "22.5.1978"], ["Kb", "Kb", "Kb"])
    scores = mbr_consensus_scores(samples)
    assert scores["verbatimDate"][0] in ("22.5.1977", "22.5.1978")
    assert scores["verbatimDate"][1] == 1 / 3  # confidence: still no exact majority


def test_mbr_consensus_scores_confidence_matches_self_consistency():
    # Confidence formula is unchanged from self_consistency_scores -- only
    # text selection differs on a no-majority tie.
    samples = _samples(["A", "B", "C"], ["X", "X", "Y"])
    scores = mbr_consensus_scores(samples)
    assert scores["verbatimLocality"] == ("X", 2 / 3)


def test_decode_html_entities_numeric_reference():
    assert decode_html_entities("B&#248;rstinge Mose") == "Børstinge Mose"


def test_decode_html_entities_leaves_plain_text_unchanged():
    assert decode_html_entities("Kb") == "Kb"


def test_decode_html_entities_missing_passthrough():
    assert decode_html_entities("MISSING") == "MISSING"


def test_snap_to_corpus_snaps_hapax_near_frequent_entry():
    corpus = {"sjælland": 20}
    result = snap_to_corpus("Sielland", corpus, min_target_count=10)
    assert result == "Sjælland"


def test_snap_to_corpus_leaves_frequent_text_alone():
    corpus = {"kb": 40}
    result = snap_to_corpus("Kb", corpus, min_target_count=10)
    assert result == "Kb"  # already frequent itself, no snap needed


def test_snap_to_corpus_leaves_short_text_alone():
    # short strings are legitimate abbreviations ("Kb", "Ti") -- never snap
    corpus = {"ti": 5}
    result = snap_to_corpus("Kb", corpus, min_target_count=1)
    assert result == "Kb"


def test_snap_to_corpus_leaves_missing_alone():
    assert snap_to_corpus("MISSING", {"kb": 40}) == "MISSING"


def test_snap_to_corpus_no_nearby_frequent_entry_leaves_alone():
    corpus = {"sjælland": 20}
    result = snap_to_corpus("Randomtown", corpus, min_target_count=10)
    assert result == "Randomtown"


def test_snap_to_corpus_preserves_pipe_segments():
    corpus = {"sjælland": 20}
    result = snap_to_corpus("Ti | Sielland", corpus, min_target_count=10)
    assert result == "Ti | Sjælland"
