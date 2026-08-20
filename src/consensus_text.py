"""Track B (docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md):
text/NED reduction from Gemini samples already collected, distinct from Track
A's confidence-only reranking (src.confidence_rank_rules). Smaller headroom
than Track A -- oracle best-of-3-sample text selection only takes dev mean NED
from 0.037/0.169 down to ~0.033/0.150 (see the plan's Context table) -- but it
lowers risk at every coverage level, independent of rank. Every function here
is zero-Gemini-quota: they operate on raw_samples_json already stored in
data/processed/gemini/*.csv, or on a corpus built from that same data
(src.locality_corpus).

mbr_consensus_scores is a drop-in for src.gemini_transcribe.
self_consistency_scores via src.calibrate_gemini_confidence.
rescored_predictions_by_crop -- same {field: (text, confidence)} shape, same
confidence formula, only the TEXT selection differs on a no-majority tie.

decode_html_entities and snap_to_corpus are applied to already-merged text
(after src.clean_gemini_predictions), not to individual samples -- see
src.evaluate_gemini_dev's Track B reporting block for where they're wired in.
"""
import collections
import html

from src.gemini_transcribe import FIELDS
from src.metrics import edit_distance, ned

MIN_SNAP_SOURCE_LENGTH = 5
MIN_SNAP_TARGET_COUNT = 10
MAX_SNAP_EDIT_DISTANCE = 2


def _normalize_for_matching(text):
    return " ".join(text.strip().lower().split())


def mbr_consensus_scores(samples):
    """Like src.gemini_transcribe.self_consistency_scores, but selects the
    representative TEXT for each field via minimum Bayes risk under NED (the
    sample minimizing total edit distance to the other samples) instead of
    Counter.most_common(1)'s tie-break, which is arbitrary (Python's stable
    sort keeps first-encountered order) whenever no sample commands a
    majority -- e.g. all 3 samples mutually distinct after normalization.
    When one text does command a majority, that text is already its own MBR
    winner under NED (distance 0 to itself and to every matching sample), so
    behavior is identical to self_consistency_scores in that case. Confidence
    is unchanged: modal-agreement fraction, same formula either way -- this
    only changes which literal string gets emitted on a no-majority tie."""
    scores = {}
    for field in FIELDS:
        texts = [sample[field] for sample in samples]
        counts = collections.Counter(_normalize_for_matching(t) for t in texts)
        modal_normalized, modal_count = counts.most_common(1)[0]
        if modal_count > 1:
            text = next(t for t in texts if _normalize_for_matching(t) == modal_normalized)
        else:
            is_date = field == "verbatimDate"
            text = min(
                texts,
                key=lambda t: sum(ned(t, other, is_date=is_date) for other in texts if other != t),
            )
        scores[field] = (text, modal_count / len(texts))
    return scores


def decode_html_entities(text):
    """Decodes stray HTML character references (observed in the wild:
    "B&#248;rstinge Mose" for "Børstinge Mose", presumably a web-scraped
    reference source leaking into a sample) via the stdlib html.unescape.
    A no-op on ordinary text or "MISSING"."""
    return html.unescape(text)


def _snap_segment(segment, corpus, min_target_count, min_source_length, max_edit_distance):
    norm = _normalize_for_matching(segment)
    if len(norm) < min_source_length:
        return segment  # too short to safely snap -- a legitimate abbreviation ("Kb", "Ti")
    if corpus.get(norm, 0) >= min_target_count:
        return segment  # already well-supported on its own, nothing to snap to
    best_target, best_count = None, 0
    for candidate, count in corpus.items():
        if count < min_target_count or candidate == norm:
            continue
        if abs(len(candidate) - len(norm)) > max_edit_distance:
            continue
        if count > best_count and edit_distance(norm, candidate) <= max_edit_distance:
            best_target, best_count = candidate, count
    if best_target is None:
        return segment
    return " ".join(word.capitalize() for word in best_target.split())


def snap_to_corpus(text, corpus, min_target_count=MIN_SNAP_TARGET_COUNT,
                    min_source_length=MIN_SNAP_SOURCE_LENGTH, max_edit_distance=MAX_SNAP_EDIT_DISTANCE):
    """Snaps a rare verbatimLocality segment to a frequent near-neighbour
    already observed elsewhere in the corpus (src.locality_corpus.
    build_locality_corpus), when it's safe to: the segment isn't already
    well-supported on its own, is long enough not to be a legitimate short
    abbreviation, and a within-max_edit_distance target is frequent enough to
    trust. The corpus only stores normalized (lowercased) text, so casing is
    reconstructed via simple title-casing -- correct for the vast majority of
    Danish place names.

    Deliberately snaps only to strings this pipeline actually observed
    elsewhere in real Gemini samples, never an external gazetteer -- ground
    truth preserves historical spelling verbatim, and src.
    clean_gemini_predictions' docstring already explains why a gazetteer
    risks corrupting a legitimate historical spelling it doesn't recognize."""
    if text == "MISSING":
        return text
    segments = [s.strip() for s in text.split("|")]
    snapped = [
        _snap_segment(s, corpus, min_target_count, min_source_length, max_edit_distance)
        for s in segments
    ]
    return " | ".join(snapped)
