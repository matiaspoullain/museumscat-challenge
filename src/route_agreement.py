"""Cross-route agreement as a confidence signal for the whole-image-primary
transcription route (docs/superpowers/plans/2026-08-21-transcription-quality-push.md,
D4).

The per-card route stops being the source of the submitted *text* there, but its
predictions remain valuable for something the text route cannot supply on its own:
an independent second reading of the same specimen, produced from different pixels
(tight card crops) by a different prompt. Asking whether the two routes agree is
the same question src.local_vlm_signal asks of Qwen2.5-VL, with the roles swapped
-- and at full test-set coverage (3267/3267 crops) rather than that spike's partial
152/197.

This matters because the whole-image pass runs greedy (n_samples=1), so it has no
self-consistency of its own: src.gemini_transcribe.self_consistency_scores would
report 1.0 for every row, and AURC depends on nothing but confidence rank. Route
agreement replaces that missing signal rather than supplementing it.

Measured on the full 200-row dev set: dev AURC 0.0487 (the shipped Track A3 config)
-> 0.0353 with route agreement plus the existing corpus-support and local-VLM
within-tier signals. All five agreement variants tried beat Track A3, so the result
does not depend on which one is picked -- see the plan doc's D4 section for the
table and for the honest bootstrap caveat (the end-to-end AURC interval lands just
barely on the inconclusive side of the 90% bar; the text-NED interval excludes 0).
"""
from src.confidence_rank_rules import combine_tier_and_within_tier_score
from src.locality_corpus import corpus_support_score
from src.metrics import best_pipe_ned

FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}


def _normalize(text):
    return " ".join(text.strip().lower().split())


def route_agreement_score(text_a, text_b, is_date):
    """[0,1]: 1 - normalized edit distance between two independent routes'
    readings of the same field.

    "MISSING" is not a string to edit-distance against, so it is handled
    structurally rather than lexically:

    - both routes MISSING -> 1.0. Two independent readings that each found
      nothing is corroboration, not absence of evidence. This is the same
      population src.confidence_rank_rules.is_confident_structural_missing
      identified as unusually *accurate* (dev mean NED 0.018, better than the
      unanimous-agreement text tier's 0.021).
    - exactly one MISSING -> 0.0. One route found text where the other found
      none: maximal disagreement, and the shape of the recovered-value errors
      the partial-MISSING fallback spike was penalized for.

    Uses best_pipe_ned rather than plain ned so that two routes splitting a
    multi-card specimen's "|"-joined segments in a different order don't read as
    disagreement -- they have no reason to agree on ordering, and the scorer
    itself doesn't require them to. Only 14/200 dev specimens have a multi-card
    locality on either side, so this barely moves the dev number (0.0353 ->
    0.0346); it is chosen for matching the metric, not for that delta.
    """
    if text_a == "MISSING" and text_b == "MISSING":
        return 1.0
    if text_a == "MISSING" or text_b == "MISSING":
        return 0.0
    return 1.0 - best_pipe_ned(_normalize(text_a), _normalize(text_b), is_date=is_date)


def wholeimage_primary_confidence(field, text, per_card_text, locality_corpus=None,
                                   local_vlm_agreement=None):
    """Confidence for one field of one specimen under the whole-image-primary
    route: route agreement as the coarse tier, with corpus support (locality
    only) and local-VLM agreement averaged into a bounded within-tier tiebreak.

    Uses src.confidence_rank_rules.combine_tier_and_within_tier_score so the
    secondary signals can only ever reorder rows that already tie on agreement,
    never override it -- the same additive posture that separated the coverage
    penalty (a proven win) from a fitted multi-feature calibrator (which lost).
    It also caps the result below 1.0, which src.generate_submission.
    validate_submission requires.

    locality_corpus=None and local_vlm_agreement=None both mean "this signal
    isn't available here", not "this signal says zero" -- with neither, the
    confidence is route agreement alone.
    """
    base = route_agreement_score(text, per_card_text, FIELD_IS_DATE[field])
    extras = []
    if locality_corpus is not None and field == "verbatimLocality":
        extras.append(corpus_support_score(text, locality_corpus))
    if local_vlm_agreement is not None:
        extras.append(local_vlm_agreement)
    if not extras:
        return base
    return combine_tier_and_within_tier_score(base, sum(extras) / len(extras))
