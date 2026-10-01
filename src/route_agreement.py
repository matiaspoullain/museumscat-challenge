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
from collections.abc import Mapping

from src.confidence_rank_rules import combine_tier_and_within_tier_score
from src.locality_corpus import corpus_support_score
from src.metrics import best_pipe_ned

FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def route_agreement_score(text_a: str, text_b: str, is_date: bool) -> float:
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


# Risk-ordered tiers for the whole-image-primary route, measured on the full
# 200-row dev set (mean NED per group, lower is better):
#
#   field     exact-agree   both-MISSING   one-MISSING   partial-agree
#   date          0.0132        0.0167        0.0296         0.1294
#   locality      0.0464        0.1212        0.1750         0.2195
#
# Two orderings in the original design were wrong, and both are visible
# independently in each field rather than in one alone:
#
# 1. both-MISSING sat at 1.0, the same top tier as two routes reading identical
#    text. That holds for dates (a specimen genuinely without a date is common,
#    and both routes see that correctly) but not for localities, where the group
#    is 2.6x riskier than exact agreement -- often the label does carry a
#    locality that both routes missed.
# 2. one-MISSING sat at 0.0, the bottom tier, yet it measures *better* than
#    partial agreement in both fields. If one route read text and the other read
#    none, the text is usually right and the other route simply failed to see the
#    card; that is weaker evidence than corroboration, not the worst evidence
#    available. Partial agreement -- two routes reading the same card differently
#    -- is the genuinely worst group of the four.
#
# AURC depends only on rank order, so these constants encode an ordering, not
# fitted magnitudes; `partial_scale` is what pushes the partial-agreement
# continuum below both MISSING groups.
ROUTE_TIERS: dict[str, dict[str, float]] = {
    "verbatimDate": {"both_missing": 0.95, "one_missing": 0.75, "partial_scale": 0.5},
    "verbatimLocality": {"both_missing": 0.75, "one_missing": 0.55, "partial_scale": 0.5},
}


def route_tier_score(field: str, text: str, per_card_text: str) -> float:
    """Risk-ordered coarse tier for one field, from the two routes' readings.

    route_agreement_score is the raw agreement primitive; this is the ranking
    built on top of it, remapping the two structural MISSING cases and the
    partial-agreement band to the order the dev-set risk table actually shows
    (see ROUTE_TIERS). Exact agreement keeps the top tier at 1.0.
    """
    tiers = ROUTE_TIERS[field]
    if text == "MISSING" and per_card_text == "MISSING":
        return tiers["both_missing"]
    if text == "MISSING" or per_card_text == "MISSING":
        return tiers["one_missing"]
    agreement = route_agreement_score(text, per_card_text, FIELD_IS_DATE[field])
    if agreement == 1.0:
        return 1.0
    return agreement * tiers["partial_scale"]


def wholeimage_primary_confidence(field: str, text: str, per_card_text: str,
                                  locality_corpus: Mapping[str, int] | None = None,
                                  local_vlm_agreement: float | None = None,
                                  per_card_confidence: float | None = None) -> float:
    """Confidence for one field of one specimen under the whole-image-primary
    route: the risk-ordered route tier as the coarse rank, with corpus support
    (locality only), local-VLM agreement and the per-card route's own
    self-consistency averaged into a bounded within-tier tiebreak.

    Uses src.confidence_rank_rules.combine_tier_and_within_tier_score so the
    secondary signals can only ever reorder rows that already tie on the tier,
    never override it -- the same additive posture that separated the coverage
    penalty (a proven win) from a fitted multi-feature calibrator (which lost).
    It also caps the result below 1.0, which src.generate_submission.
    validate_submission requires.

    Each optional argument being None means "this signal isn't available here",
    not "this signal says zero" -- with none of them, the confidence is the route
    tier alone. per_card_confidence is the per-card route's 3-sample
    self-consistency, which survives the switch to a greedy whole-image text
    source and is otherwise unused by it.
    """
    base = route_tier_score(field, text, per_card_text)
    extras: list[float] = []
    if locality_corpus is not None and field == "verbatimLocality":
        extras.append(corpus_support_score(text, locality_corpus))
    if local_vlm_agreement is not None:
        extras.append(local_vlm_agreement)
    if per_card_confidence is not None:
        extras.append(per_card_confidence)
    if not extras:
        return base
    return combine_tier_and_within_tier_score(base, sum(extras) / len(extras))
