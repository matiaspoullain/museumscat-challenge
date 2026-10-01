"""Confidence *ranking* refinements, distinct from src.confidence_penalty_rules'
magnitude penalty. AURC (src.metrics.aurc) depends only on confidence rank order
-- see docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md's
"key finding" -- so with only n_samples+1 possible raw self-consistency values,
roughly half of all submission rows sit in one tie block at confidence 1.0, and
Python's stable sort resolves ties by arbitrary CSV row order. Measured on the
dev set: oracle reordering *within existing ties* (no text change at all) takes
overall AURC 0.0715 -> 0.0456, and the current tie order is statistically
indistinguishable from random (see docs/superpowers/plans/2026-08-2*-confidence-
rank-refinement.md). This module gives that ordering an explicit, auditable
structure instead of leaving it to sort() stability.

Independent tools:

1. is_confident_structural_missing / apply_structural_missing_tier -- src.
   merge_records.MISSING_CONFIDENCE (0.5) is an unjustified magic number for
   *every* MISSING prediction. But "0 or 1 card crops were Gemini-processed for
   this specimen, and none of them (nor the whole-image fallback) found a value"
   is a genuinely low-ambiguity structural signal -- on dev, this population's
   mean NED for verbatimDate (0.018) is *better* than the confidence=1.0 text
   tier's (0.021), yet it was ranked below several worse-performing text tiers
   (0.667, 0.7) purely because 0.5 < those tiers' confidence values. Deliberately
   narrower than "text == MISSING": >=2 processed crops that all read MISSING is
   exactly src.confidence_penalty_rules' own under-coverage risk pattern (a
   second card's real value may have been missed) and is intentionally excluded
   here, left to that existing penalty instead of elevated.

2. combine_tier_and_within_tier_score -- lets a secondary, finer-grained signal
   (soft consistency, visual crop quality, date plausibility, corpus support,
   ...) break ties *inside* a confidence tier without ever letting that signal
   cross into an adjacent tier's range, so any adopted signal can only reorder
   already-tied rows, never override the coarse self-consistency ranking itself
   -- the same "additive, not a global refit" posture that separates the
   coverage penalty (a proven win) from a fitted multi-feature calibrator (which
   loses to raw confidence on the mean across seeds, see the calibration plan's
   Outcome section).

3. date_plausibility_score -- a candidate within-tier signal for verbatimDate:
   does a prediction parse as a real calendar date? Targets the catalog-
   number-as-date failure the closed local-VLM spike documented ("608.",
   "2704" emitted as dates) -- a confound self-consistency can't see, since
   Gemini can be unanimously wrong across all 3 samples about a catalog
   number being a date. EVALUATED AND REJECTED by src.old.analyze_rank_headroom
   (docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md): inverted
   risk gradient in the largest (confidence=1.0) tier and a net-negative AURC
   effect. Kept here, unused by within_tier_score, as a documented dead end so
   it isn't re-proposed.

4. within_tier_score -- the shipped combiner. Per-field, per
   src.old.analyze_rank_headroom's evaluation (dev AURC 0.0715 -> 0.0575 combined
   with the structural-missing tier above): soft self-consistency alone for
   verbatimDate (date_plausibility_score was tried and rejected, see above),
   and the unweighted average of soft consistency and
   src.locality_corpus.corpus_support_score for verbatimLocality (beats
   corpus support alone on a paired bootstrap: 90% CI [-0.0012,-0.0001]).
   local_vlm_agreement (src.local_vlm_signal.local_vlm_agreement_score), when
   given, is blended in with equal weight on top of that: dev AURC
   0.0575 -> 0.0493 (date bootstrap 90% CI [-0.0089,-0.0009], clean; locality
   [-0.0285,+0.0054], directionally strong but not as clean -- partial
   152/197 dev crop coverage dilutes both the effect and the bootstrap's
   power; see the plan doc's Track A3 section).
"""
from collections.abc import Mapping

from src.locality_corpus import corpus_support_score
from src.metrics import normalize_date_punctuation

# Placed strictly below the confidence=1.0 unanimous-agreement text tier and
# above every other production confidence tier (0.7-and-below) -- "found and
# confirmed unanimously across 3 samples" is still slightly stronger evidence
# than "checked <2 cards and found nothing", but the latter still clearly beats
# every majority/minority-agreement text tier it was previously ranked below.
# A fixed, mechanistically-justified placement (not swept to fit dev noise);
# src.old.analyze_rank_headroom reports a placement sensitivity sweep for
# transparency, but this constant is the shipped value regardless of which
# exact number that sweep happens to prefer on 200 dev rows.
STRUCTURAL_MISSING_CONFIDENCE = 0.95

# Must stay well under the smallest gap between adjacent real production
# confidence tiers (~0.033, e.g. 0.667 vs 0.7) so a within-tier score can never
# reorder rows across tiers, only within one.
TIER_EPSILON = 0.01


def is_confident_structural_missing(text: str, n_processed: int, n_contributing: int) -> bool:
    """True when a field's MISSING prediction reflects low-ambiguity structural
    absence: <2 card crops were Gemini-processed for this specimen (so there's no
    second-card-coverage risk to flag), and none of them found a value for the
    field. n_contributing == 0 is implied whenever text == "MISSING" (see
    src.calibrate_gemini_confidence.merge_predictions_by_specimen), but is taken
    as an explicit parameter so this function's contract doesn't depend on that
    invariant holding in every caller."""
    return text == "MISSING" and n_processed < 2 and n_contributing == 0


def apply_structural_missing_tier(confidence: float, text: str, n_processed: int, n_contributing: int,
                                  structural_missing_confidence: float = STRUCTURAL_MISSING_CONFIDENCE,
                                  ) -> float:
    """Overrides confidence with structural_missing_confidence when
    is_confident_structural_missing holds, else returns confidence unchanged.
    Order relative to src.confidence_penalty_rules.apply_confidence_penalty
    doesn't matter: the two conditions are mutually exclusive by construction
    (the penalty's flagged condition requires n_processed >= 2, which
    is_confident_structural_missing explicitly excludes)."""
    if is_confident_structural_missing(text, n_processed, n_contributing):
        return structural_missing_confidence
    return confidence


def combine_tier_and_within_tier_score(tier_confidence: float, within_tier_score: float) -> float:
    """base + TIER_EPSILON * clip(within_tier_score, 0, 1), where base is
    tier_confidence capped at 1.0 - TIER_EPSILON. Monotonic in
    within_tier_score, and bounded so the result never reaches
    tier_confidence + the gap to the next tier up -- a within-tier signal can
    only ever reorder rows that were already tied on tier_confidence.

    The cap matters only for the top tier (raw confidence == 1.0, no tier
    above it to leave headroom in) -- roughly half of all submission rows
    live there, and it's exactly where corpus_support_score showed its
    strongest dev-set signal (n=87, mean NED 0.132 in the lower half vs.
    0.061 in the upper half), so this tier needs the same in-tier ordering
    room every other tier gets. Without the cap, tier_confidence=1.0 would
    produce values above 1.0, which src.generate_submission.
    validate_submission correctly rejects -- caught by regenerating the real
    submission (see docs/superpowers/plans/2026-08-20-confidence-rank-
    refinement.md), not by the pre-cap unit tests, since aurc() itself only
    depends on rank order and doesn't validate the [0,1] range."""
    clipped = min(1.0, max(0.0, within_tier_score))
    base = min(tier_confidence, 1.0 - TIER_EPSILON)
    return base + TIER_EPSILON * clipped


# 2-digit years below this are read as 19xx, at/above it as 18xx -- a rough
# heuristic (Danish dung-beetle specimen collection spans ~1800-present), not
# meant to be exact: this only feeds a plausibility check, not the emitted text.
_TWO_DIGIT_YEAR_CUTOFF = 30
_MIN_PLAUSIBLE_YEAR = 1700
_MAX_PLAUSIBLE_YEAR = 2026


def date_plausibility_score(text: str) -> float:
    """Heuristic in [0,1]: 1.0 for a string that parses as a real calendar
    date (day<=31, month<=12, a plausible year), 0.0 for a bare number with no
    date-like separators at all -- the catalog-number-as-date failure mode --
    or for a day/month/year combination out of range, 0.5 for anything else
    (a textual month name, an unrecognized shape) -- deliberately not
    penalized, since not matching this heuristic's narrow numeric-date shape
    isn't evidence of implausibility, just of not being checkable by it."""
    normalized = normalize_date_punctuation(text)
    parts = [p for p in normalized.split(".") if p]
    if not parts:
        return 0.5
    if len(parts) == 1:
        return 0.0 if parts[0].isdigit() else 0.5
    if len(parts) not in (2, 3):
        return 0.5
    if not all(p.isdigit() for p in parts):
        return 0.5  # contains a non-numeric token (a month name, etc.)

    numbers = [int(p) for p in parts]
    if len(numbers) == 3:
        day, month, year = numbers
    else:
        day, (month, year) = 1, numbers

    if year < 100:
        year += 1800 if year >= _TWO_DIGIT_YEAR_CUTOFF else 1900
    plausible = 1 <= day <= 31 and 1 <= month <= 12 and _MIN_PLAUSIBLE_YEAR <= year <= _MAX_PLAUSIBLE_YEAR
    return 1.0 if plausible else 0.0


def within_tier_score(field: str, text: str, soft_confidence: float,
                      locality_corpus: Mapping[str, int] | None = None,
                      local_vlm_agreement: float | None = None) -> float:
    """The shipped within-tier combiner (see module docstring point 4).
    soft_confidence is this specimen's soft (edit-distance) self-consistency
    for `field` -- src.gemini_transcribe.soft_consistency_scores merged the
    same way as production's hard self-consistency (see
    src.generate_submission_gemini). locality_corpus (src.locality_corpus.
    build_locality_corpus's output) is only consulted for verbatimLocality;
    None falls back to soft_confidence alone (equivalent to treating
    corpus_support_score as perfectly neutral), so this still degrades
    gracefully if a caller doesn't have a corpus built. local_vlm_agreement
    (src.local_vlm_signal.local_vlm_agreement_score's output), when not None,
    is blended in with equal weight on top of the field-specific base score;
    None (the default -- no local VLM run available for this split yet)
    leaves behavior identical to not having this parameter at all."""
    if field != "verbatimLocality" or locality_corpus is None:
        base = soft_confidence
    else:
        base = (soft_confidence + corpus_support_score(text, locality_corpus)) / 2
    if local_vlm_agreement is None:
        return base
    return (base + local_vlm_agreement) / 2
