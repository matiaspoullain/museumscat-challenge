"""Diagnostic report for src.confidence_rank_rules and any candidate within-tie
signal: how much of dev-set AURC is actually decided by tie-breaking (oracle vs.
anti-oracle vs. random reordering *within* existing confidence ties, no text
change at all), and whether a specific candidate signal's within-tier reordering
clears an adoption bar (a monotone risk gradient inside its tier, plus a paired-
bootstrap delta over the 200 dev specimens whose 90% interval excludes 0).

This is the harness that produced the numbers in
docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md's Context table.
Re-run it before adopting any new signal in src.confidence_rank_rules -- dev
(n=200) has disagreed with the real leaderboard twice before (see CLAUDE.md's
"Status (2026-08-20)"), so a signal only earns a place in production once its
risk table shows a real mechanism, not just a favorable dev-set roll.

Usage:
    conda run -n museumscat-challenge python -m src.analyze_rank_headroom
"""
import collections
import os
import random

import pandas as pd

from src.calibrate_gemini_confidence import (
    apply_wholeimage_fallback,
    load_predictions_by_crop,
    load_wholeimage_predictions,
    merge_predictions_by_specimen,
    rescored_predictions_by_crop,
)
from src.clean_gemini_predictions import clean_merged_predictions
from src.confidence_penalty_rules import apply_confidence_penalty, count_processed_and_contributing
from src.confidence_rank_rules import (
    STRUCTURAL_MISSING_CONFIDENCE,
    TIER_EPSILON,
    apply_structural_missing_tier,
    combine_tier_and_within_tier_score,
    date_plausibility_score,
)
from src.gemini_transcribe import soft_consistency_scores
from src.locality_corpus import build_locality_corpus, corpus_support_score
from src.metrics import aurc, best_pipe_ned
from src.pipeline import load_manifest

TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
DEV_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "dev_card_predictions.csv")
DEV_WHOLEIMAGE_PATH = os.path.join("data", "processed", "gemini", "dev_wholeimage_predictions.csv")

FIELDS = ("verbatimDate", "verbatimLocality")
FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}
N_RANDOM_DRAWS = 200
N_BOOTSTRAP = 2000
RANDOM_SEED = 0
PLACEMENT_SWEEP = [1.05, 1.0, 0.99, 0.95, 0.9, 0.85, 0.75, 0.6]


def build_dev_context(train_csv_path=TRAIN_CSV_PATH, manifest_path=MANIFEST_PATH,
                       predictions_path=DEV_PREDICTIONS_PATH, wholeimage_path=DEV_WHOLEIMAGE_PATH):
    """Reproduces the production dev merge (per-card -> clean -> whole-image
    fallback -> clean) and returns everything downstream analysis needs: text,
    errors, raw penalized confidence, and the (n_processed, n_contributing)
    pair every candidate structural signal is built from. Pure read of CSVs
    already on disk -- no Gemini API calls."""
    ground_truth_df = pd.read_csv(train_csv_path)
    manifest_by_image = load_manifest(manifest_path)
    predictions_by_crop = load_predictions_by_crop(predictions_path)
    wholeimage_predictions = load_wholeimage_predictions(wholeimage_path)
    image_files = ground_truth_df["image_file"].tolist()

    merged = merge_predictions_by_specimen(predictions_by_crop, manifest_by_image, image_files)
    merged = clean_merged_predictions(merged)
    merged = clean_merged_predictions(apply_wholeimage_fallback(merged, wholeimage_predictions))

    # Soft (edit-distance) consistency, re-derived offline from the same
    # raw_samples_json every card row already stores -- see
    # src.gemini_transcribe.soft_consistency_scores. Used only as a candidate
    # *within-tier* score below, never as the tier itself (that arm was
    # already tried and killed as a hard-consistency replacement, see
    # docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md).
    # Merged/cleaned/fallback-applied the same way as the hard-scored merge so
    # its confidence values are on the same specimen-level per-field footing;
    # its *text* should be identical to the hard merge (same modal-selection
    # logic, only the confidence formula differs).
    soft_predictions_by_crop = rescored_predictions_by_crop(predictions_path, soft_consistency_scores)
    soft_merged = merge_predictions_by_specimen(soft_predictions_by_crop, manifest_by_image, image_files)
    soft_merged = clean_merged_predictions(soft_merged)
    soft_merged = clean_merged_predictions(apply_wholeimage_fallback(soft_merged, wholeimage_predictions))

    locality_corpus = build_locality_corpus()

    by_field = {}
    for field, is_date in FIELD_IS_DATE.items():
        errors, penalized_confidences, coverage, texts, soft_confidences = [], [], [], [], []
        for _, row in ground_truth_df.iterrows():
            image_file = row["image_file"]
            text, confidence = merged[image_file][field]
            n_processed, n_contributing = count_processed_and_contributing(
                image_file, field, manifest_by_image, predictions_by_crop
            )
            penalized = apply_confidence_penalty(confidence, n_processed, n_contributing)
            errors.append(best_pipe_ned(text, str(row[field]), is_date=is_date))
            penalized_confidences.append(penalized)
            coverage.append((n_processed, n_contributing))
            texts.append(text)
            soft_confidences.append(soft_merged[image_file][field][1])
        by_field[field] = {
            "errors": errors, "confidences": penalized_confidences,
            "coverage": coverage, "texts": texts, "image_files": image_files,
            "soft_confidences": soft_confidences,
        }
    return {
        "ground_truth_df": ground_truth_df, "manifest_by_image": manifest_by_image,
        "predictions_by_crop": predictions_by_crop, "merged": merged, "by_field": by_field,
        "locality_corpus": locality_corpus,
    }


def _tie_groups(confidences):
    groups = collections.defaultdict(list)
    for i, c in enumerate(confidences):
        groups[round(c, 6)].append(i)
    return groups


def oracle_reorder(errors, confidences, ascending):
    """Reorders confidences within each existing tie group by error (ascending
    -> best case / oracle; not ascending -> worst case / anti-oracle), breaking
    each tie into synthetic sub-values via a tiny epsilon offset so aurc()'s
    stable sort respects the requested order without touching cross-tier
    ranking at all."""
    groups = _tie_groups(confidences)
    reordered = list(confidences)
    eps = 1e-6
    for key, idxs in groups.items():
        ordered = sorted(idxs, key=lambda i: errors[i], reverse=not ascending)
        m = len(ordered)
        for rank, i in enumerate(ordered):
            reordered[i] = key + eps * (m - rank)
    return reordered


def random_tiebreak_percentiles(errors, confidences, n_draws=N_RANDOM_DRAWS, seed=RANDOM_SEED):
    groups = _tie_groups(confidences)
    eps = 1e-6
    results = []
    for draw in range(n_draws):
        rng = random.Random(hash((seed, draw)))
        reordered = list(confidences)
        for key, idxs in groups.items():
            ordered = sorted(idxs, key=lambda i: rng.random())
            m = len(ordered)
            for rank, i in enumerate(ordered):
                reordered[i] = key + eps * (m - rank)
        results.append(aurc(errors, reordered))
    results.sort()
    return results


def paired_bootstrap_delta(errors, confidences_a, confidences_b, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Resamples specimens with replacement; on each resample scores confidences_b
    against confidences_a on the *same* resampled errors, so the two candidates
    are compared under identical noise. Returns the sorted list of
    (aurc_b - aurc_a) deltas -- a delta whose 5th/95th percentile straddles 0 is
    not distinguishable from noise on this 200-row dev set."""
    n = len(errors)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        e = [errors[i] for i in idx]
        ca = [confidences_a[i] for i in idx]
        cb = [confidences_b[i] for i in idx]
        deltas.append(aurc(e, cb) - aurc(e, ca))
    deltas.sort()
    return deltas


def _percentile(sorted_values, p):
    idx = min(len(sorted_values) - 1, max(0, round(p * (len(sorted_values) - 1))))
    return sorted_values[idx]


def print_headroom_report(by_field):
    print("=== tie-breaking headroom (oracle / anti-oracle / random, no text change) ===")
    for field, data in by_field.items():
        errors, confidences = data["errors"], data["confidences"]
        base = aurc(errors, confidences)
        oracle = aurc(errors, oracle_reorder(errors, confidences, ascending=True))
        anti = aurc(errors, oracle_reorder(errors, confidences, ascending=False))
        random_dist = random_tiebreak_percentiles(errors, confidences)
        p05, p50, p95 = _percentile(random_dist, 0.05), _percentile(random_dist, 0.5), _percentile(random_dist, 0.95)
        n_groups = len(_tie_groups(confidences))
        print(f"-- {field}: base={base:.4f}  oracle={oracle:.4f}  anti-oracle={anti:.4f}  "
              f"random[p05,p50,p95]=[{p05:.4f},{p50:.4f},{p95:.4f}]  ({n_groups} distinct confidence values)")


def apply_structural_missing(by_field, manifest_by_image, predictions_by_crop, ground_truth_df,
                              structural_missing_confidence=STRUCTURAL_MISSING_CONFIDENCE):
    """Returns a new by_field dict with the structural-missing tier applied on
    top of the base penalized confidences, plus a boolean mask per field for
    which rows were elevated (used by the risk table below)."""
    result = {}
    for field, data in by_field.items():
        elevated_confidences = []
        mask = []
        for text, (n_processed, n_contributing), base_conf in zip(
            data["texts"], data["coverage"], data["confidences"]
        ):
            new_conf = apply_structural_missing_tier(
                base_conf, text, n_processed, n_contributing, structural_missing_confidence
            )
            elevated_confidences.append(new_conf)
            mask.append(new_conf != base_conf)
        result[field] = dict(data, confidences=elevated_confidences, structural_missing_mask=mask)
    return result


def print_structural_missing_report(by_field, with_structural_missing):
    print("\n=== A1: structural-MISSING tier (src.confidence_rank_rules) ===")
    total_base, total_new = 0.0, 0.0
    for field in FIELDS:
        base = by_field[field]
        new = with_structural_missing[field]
        base_aurc = aurc(base["errors"], base["confidences"])
        new_aurc = aurc(new["errors"], new["confidences"])
        n_elevated = sum(new["structural_missing_mask"])
        elevated_errors = [e for e, flag in zip(new["errors"], new["structural_missing_mask"]) if flag]
        mean_elevated_ned = sum(elevated_errors) / len(elevated_errors) if elevated_errors else float("nan")
        deltas = paired_bootstrap_delta(base["errors"], base["confidences"], new["confidences"])
        lo, hi = _percentile(deltas, 0.05), _percentile(deltas, 0.95)
        print(f"-- {field}: {base_aurc:.4f} -> {new_aurc:.4f} (delta {new_aurc - base_aurc:+.4f}); "
              f"n_elevated={n_elevated} mean_ned_of_elevated={mean_elevated_ned:.4f}; "
              f"paired-bootstrap delta 90% CI=[{lo:+.4f},{hi:+.4f}]"
              f"{'  -- excludes 0, adoptable' if hi < 0 or lo > 0 else '  -- includes 0, NOT adoptable'}")
        total_base += base_aurc
        total_new += new_aurc
    print(f"-- overall: {total_base/2:.4f} -> {total_new/2:.4f} (delta {(total_new-total_base)/2:+.4f})")


def print_placement_sweep(by_field, manifest_by_image, predictions_by_crop, ground_truth_df):
    print("\n=== A1 placement sensitivity sweep (transparency only -- STRUCTURAL_MISSING_CONFIDENCE "
          f"is fixed at {STRUCTURAL_MISSING_CONFIDENCE}, not chosen by this sweep) ===")
    for value in PLACEMENT_SWEEP:
        elevated = apply_structural_missing(by_field, manifest_by_image, predictions_by_crop,
                                             ground_truth_df, structural_missing_confidence=value)
        total = sum(aurc(elevated[f]["errors"], elevated[f]["confidences"]) for f in FIELDS) / 2
        print(f"   value={value:<5} overall_aurc={total:.4f}")


def _within_tier_signal(field, data, locality_corpus):
    """{signal_name: [score, ...]} candidate within-tie scores, aligned to
    data["texts"]'s order. Each signal is neutral (0.5) on "MISSING" text by
    construction (date_plausibility_score and corpus_support_score both
    return 0.5 for it), so applying a signal uniformly across every row,
    including the structural-MISSING tier, can't spuriously reorder that
    tier -- it only ever changes ranking within a real-text tier."""
    signals = {"soft_consistency": list(data["soft_confidences"])}
    if field == "verbatimDate":
        signals["date_plausibility"] = [date_plausibility_score(t) for t in data["texts"]]
    else:
        signals["corpus_support"] = [corpus_support_score(t, locality_corpus) for t in data["texts"]]
    return signals


def _risk_by_within_tier_half(tier_confidences, within_scores, errors, min_group_size=6):
    """Checks EVERY tie group, not just the largest -- a signal can be
    structurally degenerate (zero variance) in the biggest tier while carrying
    real information in a smaller one. Example: soft consistency is identical
    to hard consistency (and so has zero within-tier variance) whenever hard
    consistency is already 1.0 (all 3 samples matched exactly), but varies
    freely within the 2-of-3 and 1-of-3 agreement tiers, which is exactly
    where it should matter -- checking only the top tier would have wrongly
    looked like no signal at all.

    Returns one (tier_key, n, mean_lo, mean_hi) row per eligible group (>=
    min_group_size rows, non-degenerate within_scores), sorted by n
    descending, so the report shows the full picture (every tier the signal
    could be tested in) rather than one cherry-picked tier -- a small tier's
    gap can easily point the "wrong" way from sampling noise alone (see e.g.
    corpus_support's 14-row 0.333 tier) without that being real counter-
    evidence against a signal whose larger tiers agree."""
    groups = _tie_groups(tier_confidences)
    rows = []
    for key, idxs in groups.items():
        if len(idxs) < min_group_size:
            continue
        scores = sorted(within_scores[i] for i in idxs)
        if scores[0] == scores[-1]:
            continue  # degenerate: no variance in this tier to test
        median = scores[len(scores) // 2]
        lo = [errors[i] for i in idxs if within_scores[i] < median]
        hi = [errors[i] for i in idxs if within_scores[i] >= median]
        if not lo or not hi:
            continue
        rows.append((key, len(idxs), sum(lo) / len(lo), sum(hi) / len(hi)))
    rows.sort(key=lambda r: -r[1])
    return rows


def print_signal_report(by_field, with_structural_missing, locality_corpus):
    print("\n=== A2: candidate within-tie signals (evaluated on top of the A1 structural-missing "
          "baseline) ===")
    for field in FIELDS:
        base = with_structural_missing[field]
        signals = _within_tier_signal(field, by_field[field], locality_corpus)
        for signal_name, scores in signals.items():
            gradient_rows = _risk_by_within_tier_half(base["confidences"], scores, base["errors"])
            combined = [
                combine_tier_and_within_tier_score(c, s) for c, s in zip(base["confidences"], scores)
            ]
            base_aurc = aurc(base["errors"], base["confidences"])
            new_aurc = aurc(base["errors"], combined)
            deltas = paired_bootstrap_delta(base["errors"], base["confidences"], combined)
            lo_ci, hi_ci = _percentile(deltas, 0.05), _percentile(deltas, 0.95)
            adoptable = hi_ci < 0 or lo_ci > 0
            print(f"-- {field} / {signal_name}:")
            if not gradient_rows:
                print("   no eligible tier to test a gradient in")
            for tier_key, n, mean_lo, mean_hi in gradient_rows:
                direction = "monotone (good)" if mean_hi <= mean_lo else "inverted"
                print(f"   tier={tier_key:<9} n={n:<4} lower-half mean_ned={mean_lo:.4f} "
                      f"upper-half mean_ned={mean_hi:.4f}  ({direction})")
            print(f"   aurc {base_aurc:.4f} -> {new_aurc:.4f} (delta {new_aurc - base_aurc:+.4f}); "
                  f"paired-bootstrap 90% CI=[{lo_ci:+.4f},{hi_ci:+.4f}]"
                  f"{'  -- ADOPT' if adoptable and new_aurc < base_aurc else '  -- reject'}")


def main():
    context = build_dev_context()
    by_field = context["by_field"]

    print_headroom_report(by_field)

    with_structural_missing = apply_structural_missing(
        by_field, context["manifest_by_image"], context["predictions_by_crop"], context["ground_truth_df"]
    )
    print_structural_missing_report(by_field, with_structural_missing)
    print_placement_sweep(by_field, context["manifest_by_image"], context["predictions_by_crop"],
                           context["ground_truth_df"])
    print_signal_report(by_field, with_structural_missing, context["locality_corpus"])


if __name__ == "__main__":
    main()
