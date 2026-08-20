from src.analyze_rank_headroom import (
    _risk_by_within_tier_half,
    apply_structural_missing,
    oracle_reorder,
    paired_bootstrap_delta,
    random_tiebreak_percentiles,
)
from src.metrics import aurc


def test_oracle_reorder_ascending_puts_lowest_error_first_within_a_tie():
    # All three rows tied at confidence 1.0 -- oracle (ascending) should rank
    # the zero-error row first, pushing the two errors to lower coverage.
    errors = [0.0, 0.5, 1.0]
    confidences = [1.0, 1.0, 1.0]
    reordered = oracle_reorder(errors, confidences, ascending=True)
    order = sorted(range(3), key=lambda i: -reordered[i])
    assert [errors[i] for i in order] == [0.0, 0.5, 1.0]


def test_oracle_reorder_anti_ascending_puts_highest_error_first():
    errors = [0.0, 0.5, 1.0]
    confidences = [1.0, 1.0, 1.0]
    reordered = oracle_reorder(errors, confidences, ascending=False)
    order = sorted(range(3), key=lambda i: -reordered[i])
    assert [errors[i] for i in order] == [1.0, 0.5, 0.0]


def test_oracle_reorder_never_moves_a_row_out_of_its_tie_group():
    # Two tie groups (1.0 and 0.5); oracle reordering must keep every row's
    # tier, only reorder within it -- so the resulting AURC must lie between
    # the untouched-order AURC's best and worst case for each group alone.
    errors = [0.9, 0.1, 0.9, 0.1]
    confidences = [1.0, 1.0, 0.5, 0.5]
    reordered = oracle_reorder(errors, confidences, ascending=True)
    # rows tied at 1.0 must both still exceed rows tied at 0.5 after reordering
    assert min(reordered[0], reordered[1]) > max(reordered[2], reordered[3])


def test_oracle_reorder_improves_or_matches_aurc_vs_original():
    errors = [0.9, 0.1, 0.9, 0.1]
    confidences = [1.0, 1.0, 0.5, 0.5]
    base = aurc(errors, confidences)
    oracle = aurc(errors, oracle_reorder(errors, confidences, ascending=True))
    assert oracle <= base


def test_random_tiebreak_percentiles_returns_sorted_list_of_requested_length():
    errors = [0.1, 0.9, 0.3, 0.7]
    confidences = [1.0, 1.0, 1.0, 1.0]
    result = random_tiebreak_percentiles(errors, confidences, n_draws=20, seed=1)
    assert len(result) == 20
    assert result == sorted(result)


def test_random_tiebreak_percentiles_deterministic_given_seed():
    errors = [0.1, 0.9, 0.3, 0.7]
    confidences = [1.0, 1.0, 1.0, 1.0]
    a = random_tiebreak_percentiles(errors, confidences, n_draws=10, seed=42)
    b = random_tiebreak_percentiles(errors, confidences, n_draws=10, seed=42)
    assert a == b


def test_paired_bootstrap_delta_is_zero_when_candidate_equals_baseline():
    errors = [0.1, 0.9, 0.3, 0.7, 0.5]
    confidences = [1.0, 0.5, 0.7, 0.3, 0.9]
    deltas = paired_bootstrap_delta(errors, confidences, confidences, n_boot=50, seed=0)
    assert all(d == 0.0 for d in deltas)


def test_paired_bootstrap_delta_negative_when_candidate_strictly_better():
    # candidate ranks by true error (oracle-like) -- should show a clear
    # negative (improving) delta against a poorly-ordered baseline.
    errors = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
    baseline_confidences = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # all tied -- arbitrary order
    candidate_confidences = [1 - e for e in errors]  # perfectly ranks by error
    deltas = paired_bootstrap_delta(errors, baseline_confidences, candidate_confidences, n_boot=200, seed=0)
    assert sum(deltas) / len(deltas) < 0


def test_risk_by_within_tier_half_skips_degenerate_tiers():
    # tier "1.0" has zero within-score variance (all scores equal) -- must be
    # skipped entirely, not reported as a false "no gradient" result.
    confidences = [1.0] * 6 + [0.7] * 6
    within_scores = [0.5] * 6 + [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    errors = [0.0] * 6 + [0.8, 0.8, 0.8, 0.1, 0.1, 0.1]
    rows = _risk_by_within_tier_half(confidences, within_scores, errors, min_group_size=6)
    tiers_reported = [r[0] for r in rows]
    assert 1.0 not in tiers_reported
    assert 0.7 in tiers_reported


def test_risk_by_within_tier_half_reports_correct_direction_for_real_signal():
    confidences = [0.7] * 6
    within_scores = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    errors = [0.8, 0.8, 0.8, 0.1, 0.1, 0.1]  # higher score -> lower error
    rows = _risk_by_within_tier_half(confidences, within_scores, errors, min_group_size=6)
    (_, n, mean_lo, mean_hi) = rows[0]
    assert n == 6
    assert mean_hi < mean_lo


def test_risk_by_within_tier_half_excludes_tiers_below_min_group_size():
    confidences = [1.0, 1.0, 0.7, 0.7, 0.7]
    within_scores = [0.1, 0.9, 0.2, 0.5, 0.8]
    errors = [0.5, 0.0, 0.5, 0.3, 0.0]
    rows = _risk_by_within_tier_half(confidences, within_scores, errors, min_group_size=6)
    assert rows == []


def test_apply_structural_missing_elevates_only_qualifying_rows():
    by_field = {
        "verbatimDate": {
            "errors": [0.0, 0.0],
            "confidences": [0.5, 0.35],
            "coverage": [(0, 0), (2, 0)],  # first qualifies, second is coverage-flagged (already 0.35)
            "texts": ["MISSING", "MISSING"],
            "image_files": ["a.jpeg", "b.jpeg"],
        }
    }
    result = apply_structural_missing(by_field, {}, {}, None, structural_missing_confidence=0.95)
    assert result["verbatimDate"]["confidences"] == [0.95, 0.35]
    assert result["verbatimDate"]["structural_missing_mask"] == [True, False]
