import itertools
import re


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n]


def normalize_date_punctuation(s):
    return re.sub(r"[.,\-·\s]+", ".", s.strip()).strip(".")


def ned(pred, truth, is_date=False):
    p, t = pred.strip().lower(), truth.strip().lower()
    if is_date:
        p, t = normalize_date_punctuation(p), normalize_date_punctuation(t)
    if p == t:
        return 0.0
    denom = max(len(p), len(t), 1)
    return edit_distance(p, t) / denom


def best_pipe_ned(pred, truth, is_date=False):
    pred_cards = [c.strip() for c in pred.split("|")]
    truth_cards = [c.strip() for c in truth.split("|")]
    if len(pred_cards) != len(truth_cards):
        # Approximation for mismatched card counts: compare the joined
        # strings directly. The official scorer's behavior in this case
        # isn't documented; this only needs to be directionally useful
        # for comparing local pipeline iterations, not to match the
        # leaderboard exactly.
        return ned(" ".join(pred_cards), " ".join(truth_cards), is_date)
    return min(
        ned(" ".join(perm), " ".join(truth_cards), is_date)
        for perm in itertools.permutations(pred_cards)
    )


def aurc(errors, confidences):
    n = len(errors)
    order = sorted(range(n), key=lambda i: -confidences[i])
    sorted_errors = [errors[i] for i in order]
    cum = 0.0
    risks = []
    for i, e in enumerate(sorted_errors, start=1):
        cum += e
        risks.append(cum / i)
    coverages = [i / n for i in range(1, n + 1)]
    area = 0.0
    for i in range(1, len(coverages)):
        area += (coverages[i] - coverages[i - 1]) * (risks[i] + risks[i - 1]) / 2
    return area
