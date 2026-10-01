"""Merge one specimen's per-card values into a single field value.

Joins texts with `" | "` (the multi-card format the competition specifies) and
averages the confidences; returns `("MISSING", MISSING_CONFIDENCE)` when no card
contributed a value.
"""

MISSING_CONFIDENCE = 0.5

# One card's (text, confidence) for a single field.
CardValue = tuple[str, float]
# {field: (text, confidence)} for one card or one specimen.
FieldScores = dict[str, CardValue]


def assemble_field(cards: list[CardValue]) -> CardValue:
    if not cards:
        return "MISSING", MISSING_CONFIDENCE
    texts = [text for text, _ in cards]
    confidences = [conf for _, conf in cards]
    return " | ".join(texts), sum(confidences) / len(confidences)
