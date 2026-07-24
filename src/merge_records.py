MISSING_CONFIDENCE = 0.5


def assemble_field(cards):
    if not cards:
        return "MISSING", MISSING_CONFIDENCE
    texts = [text for text, _ in cards]
    confidences = [conf for _, conf in cards]
    return " | ".join(texts), sum(confidences) / len(confidences)
