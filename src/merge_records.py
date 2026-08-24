"""Fusiona los valores de varias tarjetas de un mismo especimen en un solo campo.

Une los textos con `" | "` (formato multi-tarjeta que pide la competencia) y
promedia las confianzas; devuelve `("MISSING", MISSING_CONFIDENCE)` si ninguna
tarjeta aporto valor.
"""

MISSING_CONFIDENCE = 0.5


def assemble_field(cards):
    if not cards:
        return "MISSING", MISSING_CONFIDENCE
    texts = [text for text, _ in cards]
    confidences = [conf for _, conf in cards]
    return " | ".join(texts), sum(confidences) / len(confidences)
