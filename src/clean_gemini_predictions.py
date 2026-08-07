"""Post-processing text cleanup for Gemini verbatimLocality predictions, based
on consistent error patterns mined by src.analyze_gemini_errors from the dev-set
run: "o" is not a Danish letter but Gemini occasionally emits it for ø, and a
trailing period on a locality (unlike a date's punctuation) isn't normalized
by the scorer and so counts as a real edit-distance error.

verbatimDate is left untouched -- src.metrics.normalize_date_punctuation
already makes date punctuation variants free at scoring time, so nothing here
would change its AURC.

Deliberately narrow: only rules with clear, repeated support in the dev-set
error analysis. A broader idea (fuzzy-matching predictions against a gazetteer
of real Danish place names) was considered but not built -- ground truth
preserves historical/original spelling verbatim, so snapping to modern place
names risks *introducing* errors on legitimate historical spellings without a
real gazetteer and dev-set validation to bound that risk.
"""
import re

# ö isn't a Danish letter (Danish special letters are æ/ø/å) -- any ö Gemini
# emits is essentially always a misread ø.
_OE_UMLAUT_TO_OSLASH = str.maketrans("öÖ", "øØ")

_TRAILING_PUNCTUATION = re.compile(r"[.,]+\s*$")


def clean_locality(text):
    if text == "MISSING":
        return text
    text = text.translate(_OE_UMLAUT_TO_OSLASH)
    text = _TRAILING_PUNCTUATION.sub("", text).strip()
    return text or "MISSING"


def clean_merged_predictions(merged):
    """Applies clean_locality to every specimen's verbatimLocality prediction
    in a src.calibrate_gemini_confidence.merge_predictions_by_specimen-shaped
    dict, leaving verbatimDate and confidences untouched."""
    cleaned = {}
    for image_file, fields in merged.items():
        date_text, date_confidence = fields["verbatimDate"]
        locality_text, locality_confidence = fields["verbatimLocality"]
        cleaned[image_file] = {
            "verbatimDate": (date_text, date_confidence),
            "verbatimLocality": (clean_locality(locality_text), locality_confidence),
        }
    return cleaned
