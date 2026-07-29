import re

ROMAN_MONTH = r"(?:X{1,2}I{0,3}|IX|IV|VI{0,3}|I{1,3})"
DATE_PATTERN = re.compile(
    rf"\b\d{{1,2}}[.\-/ ]({ROMAN_MONTH}|\d{{1,2}})[.\-/ ]\d{{2,4}}\b"
    rf"|\b({ROMAN_MONTH})[.\-/]\d{{2,4}}\b",
    re.IGNORECASE,
)
INITIAL_PATTERN = re.compile(r"^[a-zæøå]\.?$", re.IGNORECASE)

CONFOUND_KEYWORDS = [
    "coll.", "coll ", "det.", "det ", "tilg.", "dania",
    "statens naturhistoriske museum", "kobenhavns universitet",
    "dassco", "nhmd", "mus.",
]


def classify_field(text):
    """A bare single letter (with or without a trailing period, e.g. "J.")
    is treated as confound: it's either a collector's initial or OCR
    noise, never a meaningful date/locality on its own."""
    t = text.strip().lower()
    if any(kw in t for kw in CONFOUND_KEYWORDS):
        return "confound"
    if INITIAL_PATTERN.match(t):
        return "confound"
    if DATE_PATTERN.search(text.strip()):
        return "date"
    return "locality"
