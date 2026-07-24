import re

ROMAN_MONTH = r"(?:X{1,2}I{0,3}|IX|IV|VI{0,3}|I{1,3})"
DATE_PATTERN = re.compile(
    rf"\b\d{{1,2}}[.\-/ ]({ROMAN_MONTH}|\d{{1,2}})[.\-/ ]\d{{2,4}}\b"
    rf"|\b({ROMAN_MONTH})[.\-/]\d{{2,4}}\b",
    re.IGNORECASE,
)

CONFOUND_KEYWORDS = [
    "coll.", "coll ", "det.", "det ", "tilg.", "dania",
    "statens naturhistoriske museum", "kobenhavns universitet",
    "dassco", "nhmd", "mus.",
]


def classify_field(text):
    t = text.strip().lower()
    if any(kw in t for kw in CONFOUND_KEYWORDS):
        return "confound"
    if DATE_PATTERN.search(text.strip()):
        return "date"
    return "locality"
