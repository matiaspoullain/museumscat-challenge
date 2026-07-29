import re

ROMAN_MONTH = r"(?:X{1,2}I{0,3}|IX|IV|VI{0,3}|I{1,3})"
DATE_PATTERN = re.compile(
    rf"\b\d{{1,2}}[.\-/ ]({ROMAN_MONTH}|\d{{1,2}})[.\-/ ]\d{{2,4}}\b"
    rf"|\b({ROMAN_MONTH})[.\-/]\d{{2,4}}\b",
    re.IGNORECASE,
)
BARE_YEAR_PATTERN = re.compile(r"^(1[7-9]\d{2}|20\d{2})$")
INITIAL_PATTERN = re.compile(r"^[a-zæøå]\.?$", re.IGNORECASE)

CONFOUND_KEYWORDS = [
    "coll", "loll", "løll", "oll.", "0ll",  # "loll"/"løll"/"oll."/"0ll" are doctr's
    # recurring C/L/0 misread of "coll." on these specimen labels
    "det.", "det ", "tilg.", "dania", "ania", "anna", "danva",  # last three are
    # recurring OCR misreads of "Dania"
    "statens naturhistoriske museum", "kobenhavns universitet",
    "dassco", "nhmd", "mus.",
    # recurring collector/determiner names seen in the labeled dev set --
    # a keyword blacklist, not a name-detector: known to miss unseen surnames
    "hansen", "andersen", "rosenberg", "gønget", "gonget", "bangsholt",
    "bramsholm", "johs", "hans", "schiødte", "srhiodte", "sehiodte", "sehiodee",
]


def classify_field(text):
    """A bare single letter (with or without a trailing period, e.g. "J.")
    is treated as confound: it's either a collector's initial or OCR
    noise, never a meaningful date/locality on its own. Likewise a short
    (1-3 digit) number without date punctuation is catalog/page noise, not
    a real date or locality -- but a bare 4-digit year (e.g. "1994") is a
    legitimate date value on its own."""
    t = text.strip().lower()
    if not t:
        return "confound"
    if any(kw in t for kw in CONFOUND_KEYWORDS):
        return "confound"
    if INITIAL_PATTERN.match(t):
        return "confound"
    if BARE_YEAR_PATTERN.match(t):
        return "date"
    if DATE_PATTERN.search(text.strip()):
        return "date"
    if t.isdigit() and len(t) < 4:
        return "confound"
    return "locality"
