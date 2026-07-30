import re

ROMAN_MONTH = r"(?:X{1,2}I{0,3}|IX|IV|VI{0,3}|I{1,3})"
DATE_PATTERN = re.compile(
    rf"\b\d{{1,2}}[.\-/ ]({ROMAN_MONTH}|\d{{1,2}})[.\-/ ]\d{{2,4}}\b"
    rf"|\b({ROMAN_MONTH})[.\-/]\d{{2,4}}\b",
    re.IGNORECASE,
)
DANISH_MONTH_PATTERN = re.compile(
    r"\b(?:jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)\w*\.?\s+\d{1,4}\b",
    re.IGNORECASE,
)
BARE_YEAR_PATTERN = re.compile(r"^(1[7-9]\d{2}|20\d{2})$")
INITIAL_PATTERN = re.compile(r"^[a-zæøå]\.?$", re.IGNORECASE)
DIGIT_CHAR_PATTERN = re.compile(r"\d")
ALPHA_CHAR_PATTERN = re.compile(r"[a-zæøå]", re.IGNORECASE)

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
    if DATE_PATTERN.search(text.strip()) or DANISH_MONTH_PATTERN.search(t):
        return "date"
    if t.isdigit() and len(t) < 4:
        return "confound"
    digit_count = len(DIGIT_CHAR_PATTERN.findall(t))
    alpha_count = len(ALPHA_CHAR_PATTERN.findall(t))
    if digit_count >= 3 and digit_count >= 1.5 * alpha_count:
        # Line detection frequently mangles a date's punctuation into
        # spaces and duplicates digits (e.g. "22.5.1977" reconstructed as
        # "22 5 1 1"). It won't match DATE_PATTERN, but a noisy date guess
        # still beats letting it default to locality: NED ~0.44 vs 1.0 for
        # MISSING on real examples. The digit_count>=3 floor keeps this
        # from firing on short scraps ("1 1", "K 8") that are just as
        # likely to be stray OCR noise as a real date fragment.
        return "date"
    if digit_count > 0 and digit_count >= alpha_count:
        # Digit-dominant but below the confidence bar for a date guess --
        # still not a real locality word, so drop it rather than let it
        # contaminate the locality field.
        return "confound"
    return "locality"
