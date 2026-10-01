"""Corpus-frequency support signal for verbatimLocality within-tie ranking
(src.confidence_rank_rules.combine_tier_and_within_tier_score): a locality
string seen many times across the whole prediction corpus is more likely a
real, correctly-read place name than a hapax, which is more likely a one-off
misread. Built entirely from the raw per-sample text already stored in
data/processed/gemini/*.csv's raw_samples_json columns -- zero new Gemini
calls, and every one of the ~15,000 already-paid samples counts, not just the
final modal prediction per crop/specimen.

The task demands verbatim transcription, so this is corpus support (what did
this pipeline actually observe elsewhere), never an external gazetteer --
src.clean_gemini_predictions' docstring already explains why snapping to a
gazetteer risks corrupting legitimate historical spellings ground truth
preserves verbatim.
"""
import collections
import csv
import json
import os
from collections.abc import Iterable, Mapping

from src.metrics import edit_distance

TEST_CARD_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "test_card_predictions.csv")
DEV_CARD_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "dev_card_predictions.csv")
TEST_WHOLEIMAGE_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "test_wholeimage_predictions.csv")
DEV_WHOLEIMAGE_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "dev_wholeimage_predictions.csv")

DEFAULT_SOURCES = (
    TEST_CARD_PREDICTIONS_PATH, DEV_CARD_PREDICTIONS_PATH,
    TEST_WHOLEIMAGE_PREDICTIONS_PATH, DEV_WHOLEIMAGE_PREDICTIONS_PATH,
)

# Fuzzy-neighbour lookup is only attempted for a hapax at least this long --
# below this, edit distance <= 2 matches nearly everything (e.g. "Kb" vs "Ti"
# is distance 2), which would inject noise into exactly the short, legitimate
# abbreviations the transcription prompt explicitly protects.
MIN_LENGTH_FOR_FUZZY_MATCH = 5
FUZZY_MAX_EDIT_DISTANCE = 2


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_locality_corpus(paths: Iterable[str] = DEFAULT_SOURCES,
                          field: str = "verbatimLocality") -> collections.Counter[str]:
    """Counter of normalized locality segments across every raw sample (not
    just the modal/final prediction) in the given prediction CSVs. A missing
    source file (a pass that hasn't been run, e.g. test_wholeimage before the
    fallback pass) is skipped rather than raising."""
    counts: collections.Counter[str] = collections.Counter()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                samples = json.loads(row["raw_samples_json"])
                for sample in samples:
                    text = sample.get(field, "MISSING")
                    if not text or text == "MISSING":
                        continue
                    for segment in text.split("|"):
                        norm = _normalize(segment)
                        if norm:
                            counts[norm] += 1
    return counts


def _fuzzy_support(norm: str, corpus: Mapping[str, int]) -> int:
    if len(norm) < MIN_LENGTH_FOR_FUZZY_MATCH:
        return 0
    best = 0
    for candidate, count in corpus.items():
        if abs(len(candidate) - len(norm)) > FUZZY_MAX_EDIT_DISTANCE:
            continue
        if edit_distance(norm, candidate) <= FUZZY_MAX_EDIT_DISTANCE:
            best = max(best, count)
    return best


def corpus_support_score(text: str, corpus: Mapping[str, int], max_count_for_saturation: int = 10) -> float:
    """[0,1]: 0 for a locality never seen (exactly or within edit distance 2)
    anywhere else in the corpus -- the single most common signature of a
    one-off misread -- rising to 1.0 once total corroborating evidence reaches
    max_count_for_saturation. "MISSING" scores neutral (0.5): this signal has
    nothing to say about a field the pipeline found no text for at all, and a
    neutral score in combine_tier_and_within_tier_score leaves such rows in
    their original within-tier position rather than pushing them to either
    extreme."""
    if text == "MISSING":
        return 0.5
    total = 0
    for segment in text.split("|"):
        norm = _normalize(segment)
        if not norm:
            continue
        exact = corpus.get(norm, 0)
        total += exact if exact > 0 else _fuzzy_support(norm, corpus)
    return min(1.0, total / max_count_for_saturation)
