import csv
import json
import os

from src.locality_corpus import build_locality_corpus, corpus_support_score


def _write_predictions_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["crop_filename", "source_image_file", "verbatimDate", "verbatimDate_confidence",
                  "verbatimLocality", "verbatimLocality_confidence", "models_used", "raw_samples_json"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(crop, samples):
    return {
        "crop_filename": crop, "source_image_file": crop.split("_card")[0] + ".jpeg",
        "verbatimDate": samples[0]["verbatimDate"], "verbatimDate_confidence": "1.0",
        "verbatimLocality": samples[0]["verbatimLocality"], "verbatimLocality_confidence": "1.0",
        "models_used": "gemini-3.5-flash-lite", "raw_samples_json": json.dumps(samples),
    }


def test_build_locality_corpus_counts_every_raw_sample_not_just_modal(tmp_path):
    path = str(tmp_path / "preds.csv")
    _write_predictions_csv(path, [
        _row("a_card00.jpeg", [
            {"verbatimDate": "1.1.1900", "verbatimLocality": "Kb"},
            {"verbatimDate": "1.1.1900", "verbatimLocality": "Kb"},
            {"verbatimDate": "1.1.1900", "verbatimLocality": "Kbenhavn"},  # minority sample, still counted
        ]),
    ])
    corpus = build_locality_corpus(paths=[path])
    assert corpus["kb"] == 2
    assert corpus["kbenhavn"] == 1


def test_build_locality_corpus_skips_missing_and_splits_pipe_segments(tmp_path):
    path = str(tmp_path / "preds.csv")
    _write_predictions_csv(path, [
        _row("a_card00.jpeg", [
            {"verbatimDate": "1.1.1900", "verbatimLocality": "MISSING"},
            {"verbatimDate": "1.1.1900", "verbatimLocality": "Ti | Kulhuset"},
            {"verbatimDate": "1.1.1900", "verbatimLocality": "Ti | Kulhuset"},
        ]),
    ])
    corpus = build_locality_corpus(paths=[path])
    assert "missing" not in corpus
    assert corpus["ti"] == 2
    assert corpus["kulhuset"] == 2


def test_build_locality_corpus_ignores_missing_files(tmp_path):
    corpus = build_locality_corpus(paths=[str(tmp_path / "does_not_exist.csv")])
    assert corpus == {}


def test_corpus_support_score_zero_for_never_seen_string():
    corpus = {"kb": 40, "sjaelland": 20}
    assert corpus_support_score("Randomtown", corpus) == 0.0


def test_corpus_support_score_saturates_at_one():
    corpus = {"kb": 40}
    assert corpus_support_score("Kb", corpus, max_count_for_saturation=10) == 1.0


def test_corpus_support_score_scales_between_zero_and_saturation():
    corpus = {"kb": 5}
    assert corpus_support_score("Kb", corpus, max_count_for_saturation=10) == 0.5


def test_corpus_support_score_missing_text_is_neutral():
    assert corpus_support_score("MISSING", {"kb": 40}) == 0.5


def test_corpus_support_score_fuzzy_neighbour_gives_partial_credit():
    # "Sielland." (a plausible one-character misread of "Sjælland") should
    # get credit from the frequent neighbour, not score as a pure hapax.
    corpus = {"sjælland": 20}
    score = corpus_support_score("Sielland", corpus, max_count_for_saturation=10)
    assert score > 0.0
