"""Mines character-level substitution/insertion/deletion patterns from the
Gemini dev-set predictions' errors, to find consistent, correctable mistakes
(e.g. one diacritic swapped for another) rather than one-off noise -- input
for deciding what rules belong in src.clean_gemini_predictions.

Usage:
    conda run -n museumscat-challenge python -m src.analyze_gemini_errors
"""
import collections
import difflib
import os

import pandas as pd

from src.calibrate_gemini_confidence import load_predictions_by_crop, merge_predictions_by_specimen
from src.pipeline import load_manifest

TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")

FIELDS = ("verbatimDate", "verbatimLocality")


def pair_segments(pred, truth):
    """Split both pred/truth on '|' and pair them positionally -- an
    approximation of best_pipe_ned's permutation matching, good enough for
    mining common substitution patterns (not for scoring)."""
    pred_segments = [s.strip() for s in pred.split("|")]
    truth_segments = [s.strip() for s in truth.split("|")]
    if len(pred_segments) != len(truth_segments):
        return [(pred.strip(), truth.strip())]
    return list(zip(pred_segments, truth_segments))


def diff_ops(pred, truth):
    """Non-'equal' opcodes from a SequenceMatcher alignment between the
    lowercased pred/truth segment text: (tag, pred_substring, truth_substring)."""
    matcher = difflib.SequenceMatcher(None, pred.lower(), truth.lower())
    ops = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            ops.append((tag, pred.lower()[i1:i2], truth.lower()[j1:j2]))
    return ops


def collect_diff_counts(merged, ground_truth_df):
    counts = {field: collections.Counter() for field in FIELDS}
    examples = {field: collections.defaultdict(list) for field in FIELDS}
    for _, row in ground_truth_df.iterrows():
        for field in FIELDS:
            pred_text, _ = merged[row["image_file"]][field]
            truth_text = str(row[field])
            if pred_text == "MISSING" or truth_text == "MISSING":
                continue
            for pred_seg, truth_seg in pair_segments(pred_text, truth_text):
                if pred_seg == truth_seg:
                    continue
                for tag, p, t in diff_ops(pred_seg, truth_seg):
                    key = (tag, p, t)
                    counts[field][key] += 1
                    if len(examples[field][key]) < 3:
                        examples[field][key].append((row["image_file"], pred_seg, truth_seg))
    return counts, examples


def main():
    ground_truth_df = pd.read_csv(TRAIN_CSV_PATH)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    predictions_by_crop = load_predictions_by_crop()
    merged = merge_predictions_by_specimen(
        predictions_by_crop, manifest_by_image, ground_truth_df["image_file"].tolist()
    )

    counts, examples = collect_diff_counts(merged, ground_truth_df)
    for field in FIELDS:
        print(f"--- {field}: top diff patterns ---")
        for (tag, p, t), n in counts[field].most_common(25):
            print(f"  {n:3d}x  {tag:8s}  {p!r:>15} -> {t!r:<15}  e.g. {examples[field][(tag, p, t)]}")


if __name__ == "__main__":
    main()
