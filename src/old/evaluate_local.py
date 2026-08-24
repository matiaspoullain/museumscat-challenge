"""Evalua el pipeline clasico (OCR + reglas) contra las 200 filas etiquetadas.

Reporta NED por campo y AURC usando `src.metrics`, sobre las predicciones de
palabras ya generadas. Superado por `src.evaluate_gemini_dev`.
"""

import os

import pandas as pd

from src.card_relevance_visual import compute_dev_relevance_flags
from src.metrics import aurc, best_pipe_ned
from src.pipeline import load_manifest, load_predictions, run_pipeline

FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}


def score_predictions(predicted, ground_truth_df):
    report = {}
    for column, is_date in FIELD_IS_DATE.items():
        errors, confidences = [], []
        for _, row in ground_truth_df.iterrows():
            text, confidence = predicted[row["image_file"]][column]
            errors.append(best_pipe_ned(text, str(row[column]), is_date=is_date))
            confidences.append(confidence)
        report[column] = {
            "mean_ned": sum(errors) / len(errors),
            "aurc": aurc(errors, confidences),
        }
    return report


def _print_report(label, report):
    print(f"--- {label} ---")
    for column, scores in report.items():
        print(f"{column}: mean_ned={scores['mean_ned']:.3f} aurc={scores['aurc']:.3f}")
    overall_aurc = sum(scores["aurc"] for scores in report.values()) / len(report)
    print(f"overall AURC (mean across fields, matches competition scoring): {overall_aurc:.3f}")
    return overall_aurc


def main():
    manifest_path = os.path.join("data", "processed", "card_crops", "manifest.csv")
    predictions_path = os.path.join("data", "processed", "card_crops", "predictions.csv")
    train_path = os.path.join("data", "raw", "train.csv")

    ground_truth_df = pd.read_csv(train_path)
    manifest_by_image = load_manifest(manifest_path)
    lines_by_crop = load_predictions(predictions_path)
    image_files = ground_truth_df["image_file"].tolist()

    predicted_baseline = run_pipeline(image_files, manifest_by_image, lines_by_crop)
    report_baseline = score_predictions(predicted_baseline, ground_truth_df)
    aurc_baseline = _print_report("baseline (no card-level relevance filter)", report_baseline)

    dev_specimens = {os.path.splitext(f)[0] for f in ground_truth_df["image_file"]}
    relevance_flags = compute_dev_relevance_flags(dev_specimens)
    is_relevant = lambda crop_filename: relevance_flags.get(crop_filename, True)
    predicted_visual = run_pipeline(image_files, manifest_by_image, lines_by_crop, is_relevant=is_relevant)
    report_visual = score_predictions(predicted_visual, ground_truth_df)
    aurc_visual = _print_report("with visual card-relevance filter", report_visual)

    print(f"overall AURC delta (visual - baseline): {aurc_visual - aurc_baseline:+.3f}")


if __name__ == "__main__":
    main()
