import os

import pandas as pd

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


def main():
    manifest_path = os.path.join("data", "processed", "card_crops", "manifest.csv")
    predictions_path = os.path.join("data", "processed", "card_crops", "predictions.csv")
    train_path = os.path.join("data", "raw", "train.csv")

    ground_truth_df = pd.read_csv(train_path)
    manifest_by_image = load_manifest(manifest_path)
    lines_by_crop = load_predictions(predictions_path)
    predicted = run_pipeline(ground_truth_df["image_file"].tolist(), manifest_by_image, lines_by_crop)

    report = score_predictions(predicted, ground_truth_df)
    for column, scores in report.items():
        print(f"{column}: mean_ned={scores['mean_ned']:.3f} aurc={scores['aurc']:.3f}")
    overall_aurc = sum(scores["aurc"] for scores in report.values()) / len(report)
    print(f"overall AURC (mean across fields, matches competition scoring): {overall_aurc:.3f}")


if __name__ == "__main__":
    main()
