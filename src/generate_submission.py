"""Build and validate the classical pipeline's submission CSV.

`validate_submission` (exact column order, confidence ranges, no nulls) is still
on the active path: src.generate_submission_gemini reuses it so both the
classical and Gemini submissions pass the identical format checks.
"""

import os

import pandas as pd

from src.card_relevance_visual import compute_relevance_flags_for_specimens
from src.pipeline import load_manifest, load_predictions, run_pipeline

REQUIRED_COLUMNS = [
    "image_file",
    "verbatimDate",
    "verbatimDate_confidence",
    "verbatimLocality",
    "verbatimLocality_confidence",
]


def validate_submission(df, expected_image_files):
    errors = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing column(s): {missing_cols}")
        return errors
    for col in ["verbatimDate", "verbatimLocality"]:
        if (df[col].astype(str).str.strip() == "").any():
            errors.append(f"empty string value(s) found in {col} — use 'MISSING' instead")
    for col in ["verbatimDate_confidence", "verbatimLocality_confidence"]:
        if ((df[col] < 0.0) | (df[col] > 1.0)).any():
            errors.append(f"confidence out of [0,1] range in {col}")
    present = set(df["image_file"])
    missing_rows = expected_image_files - present
    if missing_rows:
        errors.append(f"missing rows for image_file(s): {sorted(missing_rows)[:5]}...")
    return errors


def build_submission_df(image_files, manifest_by_image, lines_by_crop, is_relevant=None):
    predicted = run_pipeline(image_files, manifest_by_image, lines_by_crop, is_relevant=is_relevant)
    rows = []
    for image_file in image_files:
        date_text, date_conf = predicted[image_file]["verbatimDate"]
        loc_text, loc_conf = predicted[image_file]["verbatimLocality"]
        rows.append({
            "image_file": image_file,
            "verbatimDate": date_text,
            "verbatimDate_confidence": date_conf,
            "verbatimLocality": loc_text,
            "verbatimLocality_confidence": loc_conf,
        })
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def generate_submission(manifest_path, predictions_path, csv_path, output_path):
    image_files = pd.read_csv(csv_path)["image_file"].tolist()
    manifest_by_image = load_manifest(manifest_path)
    lines_by_crop = load_predictions(predictions_path)

    specimens = {os.path.splitext(f)[0] for f in image_files}
    relevance_flags = compute_relevance_flags_for_specimens(specimens)
    is_relevant = lambda crop_filename: relevance_flags.get(crop_filename, True)

    out_df = build_submission_df(image_files, manifest_by_image, lines_by_crop, is_relevant=is_relevant)
    errors = validate_submission(out_df, set(image_files))
    if errors:
        raise ValueError(f"submission failed validation: {errors}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"wrote {len(out_df)} rows to {output_path}")


if __name__ == "__main__":
    MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
    PREDICTIONS_PATH = os.path.join("data", "processed", "card_crops", "predictions.csv")
    TEST_CSV = os.path.join("data", "raw", "test.csv")
    OUTPUT_CSV = os.path.join("data", "processed", "submission.csv")
    generate_submission(MANIFEST_PATH, PREDICTIONS_PATH, TEST_CSV, OUTPUT_CSV)
