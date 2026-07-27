"""Package the train/val split into a folder ready to become a private
Kaggle Dataset (kaggle datasets create -p <this folder>) for the real
GPU fine-tune run. Does not call the Kaggle API itself -- see
docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md Task 6
for why that step stays manual/explicit.

Usage:
    conda run -n museumscat-challenge python -m src.package_reco_dataset
"""
import json
import os
import shutil

DATASET_DIR = os.path.join("data", "processed", "reco_finetune", "dataset")
OUT_DIR = os.path.join("data", "processed", "reco_finetune", "kaggle_dataset")
KAGGLE_DATASET_SLUG = "museumscat-danish-recognizer-crops"
KAGGLE_USERNAME = "matiaspoullain"


def package():
    """Copy train/ and val/ into OUT_DIR and write dataset-metadata.json for kaggle CLI."""
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    shutil.copytree(os.path.join(DATASET_DIR, "train"), os.path.join(OUT_DIR, "train"))
    shutil.copytree(os.path.join(DATASET_DIR, "val"), os.path.join(OUT_DIR, "val"))

    metadata = {
        "title": "MuseumSCAT Danish recognizer crops",
        "id": f"{KAGGLE_USERNAME}/{KAGGLE_DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    with open(os.path.join(OUT_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Packaged at {OUT_DIR}")


if __name__ == "__main__":
    package()
