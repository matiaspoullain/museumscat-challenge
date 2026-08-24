"""Package the train/val split into a folder ready to become a private
Kaggle Dataset (kaggle datasets create -p <this folder>) for the Danish
recognizer fine-tune run. Does not call the Kaggle API itself -- see
docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md for
why that step stays manual/explicit.

Usage:
    conda run -n museumscat-challenge python -m src.old.package_reco_dataset_v2
"""
import json
import os
import shutil

DATASET_DIR = os.path.join("data", "processed", "reco_finetune_v2", "dataset")
OUT_DIR = os.path.join("data", "processed", "reco_finetune_v2", "kaggle_dataset")
KAGGLE_DATASET_SLUG = "museumscat-danish-recognizer-word-boxes"


def package():
    """Copy train/ and val/ into OUT_DIR and write dataset-metadata.json for kaggle CLI."""
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    shutil.copytree(os.path.join(DATASET_DIR, "train"), os.path.join(OUT_DIR, "train"))
    shutil.copytree(os.path.join(DATASET_DIR, "val"), os.path.join(OUT_DIR, "val"))

    metadata = {
        "title": "MuseumSCAT Danish recognizer word boxes",
        "id": f"CHANGE_ME/{KAGGLE_DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    with open(os.path.join(OUT_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Packaged at {OUT_DIR}")
    print("Edit dataset-metadata.json 'id' to '<your-kaggle-username>/" + KAGGLE_DATASET_SLUG + "' before upload.")


if __name__ == "__main__":
    package()
