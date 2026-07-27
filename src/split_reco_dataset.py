"""Split the human-labeled Danish word-crop dataset into train/val folders
in the layout third_party/doctr_recognition/train.py expects
(images/ + labels.json in each).

Usage:
    conda run -n museumscat-challenge python -m src.split_reco_dataset
"""
import json
import os
import random
import shutil

DATASET_DIR = os.path.join("data", "processed", "reco_finetune", "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.json")
VAL_FRACTION = 0.2
SEED = 7


def split_labels(labels, val_fraction, seed):
    """Deterministically split a {filename: text} dict into (train, val) dicts."""
    filenames = sorted(labels)
    rng = random.Random(seed)
    rng.shuffle(filenames)
    n_val = round(len(filenames) * val_fraction)
    val_filenames = set(filenames[:n_val])
    train = {f: labels[f] for f in filenames if f not in val_filenames}
    val = {f: labels[f] for f in filenames if f in val_filenames}
    return train, val


def _write_split(name, split_labels_dict):
    """Copy each labeled crop into DATASET_DIR/<name>/images/ and write labels.json there.
    Removes any pre-existing split_dir first, so reruns with a different
    label set don't leave stale copies from a previous split behind."""
    split_dir = os.path.join(DATASET_DIR, name)
    split_images_dir = os.path.join(split_dir, "images")
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_images_dir, exist_ok=True)
    for filename in split_labels_dict:
        shutil.copy(os.path.join(IMAGES_DIR, filename), os.path.join(split_images_dir, filename))
    with open(os.path.join(split_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(split_labels_dict, f, ensure_ascii=False, indent=2)


def main():
    """Read labels.json, split it, and write train/ and val/ subfolders."""
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    if "null" in labels:
        # A client-side bug in label_crops_app.py can POST filename: null,
        # which json.dump coerces into the literal string key "null" -- it
        # doesn't correspond to any real crop file and can't be recovered
        # since the intended filename was never recorded.
        print(f"dropping corrupt 'null' label entry (value: {labels['null']!r})")
        del labels["null"]
    train, val = split_labels(labels, VAL_FRACTION, SEED)
    _write_split("train", train)
    _write_split("val", val)
    print(f"train: {len(train)} crops, val: {len(val)} crops")


if __name__ == "__main__":
    main()
