import csv
import os
import tempfile

import cv2
import numpy as np

from src.card_relevance_visual import compute_relevance_flags_for_specimens
from src.eda_card_visual_features import FEATURE_NAMES


def _write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_image_file", "card_index", "crop_filename"])
        writer.writeheader()
        for source_image_file, card_index, crop_filename in rows:
            writer.writerow({"source_image_file": source_image_file, "card_index": card_index,
                              "crop_filename": crop_filename})


def _write_tiny_crop(images_dir, crop_filename, fill_value):
    image = np.full((40, 60, 3), fill_value, dtype=np.uint8)
    cv2.imwrite(os.path.join(images_dir, crop_filename), image)


class _FakeModel:
    """Predicts True for any row whose mean_value feature is >= 128, False
    otherwise -- lets tests control the prediction via each crop's fill
    value without needing a real fitted RandomForestClassifier."""

    def predict(self, x):
        mean_value_index = FEATURE_NAMES.index("mean_value")
        return np.array([row[mean_value_index] >= 128 for row in x])


def test_compute_relevance_flags_for_specimens_matches_per_crop_prediction():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.csv")
        images_dir = os.path.join(tmp_dir, "images")
        os.makedirs(images_dir)
        _write_manifest(manifest_path, [
            ("a.jpeg", 0, "a_card00.jpeg"), ("a.jpeg", 1, "a_card01.jpeg"),
        ])
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)  # bright -> relevant
        _write_tiny_crop(images_dir, "a_card01.jpeg", fill_value=20)   # dark -> confound

        flags = compute_relevance_flags_for_specimens(
            ["a"], model=_FakeModel(), manifest_path=manifest_path, images_dir=images_dir)

        assert flags == {"a_card00.jpeg": True, "a_card01.jpeg": False}


def test_compute_relevance_flags_for_specimens_only_includes_given_specimens_crops():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.csv")
        images_dir = os.path.join(tmp_dir, "images")
        os.makedirs(images_dir)
        _write_manifest(manifest_path, [
            ("a.jpeg", 0, "a_card00.jpeg"), ("b.jpeg", 0, "b_card00.jpeg"),
        ])
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)
        _write_tiny_crop(images_dir, "b_card00.jpeg", fill_value=200)

        flags = compute_relevance_flags_for_specimens(
            ["a"], model=_FakeModel(), manifest_path=manifest_path, images_dir=images_dir)

        assert list(flags) == ["a_card00.jpeg"]


def test_compute_relevance_flags_for_specimens_fails_open_on_unreadable_crop():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.csv")
        images_dir = os.path.join(tmp_dir, "images")
        os.makedirs(images_dir)
        _write_manifest(manifest_path, [("a.jpeg", 0, "a_card00.jpeg")])
        # no image actually written for a_card00.jpeg -- cv2.imread returns None

        flags = compute_relevance_flags_for_specimens(
            ["a"], model=_FakeModel(), manifest_path=manifest_path, images_dir=images_dir)

        assert flags == {"a_card00.jpeg": True}


def test_compute_relevance_flags_for_specimens_prints_progress(capsys):
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.csv")
        images_dir = os.path.join(tmp_dir, "images")
        os.makedirs(images_dir)
        rows = [("a.jpeg", i, f"a_card{i:02d}.jpeg") for i in range(3)]
        _write_manifest(manifest_path, rows)
        for _, _, crop_filename in rows:
            _write_tiny_crop(images_dir, crop_filename, fill_value=200)

        compute_relevance_flags_for_specimens(
            ["a"], model=_FakeModel(), manifest_path=manifest_path, images_dir=images_dir,
            progress_every=2)

        assert "2/3 crops" in capsys.readouterr().out


def test_compute_relevance_flags_for_specimens_progress_every_zero_disables_printing(capsys):
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.csv")
        images_dir = os.path.join(tmp_dir, "images")
        os.makedirs(images_dir)
        _write_manifest(manifest_path, [("a.jpeg", 0, "a_card00.jpeg")])
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)

        compute_relevance_flags_for_specimens(
            ["a"], model=_FakeModel(), manifest_path=manifest_path, images_dir=images_dir,
            progress_every=0)

        assert capsys.readouterr().out == ""
