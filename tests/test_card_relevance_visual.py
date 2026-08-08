import csv
import os
import tempfile

import cv2
import numpy as np

from src.card_relevance_visual import compute_relevance_flags_for_specimens, load_or_compute_relevance_flags
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


class _CountingFakeModel(_FakeModel):
    """Same predictions as _FakeModel, but counts how many times predict was
    called -- tests use this to prove the cache path skips computation."""

    def __init__(self):
        self.predict_calls = 0

    def predict(self, x):
        self.predict_calls += 1
        return super().predict(x)


def test_load_or_compute_relevance_flags_computes_and_caches_on_first_call():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.csv")
        images_dir = os.path.join(tmp_dir, "images")
        cache_path = os.path.join(tmp_dir, "cache.csv")
        os.makedirs(images_dir)
        _write_manifest(manifest_path, [("a.jpeg", 0, "a_card00.jpeg")])
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)

        flags = load_or_compute_relevance_flags(
            ["a"], cache_path, model=_FakeModel(), manifest_path=manifest_path, images_dir=images_dir)

        assert flags == {"a_card00.jpeg": True}
        assert os.path.exists(cache_path)


def test_load_or_compute_relevance_flags_reads_cache_without_recomputing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.csv")
        images_dir = os.path.join(tmp_dir, "images")
        cache_path = os.path.join(tmp_dir, "cache.csv")
        os.makedirs(images_dir)
        _write_manifest(manifest_path, [("a.jpeg", 0, "a_card00.jpeg")])
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)

        model = _CountingFakeModel()
        load_or_compute_relevance_flags(
            ["a"], cache_path, model=model, manifest_path=manifest_path, images_dir=images_dir)
        assert model.predict_calls == 1

        # second call, same cache_path, fresh model instance -- if it hit the
        # cache it never touches the model at all
        flags = load_or_compute_relevance_flags(
            ["a"], cache_path, model=_CountingFakeModel(), manifest_path=manifest_path, images_dir=images_dir)

        assert flags == {"a_card00.jpeg": True}


def test_load_or_compute_relevance_flags_recomputes_when_cache_missing_crops():
    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_path = os.path.join(tmp_dir, "manifest.csv")
        images_dir = os.path.join(tmp_dir, "images")
        cache_path = os.path.join(tmp_dir, "cache.csv")
        os.makedirs(images_dir)
        # cache only knows about a_card00 -- a newly-added a_card01 isn't in it
        with open(cache_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["crop_filename", "is_relevant"])
            writer.writeheader()
            writer.writerow({"crop_filename": "a_card00.jpeg", "is_relevant": "True"})

        _write_manifest(manifest_path, [("a.jpeg", 0, "a_card00.jpeg"), ("a.jpeg", 1, "a_card01.jpeg")])
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)
        _write_tiny_crop(images_dir, "a_card01.jpeg", fill_value=20)

        flags = load_or_compute_relevance_flags(
            ["a"], cache_path, model=_FakeModel(), manifest_path=manifest_path, images_dir=images_dir)

        assert flags == {"a_card00.jpeg": True, "a_card01.jpeg": False}


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
