"""Evaluate the fine-tuned Danish-vocab doctr recognizer against the
baseline default recognizer, directly on the held-out val split's word
crops (data/processed/reco_finetune_v2/dataset/val/), which have exact
per-word ground truth from the manual box-labeling app.

Deviates from the plan's original design
(docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md,
Task 9): that version ran src.old.benchmark_ocr.ENGINES, which wraps a full
detector+recognizer pipeline. Since these crops are already tight
single-word images, the db_resnet50/fast_base detector frequently fails to
find any text region at all (~60-70% empty results in a manual check),
which dominates the resulting NED and tells you nothing about recognizer
quality. This script instead calls doctr's recognition_predictor directly,
skipping detection entirely, to isolate the recognizer's own accuracy.

Usage:
    conda run -n museumscat-challenge python -m src.old.evaluate_card_reco
"""
import json
import os

import cv2
import torch
from doctr.datasets import VOCABS
from doctr.models import recognition, recognition_predictor

from src.metrics import ned

VAL_DIR = os.path.join("data", "processed", "reco_finetune_v2", "dataset", "val")
VAL_IMAGES_DIR = os.path.join(VAL_DIR, "images")
VAL_LABELS_PATH = os.path.join(VAL_DIR, "labels.json")
FINETUNED_DANISH_CHECKPOINT = os.path.join("models", "reco_finetune_v2", "danish_crnn_vgg16_bn_stage2.pt")


def _build_predictor(name):
    """name: 'doctr' for the stock pretrained crnn_vgg16_bn recognizer, or
    'doctr-finetuned-danish' for the Danish-vocab fine-tuned checkpoint."""
    if name == "doctr":
        return recognition_predictor(arch="crnn_vgg16_bn", pretrained=True)
    if name == "doctr-finetuned-danish":
        reco_model = recognition.__dict__["crnn_vgg16_bn"](pretrained=False, vocab=VOCABS["danish"])
        reco_model.load_state_dict(torch.load(FINETUNED_DANISH_CHECKPOINT, map_location="cpu", weights_only=True))
        return recognition_predictor(arch=reco_model)
    raise ValueError(f"unknown engine name: {name}")


def _recognize(predictor, image_path):
    """Run a recognition_predictor on a single tight word crop and return
    its best-guess text."""
    crop = cv2.imread(image_path)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    (text, _confidence), = predictor([crop_rgb])
    return text


def evaluate(engine_name):
    """Run the named recognizer over every val word crop, return mean NED
    against the true word (case-insensitive, per src.metrics.ned)."""
    with open(VAL_LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    predictor = _build_predictor(engine_name)
    errors = [ned(_recognize(predictor, os.path.join(VAL_IMAGES_DIR, filename)), truth)
              for filename, truth in labels.items()]
    return sum(errors) / len(errors)


if __name__ == "__main__":
    for name in ("doctr", "doctr-finetuned-danish"):
        mean_ned = evaluate(name)
        print(f"{name}: mean word NED = {mean_ned:.3f}")
