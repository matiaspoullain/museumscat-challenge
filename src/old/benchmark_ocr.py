"""Benchmark an OCR engine on real detected ROI crops, for visual inspection.

Usage:
    conda run -n museumscat-challenge python -m src.old.benchmark_ocr <engine> [n_images]

<engine>: one of "doctr", "easyocr", "pytesseract" (only "doctr" implemented so far).
n_images: how many train images to sample (default 5).

For each sampled image, saves one composite jpeg to
tables_figures/eda/ocr_benchmark_<engine>/<image_file>: top panel is the
full deskewed frame with the detected search ROI (red box); bottom panel is
that ROI crop with each recognized word boxed in green and labelled
"text (confidence)".
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

from src.detect_labels import _deskew, find_search_roi

IMAGES_DIR = os.path.join("data", "raw", "images")
TRAIN_CSV = os.path.join("data", "raw", "train.csv")
OUT_DIR_TEMPLATE = os.path.join("tables_figures", "eda", "ocr_benchmark_{engine}")

CONTEXT_PANEL_WIDTH = 1400
CROP_MAX_WIDTH = 1800


def _sample_image_files(n, seed=7):
    df = pd.read_csv(TRAIN_CSV)
    sub = df[(df.verbatimDate != "MISSING") & (df.verbatimLocality != "MISSING")]
    return sub.sample(n, random_state=seed)["image_file"].tolist()


def _doctr_words(crop_bgr):
    """Run doctr on a single crop, return list of (text, confidence, (x1,y1,x2,y2))
    in crop pixel coordinates."""
    from doctr.models import ocr_predictor

    if not hasattr(_doctr_words, "_predictor"):
        _doctr_words._predictor = ocr_predictor(pretrained=True, assume_straight_pages=True)

    h, w = crop_bgr.shape[:2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    result = _doctr_words._predictor([crop_rgb])

    words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (rx1, ry1), (rx2, ry2) = word.geometry
                    words.append((word.value, word.confidence, (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h))))
    return words


def _easyocr_words(crop_bgr):
    """Run easyocr on a single crop, return list of (text, confidence, (x1,y1,x2,y2))
    in crop pixel coordinates."""
    import easyocr

    if not hasattr(_easyocr_words, "_reader"):
        _easyocr_words._reader = easyocr.Reader(["da", "en"], gpu=False)

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    detections = _easyocr_words._reader.readtext(crop_rgb)

    words = []
    for box, text, conf in detections:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append((text, conf, (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))))
    return words


def _make_doctr_multilingual_words(det_arch):
    """Same as _doctr_words but swaps the default (French-vocab) recognizer for
    Felix92/doctr-torch-parseq-multilingual-v1, a community PARSeq checkpoint
    whose vocab includes aeoeaa/AEOEAA (Danish) -- see
    tables_figures/eda/ocr_benchmark_doctr-multilingual/notes.md. Parameterized
    over det_arch to isolate whether detector or recognizer drives the
    confidence-calibration issue found with db_resnet50."""

    def run(crop_bgr):
        from doctr.models import from_hub, ocr_predictor

        if not hasattr(run, "_predictor"):
            reco_model = from_hub("Felix92/doctr-torch-parseq-multilingual-v1")
            run._predictor = ocr_predictor(
                det_arch=det_arch, reco_arch=reco_model, pretrained=True, assume_straight_pages=True
            )

        h, w = crop_bgr.shape[:2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = run._predictor([crop_rgb])

        words = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        (rx1, ry1), (rx2, ry2) = word.geometry
                        words.append((word.value, word.confidence, (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h))))
        return words

    return run


_doctr_multilingual_words = _make_doctr_multilingual_words("db_resnet50")
_doctr_multilingual_fastdet_words = _make_doctr_multilingual_words("fast_base")

FINETUNED_DANISH_CHECKPOINT = os.path.join("models", "reco_finetune_v2", "danish_crnn_vgg16_bn_stage2.pt")


def _doctr_finetuned_danish_words(crop_bgr):
    """Run doctr on a single crop using the Danish-vocab fine-tuned
    crnn_vgg16_bn recognizer (see
    docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md),
    return list of (text, confidence, (x1,y1,x2,y2)) in crop pixel
    coordinates."""
    import torch
    from doctr.datasets import VOCABS
    from doctr.models import ocr_predictor, recognition

    if not hasattr(_doctr_finetuned_danish_words, "_predictor"):
        reco_model = recognition.__dict__["crnn_vgg16_bn"](pretrained=False, vocab=VOCABS["danish"])
        reco_model.load_state_dict(torch.load(FINETUNED_DANISH_CHECKPOINT, map_location="cpu", weights_only=True))
        _doctr_finetuned_danish_words._predictor = ocr_predictor(
            det_arch="db_resnet50", reco_arch=reco_model, pretrained=True, assume_straight_pages=True
        )

    h, w = crop_bgr.shape[:2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    result = _doctr_finetuned_danish_words._predictor([crop_rgb])

    words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (rx1, ry1), (rx2, ry2) = word.geometry
                    words.append((word.value, word.confidence, (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h))))
    return words


FINETUNED_DANISH_V1_CHECKPOINT = os.path.join("models", "reco_finetune", "danish_crnn_vgg16_bn.pt")
FINETUNED_DANISH_V1_STAGE1_CHECKPOINT = os.path.join("models", "reco_finetune", "danish_crnn_vgg16_bn_stage1.pt")
FINETUNED_DANISH_V1_STAGE2_CHECKPOINT = os.path.join("models", "reco_finetune", "danish_crnn_vgg16_bn_stage2.pt")


def _make_doctr_finetuned_danish_v1_words(checkpoint_path):
    """Build a doctr-danish-vocab engine loading a checkpoint from the first,
    superseded fine-tune iteration (see
    docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md) -- kept
    only for historical stage1-vs-stage2 comparison; the active engine is
    _doctr_finetuned_danish_words above, from the v2 iteration. Returns a
    run(crop_bgr) -> list[(text, confidence, (x1,y1,x2,y2))] closure."""

    def run(crop_bgr):
        import torch
        from doctr.datasets import VOCABS
        from doctr.models import ocr_predictor, recognition

        if not hasattr(run, "_predictor"):
            reco_model = recognition.__dict__["crnn_vgg16_bn"](pretrained=False, vocab=VOCABS["danish"])
            reco_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
            run._predictor = ocr_predictor(
                det_arch="db_resnet50", reco_arch=reco_model, pretrained=True, assume_straight_pages=True
            )

        h, w = crop_bgr.shape[:2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        result = run._predictor([crop_rgb])

        words = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        (rx1, ry1), (rx2, ry2) = word.geometry
                        words.append((word.value, word.confidence, (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h))))
        return words

    return run


_doctr_finetuned_danish_v1_words = _make_doctr_finetuned_danish_v1_words(FINETUNED_DANISH_V1_CHECKPOINT)
_doctr_finetuned_danish_v1_stage1_words = _make_doctr_finetuned_danish_v1_words(FINETUNED_DANISH_V1_STAGE1_CHECKPOINT)
_doctr_finetuned_danish_v1_stage2_words = _make_doctr_finetuned_danish_v1_words(FINETUNED_DANISH_V1_STAGE2_CHECKPOINT)


def _pytesseract_words(crop_bgr):
    """Run pytesseract on a single crop, return list of (text, confidence, (x1,y1,x2,y2))
    in crop pixel coordinates. Confidence is tesseract's own 0-100 scale, rescaled to 0-1."""
    import pytesseract

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(crop_rgb, lang="dan+eng", output_type=pytesseract.Output.DICT)

    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append((text, conf / 100.0, (x, y, x + w, y + h)))
    return words


ENGINES = {
    "doctr": _doctr_words,
    "doctr-multilingual": _doctr_multilingual_words,
    "doctr-multilingual-fastdet": _doctr_multilingual_fastdet_words,
    "doctr-finetuned-danish": _doctr_finetuned_danish_words,
    "doctr-finetuned-danish-v1": _doctr_finetuned_danish_v1_words,
    "doctr-finetuned-danish-v1-stage1": _doctr_finetuned_danish_v1_stage1_words,
    "doctr-finetuned-danish-v1-stage2": _doctr_finetuned_danish_v1_stage2_words,
    "easyocr": _easyocr_words,
    "pytesseract": _pytesseract_words,
}


def _context_panel(deskewed, roi_box):
    x1, y1, x2, y2 = roi_box
    img = deskewed.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10)
    h, w = img.shape[:2]
    scale = CONTEXT_PANEL_WIDTH / w
    return cv2.resize(img, (CONTEXT_PANEL_WIDTH, int(h * scale)))


def _annotated_crop_panel(crop, words):
    h, w = crop.shape[:2]
    scale = min(1.0, CROP_MAX_WIDTH / w)
    out = cv2.resize(crop, (int(w * scale), int(h * scale)))

    for text, conf, (wx1, wy1, wx2, wy2) in words:
        x1, y1, x2, y2 = (int(v * scale) for v in (wx1, wy1, wx2, wy2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{text} ({conf:.2f})"
        ty = max(0, y1 - 6)
        cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def _stack_panels(top, bottom):
    w = max(top.shape[1], bottom.shape[1])

    def pad(panel):
        if panel.shape[1] == w:
            return panel
        pad_width = w - panel.shape[1]
        return cv2.copyMakeBorder(panel, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return np.vstack([pad(top), pad(bottom)])


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ENGINES:
        print(__doc__)
        sys.exit(1)

    engine = sys.argv[1]
    n_images = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_engine = ENGINES[engine]

    out_dir = OUT_DIR_TEMPLATE.format(engine=engine)
    os.makedirs(out_dir, exist_ok=True)

    for image_file in _sample_image_files(n_images):
        image_path = os.path.join(IMAGES_DIR, image_file)
        raw = cv2.imread(image_path)
        deskewed, _ = _deskew(raw)
        _, roi_box = find_search_roi(image_path)
        x1, y1, x2, y2 = roi_box
        crop = deskewed[y1:y2, x1:x2]

        words = run_engine(crop)

        top = _context_panel(deskewed, roi_box)
        bottom = _annotated_crop_panel(crop, words)
        composite = _stack_panels(top, bottom)

        out_path = os.path.join(out_dir, image_file)
        cv2.imwrite(out_path, composite)
        print(f"{image_file}: {len(words)} words -> {out_path}")


if __name__ == "__main__":
    main()
