"""Per-card detection within a specimen photo's search ROI, via Segment
Anything (SAM) zero-shot segmentation.

Three hand-crafted CV approaches (brightness+texture, Canny+rectangularity,
blur-then-variance) were tried and abandoned for this -- see
tables_figures/eda/card_detection_variance_test/notes.md. SAM segments
individual cards zero-shot; raw output also includes the mount background,
the foam edge strip, and small nested masks on individual words/letters,
which select_card_masks filters out. Calibrated and validated against a
6-image sample in tables_figures/eda/sam_card_detection_test/notes.md.

Consumed by:
- src/eda_sam_card_detection.py (visual validation/threshold tuning)
- src/extract_card_crops.py (production: saves each detected card as its
  own full-resolution image file)
"""
import os
from typing import TypedDict

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

CHECKPOINT = os.path.join("models", "sam", "sam_vit_b_01ec64.pth")
TARGET_WIDTH = 1600  # downsample for CPU-feasible SAM runtime; card boxes are
                      # scaled back up to full resolution by detect_cards()
POINTS_PER_SIDE = 32  # denser than SAM's default 16 -- needed for recall on
                       # small confound tag cards, see notes.md's "Bug 2"; ~4x
                       # slower than 16, see notes.md's runtime tradeoff table

# select_card_masks thresholds, calibrated against the 6-sample visual review
# in tables_figures/eda/sam_card_detection_test/notes.md
MAX_AREA_FRAC = 0.25    # drop the mount background (always by far the largest mask)
MIN_AREA_FRAC = 0.0015  # drop individual letter/glyph fragments and small noise specks
MAX_ASPECT = 4.0        # drop the long thin foam-edge-strip mask
MIN_SOLIDITY = 0.7      # drop masks that don't fill their own bbox -- real cards are solid
                        # rectangles (observed solidity 0.83-0.98); a spurious mask that
                        # bridges two separate cards across a mount gap measured 0.46-0.49
CONTAINMENT_THRESH = 0.7  # drop a mask mostly nested inside an already-kept larger one

Box = tuple[int, int, int, int]  # (x1, y1, x2, y2)


class SamMask(TypedDict):
    """The subset of a SamAutomaticMaskGenerator.generate() entry this module reads."""

    segmentation: np.ndarray
    area: int


class _Candidate(TypedDict):
    bbox: Box
    bbox_area: int


def configure_threads(n: int) -> None:
    """Cap torch's and OpenCV's intra-op CPU parallelism to n threads. SAM
    inference is CPU-bound (matmul/conv-heavy) and both libraries default to
    using every available core, so without this a run saturates the whole
    machine -- callers (e.g. src/extract_card_crops.py) expose this as a
    --workers flag so that's opt-out, not silent."""
    torch.set_num_threads(n)
    cv2.setNumThreads(n)


def load_generator() -> SamAutomaticMaskGenerator:
    """Load the SAM ViT-B model (CPU) and return a SamAutomaticMaskGenerator."""
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
    sam.to("cpu")
    return SamAutomaticMaskGenerator(sam, points_per_side=POINTS_PER_SIDE)


def downsample(crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize a full-resolution ROI crop to TARGET_WIDTH for SAM. Returns
    (small_bgr, scale) where scale = small_width / original_width."""
    h, w = crop_bgr.shape[:2]
    scale = TARGET_WIDTH / w
    small = cv2.resize(crop_bgr, (TARGET_WIDTH, int(h * scale)))
    return small, scale


def _bbox(mask_entry: SamMask) -> Box:
    ys, xs = np.where(mask_entry["segmentation"])
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _overlap_frac(box_a: Box, box_b: Box, area_a: int) -> float:
    """Fraction of box_a's area covered by the intersection with box_b."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / area_a


def select_card_masks(masks: list[SamMask], image_shape: tuple[int, ...]) -> list[Box]:
    """Filter raw SAM masks down to card-shaped candidates: drop the huge
    background mask, the thin foam-edge strip, tiny letter/noise fragments,
    and masks mostly nested inside an already-kept larger candidate."""
    roi_area = image_shape[0] * image_shape[1]

    candidates: list[_Candidate] = []
    for m in masks:
        area_frac = m["area"] / roi_area
        if area_frac > MAX_AREA_FRAC or area_frac < MIN_AREA_FRAC:
            continue
        x1, y1, x2, y2 = _bbox(m)
        w, h = x2 - x1 + 1, y2 - y1 + 1
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > MAX_ASPECT:
            continue
        bbox_area = w * h
        solidity = m["area"] / bbox_area
        if solidity < MIN_SOLIDITY:
            continue
        candidates.append({"bbox": (x1, y1, x2, y2), "bbox_area": bbox_area})

    candidates.sort(key=lambda c: -c["bbox_area"])
    kept: list[_Candidate] = []
    for c in candidates:
        if any(_overlap_frac(c["bbox"], k["bbox"], c["bbox_area"]) > CONTAINMENT_THRESH for k in kept):
            continue
        kept.append(c)
    return [c["bbox"] for c in kept]


def detect_cards(crop_bgr: np.ndarray, generator: SamAutomaticMaskGenerator) -> list[Box]:
    """Detect card bounding boxes in a full-resolution search-ROI crop.
    Returns a list of (x1, y1, x2, y2) boxes in crop_bgr's own coordinates
    (SAM runs on a downsampled copy internally; boxes are scaled back up)."""
    small, scale = downsample(crop_bgr)
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    masks = generator.generate(small_rgb)
    small_boxes = select_card_masks(masks, small.shape[:2])

    h, w = crop_bgr.shape[:2]
    full_boxes: list[Box] = []
    for (x1, y1, x2, y2) in small_boxes:
        fx1, fy1 = int(x1 / scale), int(y1 / scale)
        fx2, fy2 = int(x2 / scale), int(y2 / scale)
        full_boxes.append((max(0, fx1), max(0, fy1), min(w, fx2), min(h, fy2)))
    return full_boxes
