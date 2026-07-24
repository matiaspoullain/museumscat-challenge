# MuseumSCAT OCR Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Adaptation note:** This is a research/ML pipeline, not a deterministic software feature. Tasks 1-4 (evaluation harness, field classification, multi-card merge, submission generation) are pure/deterministic logic and follow strict TDD. Tasks 5-7 (label detection, OCR engine choice, confidence calibration) are computer-vision/ML work validated by inspection against real images and by the NED/AURC harness from Task 1, not by invented unit tests — per the "No Placeholders" rule, we do not fabricate asserts for things that can't be asserted (e.g. "the crop contains the label" without ground-truth boxes).

**Goal:** Build a two-stage pipeline (detect label cards on the specimen photo → OCR only the crops → classify/merge into `verbatimDate`/`verbatimLocality` with calibrated confidence) that produces a valid, scored Kaggle submission for the MuseumSCAT competition, plus a local NED/AURC evaluation harness to iterate without burning submission quota.

**Architecture:** Classical CV for label localization (the studio photos have a fixed uniform background and light rectangular tag cards — no bounding-box ground truth exists, so a trained detector isn't an option yet; contour/color-based segmentation is). Cropped label images go through an OCR engine (candidate: `doctr`, benchmarked against `easyocr`/`pytesseract`) to get text + token confidence. A rule-based classifier splits recognized snippets into date / locality / confounding (collector, determination, museum-boilerplate) text. A merge step joins same-specimen multi-card values with `|`, matching the competition's permutation-invariant scoring. A calibration step (isotonic regression on the 200-example train set) turns raw OCR confidence into the calibrated confidence the AURC metric rewards.

**Tech Stack:** Python 3.11, conda env `museumscat-challenge` (see `environment.yml`), pandas/numpy/pillow already installed. OCR/CV libraries (`opencv-python-headless`, `doctr` or `easyocr`) are added in Task 5/6 — ask before pinning the final choice, since these are non-trivial dependencies (torch-based).

## Global Constraints

- Images and any derived crops/model artifacts are **never** committed to git — `data/raw/`, `data/processed/`, `models/` are gitignored (see `.gitignore`). Only code, `kaggle_meta/` (text), small tables/figures, and CSVs under a few KB may be committed.
- Ground-truth train set is only 200 rows (`data/raw/train.csv`) against 3300 test rows (`data/raw/test.csv`) — every evaluation must work with this small-N reality (use all 200 as a dev set; no held-out split is large enough to be trustworthy, so prefer k-fold or just track dev-set NED/AURC directly and watch for overfitting via qualitative crop review).
- Missing values must be submitted as the literal string `"MISSING"` (not empty string) — this is a Kaggle CSV constraint, not a modeling choice.
- Submission columns, exact order: `image_file,verbatimDate,verbatimDate_confidence,verbatimLocality,verbatimLocality_confidence`.
- No paid VLM/LLM APIs for the initial pipeline (per user decision 2026-07-24). Anthropic API is a possible Phase-2 fallback for low-confidence crops only — keep `ocr_labels.py`'s interface swappable so this can be added later without a redesign.
- Competition data license is CC BY-NC 4.0 (non-commercial) — no commercial reuse of the images/annotations.

---

## Status (done as of 2026-07-24)

- [x] Repo scaffolded at `/home/matias/museumscat-challenge` (no git submodules, unlike `bayesian-kaggle`).
- [x] `environment.yml` + conda env `museumscat-challenge` created (python=3.11, kaggle, pandas, numpy, pillow, scikit-learn).
- [x] `src/download_kaggle_data.py` written (ported from `bayesian-kaggle/template`, slug filled in) and run — `data/raw/{train.csv,test.csv,images/*.jpeg}` (3500 images, 6.9GB) and `kaggle_meta/pages/*.md` are on disk.
- [x] Competition understood: OCR of `verbatimDate` + `verbatimLocality` from Danish Dung Beetle specimen label photos; 200/3500 labeled; scored via AURC (NED + confidence calibration), lower is better; multi-card specimens use `|`-joined, permutation-scored values.
- [x] Sample images inspected (`1709789.jpeg`, `1742772.jpeg`, `1709018.jpeg`): every photo has a **fixed-layout studio template** — top-left NHM Denmark logo card, DaSSCo logo card, color-checker + ruler + NHMD barcode card (bottom-left), and a foam mounting block (top/right area) carrying 1-4 small variable-position paper/plastic tag cards (white/cream/tan) plus the pinned specimen itself. The tag cards are the only OCR targets; everything else is boilerplate to exclude.

---

## Task 1: Local evaluation harness (NED + AURC)

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `edit_distance(a: str, b: str) -> int`, `normalize_date_punctuation(s: str) -> str`, `ned(pred: str, truth: str, is_date: bool = False) -> float`, `best_pipe_ned(pred: str, truth: str, is_date: bool = False) -> float`, `aurc(errors: list[float], confidences: list[float]) -> float`. All later tasks that score predictions against `data/raw/train.csv` call `best_pipe_ned` and `aurc`.

- [ ] **Step 1: Write failing tests for `edit_distance` and `normalize_date_punctuation`**

```python
# tests/test_metrics.py
from src.metrics import edit_distance, normalize_date_punctuation, ned, best_pipe_ned, aurc

def test_edit_distance_identical():
    assert edit_distance("kb", "kb") == 0

def test_edit_distance_substitution():
    assert edit_distance("kb", "kg") == 1

def test_edit_distance_insertion():
    assert edit_distance("kb", "kbx") == 1

def test_normalize_date_punctuation_variants_match():
    variants = ["27.5.2022", "27,5,2022", "27-5-2022", "27·5·2022", "27 5 2022"]
    normalized = {normalize_date_punctuation(v) for v in variants}
    assert len(normalized) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Implement `edit_distance` and `normalize_date_punctuation`**

```python
# src/metrics.py
import itertools
import re


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n]


def normalize_date_punctuation(s):
    return re.sub(r"[.,\-·\s]+", ".", s.strip()).strip(".")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py -v`
Expected: 4 passed

- [ ] **Step 5: Write failing tests for `ned`**

```python
def test_ned_exact_match_is_zero():
    assert ned("Kb", "kb") == 0.0

def test_ned_completely_wrong_is_one():
    assert ned("abc", "xyz") == 1.0

def test_ned_date_punctuation_normalized():
    assert ned("27-5-2022", "27.5.2022", is_date=True) == 0.0

def test_ned_locality_punctuation_not_normalized():
    assert ned("27-5-2022", "27.5.2022", is_date=False) > 0.0
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py -v -k test_ned`
Expected: FAIL with `ImportError: cannot import name 'ned'`

- [ ] **Step 7: Implement `ned`**

```python
def ned(pred, truth, is_date=False):
    p, t = pred.strip().lower(), truth.strip().lower()
    if is_date:
        p, t = normalize_date_punctuation(p), normalize_date_punctuation(t)
    if p == t:
        return 0.0
    denom = max(len(p), len(t), 1)
    return edit_distance(p, t) / denom
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py -v -k test_ned`
Expected: 4 passed

- [ ] **Step 9: Write failing tests for `best_pipe_ned`**

```python
def test_best_pipe_ned_single_card():
    assert best_pipe_ned("Kb", "Kb") == 0.0

def test_best_pipe_ned_tries_orderings():
    # ground truth "Ti | Kulhuset" vs prediction submitted in swapped order
    assert best_pipe_ned("Kulhuset | Ti", "Ti | Kulhuset") == 0.0

def test_best_pipe_ned_penalizes_wrong_content_even_if_reordered():
    assert best_pipe_ned("Wrong | Ti", "Ti | Kulhuset") > 0.0
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py -v -k best_pipe_ned`
Expected: FAIL with `ImportError: cannot import name 'best_pipe_ned'`

- [ ] **Step 11: Implement `best_pipe_ned`**

```python
def best_pipe_ned(pred, truth, is_date=False):
    pred_cards = [c.strip() for c in pred.split("|")]
    truth_cards = [c.strip() for c in truth.split("|")]
    if len(pred_cards) != len(truth_cards):
        # Approximation for mismatched card counts: compare the joined
        # strings directly. The official scorer's behavior in this case
        # isn't documented; this only needs to be directionally useful
        # for comparing local pipeline iterations, not to match the
        # leaderboard exactly.
        return ned(" ".join(pred_cards), " ".join(truth_cards), is_date)
    best = min(
        ned(" ".join(perm), " ".join(truth_cards), is_date)
        for perm in itertools.permutations(pred_cards)
    )
    return best
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py -v -k best_pipe_ned`
Expected: 3 passed

- [ ] **Step 13: Write failing tests for `aurc`**

```python
def test_aurc_perfect_calibration_beats_inverted():
    errors = [0.0, 0.2, 0.5, 1.0]
    good_confidences = [1.0, 0.8, 0.5, 0.0]   # high conf on low error
    bad_confidences = [0.0, 0.2, 0.5, 1.0]    # high conf on high error
    assert aurc(errors, good_confidences) < aurc(errors, bad_confidences)

def test_aurc_all_zero_error_is_zero():
    assert aurc([0.0, 0.0, 0.0], [0.9, 0.1, 0.5]) == 0.0
```

- [ ] **Step 14: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py -v -k aurc`
Expected: FAIL with `ImportError: cannot import name 'aurc'`

- [ ] **Step 15: Implement `aurc`**

```python
def aurc(errors, confidences):
    n = len(errors)
    order = sorted(range(n), key=lambda i: -confidences[i])
    sorted_errors = [errors[i] for i in order]
    cum = 0.0
    risks = []
    for i, e in enumerate(sorted_errors, start=1):
        cum += e
        risks.append(cum / i)
    coverages = [i / n for i in range(1, n + 1)]
    area = 0.0
    for i in range(1, len(coverages)):
        area += (coverages[i] - coverages[i - 1]) * (risks[i] + risks[i - 1]) / 2
    return area
```

- [ ] **Step 16: Run full test file to verify all pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py -v`
Expected: 11 passed

- [ ] **Step 17: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: add NED/AURC local evaluation harness"
```

---

## Task 2: EDA — label layout across a sample of images

**Files:**
- Create: `src/eda_label_layout.py`
- Output: `tables_figures/eda/layout_grid.png` (committed — small, no raw specimen data leaked beyond what's already public sample images), `tables_figures/eda/layout_notes.md`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: written observations in `tables_figures/eda/layout_notes.md` about (a) whether the boilerplate furniture (logo cards, color chart, barcode) sits at a fixed pixel region across all images or only fixed *relative* position/scale, (b) the color/contrast signature of tag cards vs. the foam background, (c) how many tag cards appear per image and their rough size range — this directly informs Task 5's detection thresholds.

- [ ] **Step 1: Write the sampling + grid script**

```python
# src/eda_label_layout.py
import csv
import os
import random

from PIL import Image

IMAGES_DIR = os.path.join("data", "raw", "images")
OUT_PATH = os.path.join("tables_figures", "eda", "layout_grid.png")


def sample_image_files(n=12, seed=42):
    files = sorted(os.listdir(IMAGES_DIR))
    random.Random(seed).shuffle(files)
    return files[:n]


def build_grid(files, thumb_size=(400, 267)):
    thumbs = [Image.open(os.path.join(IMAGES_DIR, f)).resize(thumb_size) for f in files]
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (thumb_size[0] * cols, thumb_size[1] * rows), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1]
        grid.paste(thumb, (x, y))
    return grid


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    files = sample_image_files()
    grid = build_grid(files)
    grid.save(OUT_PATH)
    print(f"Sampled: {files}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and view the grid**

Run: `conda run -n museumscat-challenge python src/eda_label_layout.py`
Then Read `tables_figures/eda/layout_grid.png` to visually confirm the fixed-furniture-vs-variable-tags hypothesis from the Status section across a wider sample.

- [ ] **Step 3: Write `tables_figures/eda/layout_notes.md` by hand from the visual review**

Document: pixel bounding box (approx, in original 8192x5464 coords) of the fixed furniture block if consistent, the color/brightness range of the grey-blue background vs. tag cards, and the observed range of tag-card count per image (expect 1-5 based on `train.csv` pipe-counts).

- [ ] **Step 4: Commit**

```bash
git add src/eda_label_layout.py tables_figures/eda/
git commit -m "docs: EDA on specimen photo layout"
```

---

## Task 3: Field classifier (date / locality / confound)

**Files:**
- Create: `src/classify_field.py`
- Test: `tests/test_classify_field.py`

**Interfaces:**
- Consumes: raw OCR text strings (from Task 6, but testable standalone with hand-written strings now).
- Produces: `classify_field(text: str) -> str` returning one of `"date"`, `"locality"`, `"confound"`. Task 4 (merge) groups OCR snippets by this label.

- [ ] **Step 1: Write failing tests covering the documented examples**

```python
# tests/test_classify_field.py
from src.classify_field import classify_field

def test_classifies_standard_date():
    assert classify_field("27.5.2022") == "date"

def test_classifies_roman_numeral_date():
    assert classify_field("27.IV.2022") == "date"

def test_classifies_partial_date():
    assert classify_field("IV.91") == "date"

def test_classifies_slash_date():
    assert classify_field("4/5 70") == "date"

def test_classifies_locality_kb():
    assert classify_field("Kb") == "locality"

def test_classifies_locality_with_coords():
    assert classify_field("DENMARK: NEZ: Helsingor, Gronne Vestkile, 56.0375N, 12.5598E") == "locality"

def test_classifies_confound_collector():
    assert classify_field("Coll. J. P. Johansen") == "confound"

def test_classifies_confound_determination():
    assert classify_field("det. Hansen") == "confound"

def test_classifies_confound_museum_boilerplate():
    assert classify_field("STATENS NATURHISTORISKE MUSEUM") == "confound"

def test_classifies_confound_dania():
    assert classify_field("Dania coll.") == "confound"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_classify_field.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.classify_field'`

- [ ] **Step 3: Implement `classify_field`**

```python
# src/classify_field.py
import re

ROMAN_MONTH = r"(?:I{1,3}|IV|VI{0,3}|IX|X{1,2}I{0,2})"
DATE_PATTERN = re.compile(
    rf"\b\d{{1,2}}[.\-/ ]({ROMAN_MONTH}|\d{{1,2}})[.\-/ ]\d{{2,4}}\b"
    rf"|\b({ROMAN_MONTH})[.\-/]\d{{2,4}}\b",
    re.IGNORECASE,
)

CONFOUND_KEYWORDS = [
    "coll.", "coll ", "det.", "det ", "tilg.", "dania",
    "statens naturhistoriske museum", "kobenhavns universitet",
    "dassco", "nhmd", "mus.",
]


def classify_field(text):
    t = text.strip().lower()
    if any(kw in t for kw in CONFOUND_KEYWORDS):
        return "confound"
    if DATE_PATTERN.search(text.strip()):
        return "date"
    return "locality"


if __name__ == "__main__":
    pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_classify_field.py -v`
Expected: 10 passed. If any date-regex case fails, adjust `DATE_PATTERN` iteratively — the regex is a first pass and will need real OCR output from Task 6 to refine further (revisit this task after Task 6/Task 5 produce real crops).

- [ ] **Step 5: Commit**

```bash
git add src/classify_field.py tests/test_classify_field.py
git commit -m "feat: add rule-based date/locality/confound classifier"
```

---

## Task 4: Multi-card merge + submission row assembly

**Files:**
- Create: `src/merge_records.py`
- Test: `tests/test_merge_records.py`

**Interfaces:**
- Consumes: nothing from earlier tasks directly (pure function on lists of `(text, confidence)` tuples already grouped by field type from Task 3's output).
- Produces: `assemble_field(cards: list[tuple[str, float]]) -> tuple[str, float]` — joins multiple card texts with `" | "` and combines confidences (mean), or returns `("MISSING", <default_confidence>)` for an empty list. Task 7 (submission generation) calls this once per field per specimen.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_merge_records.py
from src.merge_records import assemble_field

def test_empty_cards_is_missing():
    text, conf = assemble_field([])
    assert text == "MISSING"
    assert 0.0 <= conf <= 1.0

def test_single_card_passthrough():
    text, conf = assemble_field([("Kb", 0.9)])
    assert text == "Kb"
    assert conf == 0.9

def test_multiple_cards_joined_with_pipe_in_order():
    text, conf = assemble_field([("Ti", 0.8), ("Kulhuset Jaegerspris", 0.7)])
    assert text == "Ti | Kulhuset Jaegerspris"
    assert conf == 0.75
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_merge_records.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.merge_records'`

- [ ] **Step 3: Implement `assemble_field`**

```python
# src/merge_records.py
MISSING_CONFIDENCE = 0.5


def assemble_field(cards):
    if not cards:
        return "MISSING", MISSING_CONFIDENCE
    texts = [text for text, _ in cards]
    confidences = [conf for _, conf in cards]
    return " | ".join(texts), sum(confidences) / len(confidences)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_merge_records.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/merge_records.py tests/test_merge_records.py
git commit -m "feat: add multi-card field assembly"
```

---

## Task 5: Label card detection (classical CV)

**Files:**
- Create: `src/detect_labels.py`
- Output for review: `tables_figures/eda/detection_review/*.jpg` (annotated previews, gitignored — derived from raw images)

**Interfaces:**
- Consumes: EDA notes from Task 2 (background/tag color signature, furniture region).
- Produces: `find_label_crops(image_path: str) -> list[tuple[int, int, int, int]]` returning `(left, top, right, bottom)` boxes in original image pixel coordinates for candidate tag cards, with the fixed furniture region already excluded. Task 6 (OCR) crops each box and runs OCR on it.

- [ ] **Step 1: Add `opencv-python-headless` to `environment.yml`**

Ask the user before adding — this is a new dependency (`ask before adding new libraries/packages` is a standing preference). Once approved:

```yaml
      - opencv-python-headless
```

Then: `conda env update -n museumscat-challenge -f environment.yml`

- [ ] **Step 2: Implement contour-based candidate detection, restricted to the non-furniture region found in Task 2**

```python
# src/detect_labels.py
import cv2
import numpy as np

# Furniture exclusion box, in original-image pixel coords, from Task 2's
# layout_notes.md findings (left column containing logos/color-chart/barcode).
FURNITURE_EXCLUSION_BOX = None  # fill in from tables_figures/eda/layout_notes.md

MIN_CARD_AREA = 2000
MAX_CARD_AREA = 500000


def find_label_crops(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if not (MIN_CARD_AREA <= area <= MAX_CARD_AREA):
            continue
        if FURNITURE_EXCLUSION_BOX and _overlaps(x, y, w, h, FURNITURE_EXCLUSION_BOX):
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def _overlaps(x, y, w, h, box):
    bx1, by1, bx2, by2 = box
    return not (x + w < bx1 or x > bx2 or y + h < by1 or y > by2)
```

- [ ] **Step 2: Manual verification against 20 train images**

There is no ground-truth bounding box to assert against, so this step is inspection-driven rather than a unit test — write a small throwaway script that draws `find_label_crops` boxes on 20 images from `data/raw/train.csv` and saves them to `tables_figures/eda/detection_review/`, then Read each one. Iterate `MIN_CARD_AREA`/`MAX_CARD_AREA`/`FURNITURE_EXCLUSION_BOX` until boxes visually cover the tag cards (and only the tag cards) on at least 18/20 images. Record the final thresholds and known failure modes in `tables_figures/eda/layout_notes.md`.

- [ ] **Step 3: Commit**

```bash
git add src/detect_labels.py tables_figures/eda/layout_notes.md environment.yml
git commit -m "feat: add classical-CV label card detection"
```

---

## Task 6: OCR engine integration

**Files:**
- Create: `src/ocr_labels.py`
- Modify: `environment.yml` (OCR library — pick after benchmarking)

**Interfaces:**
- Consumes: `find_label_crops` boxes from Task 5.
- Produces: `run_ocr(crop: PIL.Image.Image) -> tuple[str, float]` (text, confidence in `[0,1]`). Task 7 calls this once per detected crop.

- [ ] **Step 1: Benchmark 2-3 OCR engines on a handful of real crops before committing to one**

This is exploratory — ask the user which engines to try (candidates: `doctr` — good with printed+some handwriting and gives per-word confidence; `easyocr` — simpler API; `pytesseract` — classic but weak on handwriting). Add the winning library to `environment.yml` (ask before adding, per standing preference) and remove candidates that lost.

- [ ] **Step 2: Implement `run_ocr` wrapping the chosen engine**

Exact implementation depends on Step 1's winner — write it as a thin adapter so `ocr_labels.py`'s only public surface is `run_ocr(crop) -> (text, confidence)`, keeping the door open to swap in a VLM-based implementation later (per the Global Constraints note on the Anthropic API being a possible Phase-2 fallback) without changing any caller.

- [ ] **Step 3: Score the full pipeline so far on the 200-row dev set**

Wire `find_label_crops` → `run_ocr` → `classify_field` → `assemble_field` → `best_pipe_ned`/`aurc` from Tasks 1-5 into a small driver script (`src/pipeline.py`, Task 7) and report mean NED and AURC per field on `data/raw/train.csv`. This is the first real accuracy signal — use it to decide whether classical OCR is sufficient or whether the Phase-2 VLM fallback is worth pursuing for low-confidence crops.

- [ ] **Step 4: Commit**

```bash
git add src/ocr_labels.py environment.yml
git commit -m "feat: add OCR engine integration"
```

---

## Task 7: Submission generation + validation

**Files:**
- Create: `src/pipeline.py`, `src/generate_submission.py`
- Test: `tests/test_generate_submission.py`

**Interfaces:**
- Consumes: `find_label_crops` (Task 5), `run_ocr` (Task 6), `classify_field` (Task 3), `assemble_field` (Task 4).
- Produces: `data/processed/submission.csv` and a pure, testable `validate_submission(df: pandas.DataFrame, expected_image_files: set[str]) -> list[str]` returning a list of validation error strings (empty list = valid).

- [ ] **Step 1: Write failing tests for `validate_submission`**

```python
# tests/test_generate_submission.py
import pandas as pd

from src.generate_submission import REQUIRED_COLUMNS, validate_submission


def test_valid_submission_has_no_errors():
    df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "MISSING", "verbatimDate_confidence": 0.5,
         "verbatimLocality": "Kb", "verbatimLocality_confidence": 0.9},
    ])
    assert validate_submission(df, {"a.jpeg"}) == []

def test_missing_column_is_an_error():
    df = pd.DataFrame([{"image_file": "a.jpeg"}])
    errors = validate_submission(df, {"a.jpeg"})
    assert any("column" in e.lower() for e in errors)

def test_empty_string_value_is_an_error():
    df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "", "verbatimDate_confidence": 0.5,
         "verbatimLocality": "Kb", "verbatimLocality_confidence": 0.9},
    ])
    errors = validate_submission(df, {"a.jpeg"})
    assert any("empty" in e.lower() for e in errors)

def test_confidence_out_of_range_is_an_error():
    df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "MISSING", "verbatimDate_confidence": 1.5,
         "verbatimLocality": "Kb", "verbatimLocality_confidence": 0.9},
    ])
    errors = validate_submission(df, {"a.jpeg"})
    assert any("confidence" in e.lower() for e in errors)

def test_missing_image_file_is_an_error():
    df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "MISSING", "verbatimDate_confidence": 0.5,
         "verbatimLocality": "Kb", "verbatimLocality_confidence": 0.9},
    ])
    errors = validate_submission(df, {"a.jpeg", "b.jpeg"})
    assert any("b.jpeg" in e for e in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_generate_submission.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.generate_submission'`

- [ ] **Step 3: Implement `validate_submission`**

```python
# src/generate_submission.py
import os

import pandas as pd

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_generate_submission.py -v`
Expected: 5 passed

- [ ] **Step 5: Implement the end-to-end driver `src/pipeline.py`**

```python
# src/pipeline.py
import os

from PIL import Image

from src.classify_field import classify_field
from src.detect_labels import find_label_crops
from src.merge_records import assemble_field
from src.ocr_labels import run_ocr

IMAGES_DIR = os.path.join("data", "raw", "images")


def process_image(image_file):
    image_path = os.path.join(IMAGES_DIR, image_file)
    boxes = find_label_crops(image_path)
    img = Image.open(image_path)
    date_cards, locality_cards = [], []
    for box in boxes:
        crop = img.crop(box)
        text, confidence = run_ocr(crop)
        field = classify_field(text)
        if field == "date":
            date_cards.append((text, confidence))
        elif field == "locality":
            locality_cards.append((text, confidence))
    date, date_conf = assemble_field(date_cards)
    locality, locality_conf = assemble_field(locality_cards)
    return date, date_conf, locality, locality_conf
```

- [ ] **Step 6: Implement submission CSV writer and run it over `test.csv`**

```python
# append to src/generate_submission.py
from src.pipeline import process_image

TEST_CSV = os.path.join("data", "raw", "test.csv")
OUTPUT_CSV = os.path.join("data", "processed", "submission.csv")


def generate_submission():
    test_df = pd.read_csv(TEST_CSV)
    rows = []
    for image_file in test_df["image_file"]:
        date, date_conf, locality, locality_conf = process_image(image_file)
        rows.append({
            "image_file": image_file,
            "verbatimDate": date,
            "verbatimDate_confidence": date_conf,
            "verbatimLocality": locality,
            "verbatimLocality_confidence": locality_conf,
        })
    out_df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    errors = validate_submission(out_df, set(test_df["image_file"]))
    if errors:
        raise ValueError(f"submission failed validation: {errors}")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"wrote {len(out_df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_submission()
```

- [ ] **Step 7: Run the full test suite**

Run: `conda run -n museumscat-challenge python -m pytest tests/ -v`
Expected: all tests pass (metrics: 11, classify_field: 10, merge_records: 3, generate_submission: 5 = 29 passed)

- [ ] **Step 8: Dry-run on the 200-row dev set and report local NED/AURC**

Write and run a small ad-hoc script (not committed, or committed as `src/evaluate_dev_set.py` if useful going forward) that calls `process_image` for every row in `data/raw/train.csv`, scores with `best_pipe_ned`/`aurc` from Task 1, and prints per-field mean NED and AURC. This is the number to watch across every later iteration of Tasks 5/6.

- [ ] **Step 9: Commit**

```bash
git add src/pipeline.py src/generate_submission.py tests/test_generate_submission.py
git commit -m "feat: add end-to-end pipeline and submission generation"
```

---

## Task 8 (later, optional): Confidence calibration

**Files:**
- Create: `src/calibrate_confidence.py`

**Interfaces:**
- Consumes: raw OCR confidences + `best_pipe_ned` errors from the Task 7 dev-set run.
- Produces: `fit_calibrator(raw_confidences: list[float], errors: list[float]) -> Callable[[float], float]` (isotonic regression via `sklearn.isotonic.IsotonicRegression`, already in `environment.yml`), used to replace `run_ocr`'s raw confidence before submission once Task 7's dev-set AURC shows raw OCR confidence is poorly calibrated (check this empirically before building it — don't build calibration the metric doesn't need).

Only pick this task up after Task 7's dev-set numbers are in — decide with the user whether calibration is worth it based on the observed gap between raw-confidence AURC and an oracle (perfectly-ordered) AURC on the same errors.

---

## Verification (end-to-end)

1. `conda run -n museumscat-challenge python -m pytest tests/ -v` — all unit tests green.
2. `conda run -n museumscat-challenge python src/generate_submission.py` — produces `data/processed/submission.csv` with 3300 rows, passing `validate_submission`.
3. Dev-set NED/AURC report (Task 7 Step 8) reviewed with the user before considering a Kaggle submission — Kaggle allows only 10/day, so don't spend one on an unvalidated pipeline.
4. First real Kaggle submission via `kaggle competitions submit` (manual, not scripted, since submitting costs quota and should be a deliberate choice) once dev-set numbers look reasonable.
