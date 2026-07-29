# End-to-End Submission Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Adaptation note:** Tasks 2-5 are pure, deterministic logic (grouping, classification, assembly, scoring) and follow strict TDD with fixture data — no doctr/torch import needed, no real images touched. Task 1 is the one ML/heavy-compute piece: it edits a Kaggle-notebook cell that reruns OCR prediction over ~6,500+ card crops on a T4x2 GPU. Per project convention, that rerun is a manual, user-triggered action on Kaggle, not something this session executes — Task 1's own code (the flattening logic mirrored into the notebook) is still unit-tested locally against a duck-typed fake doctr result, but the notebook run itself is verified by inspecting its output, not by an invented assert.
>
> **This plan supersedes Tasks 6-8 of `docs/superpowers/plans/2026-07-24-museumscat-ocr-pipeline.md`.** That plan's architecture (single search-ROI crop -> `run_ocr` -> `classify_field` -> `assemble_field`) was superseded by the card-crop-manifest + per-word-prediction + card-relevance-classifier architecture built since (see `data/processed/card_crops/manifest.csv`, `predictions.csv`, `src/classify_card_relevance.py`). This plan picks up from *that* current state and builds the missing end-to-end glue.

**Goal:** Turn the existing per-word OCR predictions (`data/processed/card_crops/predictions.csv`, one row per recognized word per card crop) into a valid, scored Kaggle submission — with the two gaps that currently make this impossible closed: (1) `predictions.csv` doesn't preserve line grouping or word order, so phrases like "Svinø strand" can't be reconstructed from loose words; (2) nothing connects `manifest.csv` (card -> specimen) + `predictions.csv` (card -> words) + `data/raw/train.csv`/`test.csv` (specimen -> ground truth / submission target) into actual `verbatimDate`/`verbatimLocality` rows. Confidence calibration is explicitly deferred — see "Deferred" section.

**Architecture:** `predict_shard.py` (Kaggle-side, Task 1) is rewritten to also record each word's `block_index`/`line_index`/`word_index` and relative bounding box, instead of discarding everything but text+confidence. `src/reconstruct_lines.py` (Task 2) groups those word rows back into ordered lines per card, exactly mirroring doctr's own `page.blocks -> block.lines -> line.words` structure. `src/pipeline.py` (Task 3) classifies each line with the existing `classify_field` (date/locality/confound), keeps date/locality lines, joins same-field lines within a card, then uses the existing `assemble_field` (`src/merge_records.py`) to join same-field values across every card belonging to one specimen (per `manifest.csv`'s `source_image_file`) with `" | "`, defaulting to `"MISSING"` for specimens with no relevant cards at all. `src/generate_submission.py` (Task 4) wires that into a validated submission CSV (reusing `validate_submission` as already speced in the old plan's Task 7). `src/evaluate_local.py` (Task 5) is the actual deliverable this plan exists to produce: it runs the whole thing against the 200 labeled rows in `data/raw/train.csv` and reports real per-field mean NED and AURC using the already-built `src/metrics.py` harness — the number that says whether the fine-tune + relevance-classifier work so far is actually paying off.

**Tech Stack:** Python 3.11, conda env `museumscat-challenge`. Tasks 2-5 use only `csv`/`pandas` (already installed) — no new dependencies. Task 1's notebook cell uses `doctr`/`torch`/`cv2`, already pinned in `notebooks/kaggle_finetune_and_predict_danish.ipynb`.

## Global Constraints

- Images and any derived crops/model artifacts are **never** committed to git — `data/`, `models/` are gitignored. Only code, `kaggle_meta/` (text), small tables/figures, and CSVs under a few KB may be committed.
- Missing values must be submitted as the literal string `"MISSING"` (not empty string).
- Submission columns, exact order: `image_file,verbatimDate,verbatimDate_confidence,verbatimLocality,verbatimLocality_confidence`.
- Multi-card specimens join per-field values with `" | "`; scoring (`src.metrics.best_pipe_ned`) is permutation-invariant across pipe-separated cards, so card ordering within a specimen does not need to match ground-truth order exactly.
- Heavy/GPU-bound processing (Task 1's full-corpus OCR rerun) runs on a Kaggle notebook using the repo's established two-GPU-shard pattern — never attempted locally, never scripted to auto-submit.
- Confidence stays raw (mean of contributing words'/lines'/cards' doctr confidence) for this plan — no calibration. See "Deferred" section.

---

## Current State (as of 2026-07-29)

- [x] `src/metrics.py` — `edit_distance`, `normalize_date_punctuation`, `ned`, `best_pipe_ned`, `aurc`. Fully tested, unchanged by this plan.
- [x] `src/classify_field.py` — `classify_field(text) -> "date"|"locality"|"confound"`, keyword+regex based, tuned against real OCR errors (see `[[project-card-word-boxes-danish-finetune-result]]` memory).
- [x] `src/merge_records.py` — `assemble_field(cards: list[tuple[str, float]]) -> tuple[str, float]`, joins with `" | "` or returns `("MISSING", 0.5)` for an empty list.
- [x] `src/classify_card_relevance.py` — card-level relevant/not-relevant diagnostic (precision 0.525, recall 0.710 on the 340 labeled cards). **Not reused by this plan's pipeline** — Task 3's per-*line* classification (drop lines that classify as `"confound"`) supersedes it for the pipeline's purposes at finer granularity. `classify_card_relevance.py` stays as-is, a standalone eval tool.
- [x] `data/processed/card_crops/manifest.csv` — `crop_filename, source_image_file, card_index, x1, y1, x2, y2` (card bounding box within the original specimen photo). Covers every card crop across the full corpus (train+test), not just the 200 labeled specimens.
- [x] `data/processed/card_crops/predictions.csv` — currently `crop_filename, predicted_text, confidence` **only**, one row per recognized word, all line/order/position information already discarded by the current `predict_shard.py` (see `notebooks/kaggle_finetune_and_predict_danish.ipynb` Cell 8). **This is gap #2 — Task 1 fixes it.**
- [x] `data/raw/train.csv` — 200 labeled specimens, columns `image_file, verbatimDate, verbatimDate_confidence, verbatimLocality, verbatimLocality_confidence` (ground truth, confidence columns always 1 here). `data/raw/test.csv` — 3300 rows, `image_file` only.
- [ ] Nothing currently connects manifest -> predictions -> `train.csv`/`test.csv` rows. **Gap #1 — Tasks 3-4 fix it.**

---

## Task 1: Preserve line grouping and word position in card-crop predictions

**Files:**
- Create: `src/predict_cards.py`
- Test: `tests/test_predict_cards.py`
- Modify: `notebooks/kaggle_finetune_and_predict_danish.ipynb` (Cell 8's inline `predict_shard.py` source)

**Interfaces:**
- Produces: `rows_from_result(result, crop_filename: str) -> list[dict]`, one dict per recognized word with keys `crop_filename, block_index, line_index, word_index, predicted_text, confidence, x1, y1, x2, y2` (the last four: relative `[0,1]` bounding box from doctr's `word.geometry`). Task 2 consumes CSVs written in exactly this schema.
- Duck-types doctr's result object (`result.pages[i].blocks[j].lines[k].words[l]`, each word having `.value`, `.confidence`, `.geometry`) rather than importing doctr, so this logic is unit-testable without the (large, torch-based) doctr dependency installed locally.

- [ ] **Step 1: Write failing tests using a fake doctr-shaped result**

```python
# tests/test_predict_cards.py
from src.predict_cards import rows_from_result


class FakeWord:
    def __init__(self, value, confidence, geometry):
        self.value = value
        self.confidence = confidence
        self.geometry = geometry


class FakeLine:
    def __init__(self, words):
        self.words = words


class FakeBlock:
    def __init__(self, lines):
        self.lines = lines


class FakePage:
    def __init__(self, blocks):
        self.blocks = blocks


class FakeResult:
    def __init__(self, pages):
        self.pages = pages


def test_rows_from_result_flattens_one_word():
    result = FakeResult([FakePage([FakeBlock([FakeLine([
        FakeWord("Kb", 0.9, ((0.1, 0.2), (0.3, 0.4))),
    ])])])])
    rows = rows_from_result(result, "a.jpeg")
    assert rows == [{
        "crop_filename": "a.jpeg", "block_index": 0, "line_index": 0, "word_index": 0,
        "predicted_text": "Kb", "confidence": 0.9,
        "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4,
    }]


def test_rows_from_result_preserves_word_order_within_a_line():
    result = FakeResult([FakePage([FakeBlock([FakeLine([
        FakeWord("Svinø", 0.9, ((0.0, 0.0), (0.1, 0.1))),
        FakeWord("strand", 0.8, ((0.1, 0.0), (0.2, 0.1))),
    ])])])])
    rows = rows_from_result(result, "a.jpeg")
    assert [r["predicted_text"] for r in rows] == ["Svinø", "strand"]
    assert [r["word_index"] for r in rows] == [0, 1]


def test_rows_from_result_indexes_multiple_blocks_and_lines():
    result = FakeResult([FakePage([
        FakeBlock([FakeLine([FakeWord("A", 0.5, ((0, 0), (1, 1)))])]),
        FakeBlock([
            FakeLine([FakeWord("B", 0.5, ((0, 0), (1, 1)))]),
            FakeLine([FakeWord("C", 0.5, ((0, 0), (1, 1)))]),
        ]),
    ])])
    rows = rows_from_result(result, "a.jpeg")
    assert [(r["block_index"], r["line_index"], r["predicted_text"]) for r in rows] == [
        (0, 0, "A"), (1, 0, "B"), (1, 1, "C"),
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_predict_cards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.predict_cards'`

- [ ] **Step 3: Implement `rows_from_result`**

```python
# src/predict_cards.py
def rows_from_result(result, crop_filename):
    rows = []
    for page in result.pages:
        for block_index, block in enumerate(page.blocks):
            for line_index, line in enumerate(block.lines):
                for word_index, word in enumerate(line.words):
                    (x1, y1), (x2, y2) = word.geometry
                    rows.append({
                        "crop_filename": crop_filename,
                        "block_index": block_index,
                        "line_index": line_index,
                        "word_index": word_index,
                        "predicted_text": word.value,
                        "confidence": word.confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_predict_cards.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit the local, tested logic**

```bash
git add src/predict_cards.py tests/test_predict_cards.py
git commit -m "feat: preserve block/line/word index and geometry when flattening OCR results"
```

- [ ] **Step 6: Mirror the same logic into the notebook's inline `predict_shard.py` source**

`notebooks/kaggle_finetune_and_predict_danish.ipynb` Cell 8 currently writes a self-contained `predict_shard.py` to `/kaggle/working/` via a single `f.write("...")` call with escaped `\n`s (the notebook has no access to this repo's `src/`, so the logic must be duplicated inline, not imported — this is the existing convention for every Kaggle-side script in this notebook). Replace that cell's source with:

```python
with open("/kaggle/working/predict_shard.py", "w") as f:
    f.write('''import argparse
import os

import cv2
import pandas as pd
import torch
from doctr.datasets import VOCABS
from doctr.models import ocr_predictor, recognition

parser = argparse.ArgumentParser()
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--num-shards", type=int, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--manifest", type=str, required=True)
parser.add_argument("--images-dir", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

reco_model = recognition.__dict__["crnn_vgg16_bn"](pretrained=False, vocab=VOCABS["danish"])
reco_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
predictor = ocr_predictor(
    det_arch="db_resnet50", reco_arch=reco_model, pretrained=True, assume_straight_pages=True
).cuda()

manifest = pd.read_csv(args.manifest)
crop_filenames = manifest["crop_filename"].tolist()[args.shard::args.num_shards]

rows = []
for crop_filename in crop_filenames:
    img_bgr = cv2.imread(os.path.join(args.images_dir, crop_filename))
    if img_bgr is None:
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = predictor([img_rgb])
    for page in result.pages:
        for block_index, block in enumerate(page.blocks):
            for line_index, line in enumerate(block.lines):
                for word_index, word in enumerate(line.words):
                    (x1, y1), (x2, y2) = word.geometry
                    rows.append({
                        "crop_filename": crop_filename,
                        "block_index": block_index,
                        "line_index": line_index,
                        "word_index": word_index,
                        "predicted_text": word.value,
                        "confidence": word.confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    })

COLUMNS = ["crop_filename", "block_index", "line_index", "word_index",
           "predicted_text", "confidence", "x1", "y1", "x2", "y2"]
pd.DataFrame(rows, columns=COLUMNS).to_csv(args.output, index=False)
print(f"shard {args.shard}: wrote {len(rows)} rows for {len(crop_filenames)} cards")
''')
print("wrote /kaggle/working/predict_shard.py")
```

(Switched from the old single-escaped-line string to a triple-quoted one — same cell, same job, but a hand-edited multi-line script embedded as one `\n`-escaped line is error-prone to review; this is a readability fix, not a behavior change beyond the new columns.) Also update Cell 8's markdown header to mention block/line/word index and geometry are now preserved. Cells 8b, 9, 10, 11 need no changes — the two-shard run, the log-tailing, and the `pd.concat` merge all work unchanged with the wider column set.

- [ ] **Step 7: Commit the notebook edit**

```bash
git add notebooks/kaggle_finetune_and_predict_danish.ipynb
git commit -m "feat: record line/word order and position in Kaggle prediction script"
```

- [ ] **Step 8 (manual, on Kaggle — not run by this session): rerun the notebook's Cells 7-11**

The Stage 1/2 fine-tune checkpoints don't need retraining — only Cells 7 onward (locate card-crops dataset, write the updated `predict_shard.py`, run both GPU shards, merge). This reprocesses all ~6,500+ card crops with the existing fine-tuned checkpoint, now saving line/position. Download the resulting `/kaggle/working/predictions.csv` and overwrite `data/processed/card_crops/predictions.csv` locally (back up the old word-only file first, e.g. to `predictions.csv.bak`, since Task 5 needs the *new* format to produce a real number — the old file has no `block_index`/`line_index`/`word_index` columns and `src.pipeline.load_predictions`, Task 3, will `KeyError` on it by design, not silently degrade).

---

## Task 2: Reconstruct lines from per-word predictions

**Files:**
- Create: `src/reconstruct_lines.py`
- Test: `tests/test_reconstruct_lines.py`

**Interfaces:**
- Consumes: dicts shaped like Task 1's `rows_from_result` output (or rows read from the regenerated `predictions.csv` — same schema either way).
- Produces: `group_into_lines(word_rows: list[dict]) -> dict[str, list[dict]]` — `crop_filename -> ordered list of {"text": str, "confidence": float}` line dicts, words within a line joined by a single space in `word_index` order, lines ordered by `(block_index, line_index)` (doctr's own reading order). Task 3 consumes this directly.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reconstruct_lines.py
from src.reconstruct_lines import group_into_lines


def _row(crop_filename, block_index, line_index, word_index, text, confidence):
    return {
        "crop_filename": crop_filename, "block_index": block_index, "line_index": line_index,
        "word_index": word_index, "predicted_text": text, "confidence": confidence,
    }


def test_orders_words_within_a_line_and_averages_confidence():
    rows = [
        _row("a.jpeg", 0, 0, 1, "strand", 0.8),
        _row("a.jpeg", 0, 0, 0, "Svinø", 0.9),
    ]
    assert group_into_lines(rows) == {"a.jpeg": [{"text": "Svinø strand", "confidence": 0.85}]}


def test_orders_lines_by_block_then_line_index():
    rows = [
        _row("a.jpeg", 1, 0, 0, "second", 0.5),
        _row("a.jpeg", 0, 0, 0, "first", 0.5),
    ]
    result = group_into_lines(rows)
    assert [line["text"] for line in result["a.jpeg"]] == ["first", "second"]


def test_skips_lines_with_no_nonempty_words():
    rows = [_row("a.jpeg", 0, 0, 0, "", 0.1)]
    assert group_into_lines(rows) == {}


def test_separates_different_crops():
    rows = [_row("a.jpeg", 0, 0, 0, "Kb", 0.9), _row("b.jpeg", 0, 0, 0, "Ti", 0.7)]
    result = group_into_lines(rows)
    assert set(result.keys()) == {"a.jpeg", "b.jpeg"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_reconstruct_lines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.reconstruct_lines'`

- [ ] **Step 3: Implement `group_into_lines`**

```python
# src/reconstruct_lines.py
def group_into_lines(word_rows):
    by_crop_line = {}
    for row in word_rows:
        key = (row["crop_filename"], row["block_index"], row["line_index"])
        by_crop_line.setdefault(key, []).append(row)

    lines_by_crop = {}
    for (crop_filename, _block_index, _line_index), rows in sorted(by_crop_line.items()):
        rows = sorted(rows, key=lambda r: r["word_index"])
        words = [r["predicted_text"] for r in rows if r["predicted_text"]]
        if not words:
            continue
        text = " ".join(words)
        confidence = sum(r["confidence"] for r in rows) / len(rows)
        lines_by_crop.setdefault(crop_filename, []).append({"text": text, "confidence": confidence})
    return lines_by_crop
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_reconstruct_lines.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/reconstruct_lines.py tests/test_reconstruct_lines.py
git commit -m "feat: reconstruct reading-ordered lines from per-word predictions"
```

---

## Task 3: Per-specimen field assembly (the core pipeline)

**Files:**
- Create: `src/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `group_into_lines` (Task 2), `classify_field` (`src/classify_field.py`, already built), `assemble_field` (`src/merge_records.py`, already built).
- Produces:
  - `classify_card_field_values(lines: list[dict]) -> dict[str, tuple[str, float]]` — for one card's ordered lines, classifies each with `classify_field`, drops `"confound"`, joins same-field lines with a space, averages their confidence. Keys present only for fields with ≥1 matching line.
  - `process_specimen(crop_filenames: list[str], lines_by_crop: dict[str, list[dict]]) -> dict[str, tuple[str, float]]` — for one specimen's ordered card-crop filenames (from `manifest.csv`), returns `{"verbatimDate": (text, confidence), "verbatimLocality": (text, confidence)}`.
  - `load_manifest(path: str) -> dict[str, list[str]]` — `source_image_file -> crop_filenames` ordered by `card_index`.
  - `load_predictions(path: str) -> dict[str, list[dict]]` — reads the regenerated `predictions.csv` and returns `group_into_lines`'s output.
  - `run_pipeline(image_files: list[str], manifest_by_image: dict, lines_by_crop: dict) -> dict[str, dict]` — `image_file -> process_specimen(...)` result, for every requested `image_file` (specimens absent from the manifest default to no cards, i.e. `"MISSING"`). Task 4 calls this to build submission rows; Task 5 calls it to score against `train.csv`.

- [ ] **Step 1: Write failing tests for `classify_card_field_values`**

```python
# tests/test_pipeline.py
from src.pipeline import classify_card_field_values


def test_classify_card_field_values_joins_same_field_lines():
    lines = [{"text": "Svinø", "confidence": 0.9}, {"text": "strand", "confidence": 0.7}]
    assert classify_card_field_values(lines) == {"locality": ("Svinø strand", 0.8)}


def test_classify_card_field_values_drops_confound_lines():
    lines = [{"text": "Coll. J. P. Johansen", "confidence": 0.9}]
    assert classify_card_field_values(lines) == {}


def test_classify_card_field_values_separates_date_and_locality():
    lines = [{"text": "27.5.2022", "confidence": 0.9}, {"text": "Kb", "confidence": 0.8}]
    result = classify_card_field_values(lines)
    assert result == {"date": ("27.5.2022", 0.9), "locality": ("Kb", 0.8)}


def test_classify_card_field_values_empty_lines_is_empty_dict():
    assert classify_card_field_values([]) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 3: Implement `classify_card_field_values`**

```python
# src/pipeline.py
import csv

from src.classify_field import classify_field
from src.merge_records import assemble_field
from src.reconstruct_lines import group_into_lines

FIELD_TO_COLUMN = {"date": "verbatimDate", "locality": "verbatimLocality"}


def classify_card_field_values(lines):
    grouped = {"date": [], "locality": []}
    for line in lines:
        field = classify_field(line["text"])
        if field in grouped:
            grouped[field].append(line)

    values = {}
    for field, field_lines in grouped.items():
        if not field_lines:
            continue
        text = " ".join(line["text"] for line in field_lines)
        confidence = sum(line["confidence"] for line in field_lines) / len(field_lines)
        values[field] = (text, confidence)
    return values
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_pipeline.py -v -k classify_card_field_values`
Expected: 4 passed

- [ ] **Step 5: Write failing tests for `process_specimen`**

Update the import line at the top of `tests/test_pipeline.py` to `from src.pipeline import classify_card_field_values, process_specimen` (it isn't implemented yet — this makes Step 6's failure an `ImportError`, not a collection error masking it), then append:

```python
def test_process_specimen_joins_two_cards_for_the_same_field():
    lines_by_crop = {
        "a_card00.jpeg": [{"text": "Ti", "confidence": 0.8}],
        "a_card01.jpeg": [{"text": "Kulhuset Jaegerspris", "confidence": 0.6}],
    }
    result = process_specimen(["a_card00.jpeg", "a_card01.jpeg"], lines_by_crop)
    assert result["verbatimLocality"] == ("Ti | Kulhuset Jaegerspris", 0.7)
    assert result["verbatimDate"] == ("MISSING", 0.5)


def test_process_specimen_no_cards_is_missing_both_fields():
    result = process_specimen([], {})
    assert result["verbatimDate"] == ("MISSING", 0.5)
    assert result["verbatimLocality"] == ("MISSING", 0.5)


def test_process_specimen_ignores_crops_with_no_recorded_lines():
    result = process_specimen(["missing_from_predictions.jpeg"], {})
    assert result["verbatimDate"] == ("MISSING", 0.5)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_pipeline.py -v -k process_specimen`
Expected: FAIL with `ImportError: cannot import name 'process_specimen'`

- [ ] **Step 7: Implement `process_specimen`**

```python
def process_specimen(crop_filenames, lines_by_crop):
    per_field_cards = {"date": [], "locality": []}
    for crop_filename in crop_filenames:
        card_values = classify_card_field_values(lines_by_crop.get(crop_filename, []))
        for field, value in card_values.items():
            per_field_cards[field].append(value)
    return {
        FIELD_TO_COLUMN[field]: assemble_field(cards)
        for field, cards in per_field_cards.items()
    }
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_pipeline.py -v -k process_specimen`
Expected: 3 passed

- [ ] **Step 9: Write failing tests for `run_pipeline`**

Update the import line to `from src.pipeline import classify_card_field_values, process_specimen, run_pipeline`, then append:

```python
def test_run_pipeline_defaults_absent_specimens_to_missing():
    lines_by_crop = {"a_card00.jpeg": [{"text": "Kb", "confidence": 0.9}]}
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"]}
    result = run_pipeline(["a.jpeg", "b.jpeg"], manifest_by_image, lines_by_crop)
    assert result["a.jpeg"]["verbatimLocality"] == ("Kb", 0.9)
    assert result["b.jpeg"]["verbatimLocality"] == ("MISSING", 0.5)
```

- [ ] **Step 10: Run test to verify it fails**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_pipeline.py -v -k run_pipeline`
Expected: FAIL with `ImportError: cannot import name 'run_pipeline'`

- [ ] **Step 11: Implement `run_pipeline` and the CSV loaders**

```python
def run_pipeline(image_files, manifest_by_image, lines_by_crop):
    return {
        image_file: process_specimen(manifest_by_image.get(image_file, []), lines_by_crop)
        for image_file in image_files
    }


def load_manifest(path):
    by_image = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_image.setdefault(row["source_image_file"], []).append(
                (int(row["card_index"]), row["crop_filename"])
            )
    return {
        image_file: [crop_filename for _, crop_filename in sorted(entries)]
        for image_file, entries in by_image.items()
    }


def load_predictions(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "crop_filename": row["crop_filename"],
                "block_index": int(row["block_index"]),
                "line_index": int(row["line_index"]),
                "word_index": int(row["word_index"]),
                "predicted_text": row["predicted_text"],
                "confidence": float(row["confidence"]),
            })
    return group_into_lines(rows)
```

- [ ] **Step 12: Run the full test file**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_pipeline.py -v`
Expected: 8 passed

- [ ] **Step 13: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: assemble per-specimen date/locality fields from card-crop predictions"
```

---

## Task 4: Submission generation + validation

**Files:**
- Create: `src/generate_submission.py`
- Test: `tests/test_generate_submission.py`

**Interfaces:**
- Consumes: `run_pipeline`, `load_manifest`, `load_predictions` (Task 3).
- Produces: `REQUIRED_COLUMNS`, `validate_submission(df: pandas.DataFrame, expected_image_files: set[str]) -> list[str]` (empty list = valid), `build_submission_df(image_files: list[str], manifest_by_image: dict, lines_by_crop: dict) -> pandas.DataFrame`, and a CLI entry point `generate_submission(manifest_path, predictions_path, csv_path, output_path)` writing `data/processed/submission.csv`.

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

- [ ] **Step 3: Implement `REQUIRED_COLUMNS` and `validate_submission`**

```python
# src/generate_submission.py
import os

import pandas as pd

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_generate_submission.py -v`
Expected: 5 passed

- [ ] **Step 5: Write a failing test for `build_submission_df`**

Update the import line at the top of `tests/test_generate_submission.py` to `from src.generate_submission import REQUIRED_COLUMNS, build_submission_df, validate_submission` (it isn't implemented yet — this makes Step 6's failure an `ImportError`, not a collection error masking it), then append:

```python
def test_build_submission_df_has_one_row_per_image_file_in_required_column_order():
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg"]}
    lines_by_crop = {"a_card00.jpeg": [{"text": "Kb", "confidence": 0.9}]}
    df = build_submission_df(["a.jpeg", "b.jpeg"], manifest_by_image, lines_by_crop)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert df.set_index("image_file").loc["a.jpeg", "verbatimLocality"] == "Kb"
    assert df.set_index("image_file").loc["b.jpeg", "verbatimDate"] == "MISSING"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_generate_submission.py -v -k build_submission_df`
Expected: FAIL with `ImportError: cannot import name 'build_submission_df'`

- [ ] **Step 7: Implement `build_submission_df` and the CLI entry point**

```python
def build_submission_df(image_files, manifest_by_image, lines_by_crop):
    predicted = run_pipeline(image_files, manifest_by_image, lines_by_crop)
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

    out_df = build_submission_df(image_files, manifest_by_image, lines_by_crop)
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
```

- [ ] **Step 8: Run the full test file**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_generate_submission.py -v`
Expected: 6 passed

- [ ] **Step 9: Commit**

```bash
git add src/generate_submission.py tests/test_generate_submission.py
git commit -m "feat: add submission generation and validation"
```

---

## Task 5: Local NED/AURC evaluation against the 200 labeled specimens

**Files:**
- Create: `src/evaluate_local.py`
- Test: `tests/test_evaluate_local.py`

**Interfaces:**
- Consumes: `best_pipe_ned`, `aurc` (`src/metrics.py`), `run_pipeline`/`load_manifest`/`load_predictions` (Task 3).
- Produces: `score_predictions(predicted: dict[str, dict[str, tuple[str, float]]], ground_truth_df: pandas.DataFrame) -> dict` — `{"verbatimDate": {"mean_ned": float, "aurc": float}, "verbatimLocality": {...}}`, the pure scoring logic (unit-tested with fixtures). `main()` wires the real `data/raw/train.csv` + the regenerated `manifest.csv`/`predictions.csv` through `run_pipeline` and prints the report — this is the number the whole plan exists to produce, and it can only be trusted once Task 1's Kaggle rerun has replaced `predictions.csv` with the line-aware version.

- [ ] **Step 1: Write failing tests for `score_predictions`**

```python
# tests/test_evaluate_local.py
import pandas as pd

from src.evaluate_local import score_predictions


def test_score_predictions_perfect_match_is_zero_ned():
    predicted = {"a.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("Kb", 0.8)}}
    ground_truth = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
    ])
    report = score_predictions(predicted, ground_truth)
    assert report["verbatimDate"]["mean_ned"] == 0.0
    assert report["verbatimLocality"]["mean_ned"] == 0.0


def test_score_predictions_date_normalizes_punctuation_locality_does_not():
    predicted = {"a.jpeg": {"verbatimDate": ("1-7-2000", 0.9), "verbatimLocality": ("kb ", 0.8)}}
    ground_truth = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
    ])
    report = score_predictions(predicted, ground_truth)
    assert report["verbatimDate"]["mean_ned"] == 0.0
    assert report["verbatimLocality"]["mean_ned"] == 0.0  # case/whitespace-insensitive per src.metrics.ned


def test_score_predictions_returns_aurc_key_per_field():
    predicted = {"a.jpeg": {"verbatimDate": ("MISSING", 0.5), "verbatimLocality": ("MISSING", 0.5)}}
    ground_truth = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "MISSING", "verbatimLocality": "MISSING"},
    ])
    report = score_predictions(predicted, ground_truth)
    assert "aurc" in report["verbatimDate"]
    assert "aurc" in report["verbatimLocality"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_evaluate_local.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluate_local'`

- [ ] **Step 3: Implement `score_predictions` and `main`**

```python
# src/evaluate_local.py
import os

import pandas as pd

from src.metrics import aurc, best_pipe_ned
from src.pipeline import load_manifest, load_predictions, run_pipeline

FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}


def score_predictions(predicted, ground_truth_df):
    report = {}
    for column, is_date in FIELD_IS_DATE.items():
        errors, confidences = [], []
        for _, row in ground_truth_df.iterrows():
            text, confidence = predicted[row["image_file"]][column]
            errors.append(best_pipe_ned(text, str(row[column]), is_date=is_date))
            confidences.append(confidence)
        report[column] = {
            "mean_ned": sum(errors) / len(errors),
            "aurc": aurc(errors, confidences),
        }
    return report


def main():
    manifest_path = os.path.join("data", "processed", "card_crops", "manifest.csv")
    predictions_path = os.path.join("data", "processed", "card_crops", "predictions.csv")
    train_path = os.path.join("data", "raw", "train.csv")

    ground_truth_df = pd.read_csv(train_path)
    manifest_by_image = load_manifest(manifest_path)
    lines_by_crop = load_predictions(predictions_path)
    predicted = run_pipeline(ground_truth_df["image_file"].tolist(), manifest_by_image, lines_by_crop)

    report = score_predictions(predicted, ground_truth_df)
    for column, scores in report.items():
        print(f"{column}: mean_ned={scores['mean_ned']:.3f} aurc={scores['aurc']:.3f}")
    overall_aurc = sum(scores["aurc"] for scores in report.values()) / len(report)
    print(f"overall AURC (mean across fields, matches competition scoring): {overall_aurc:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_evaluate_local.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluate_local.py tests/test_evaluate_local.py
git commit -m "feat: add local NED/AURC evaluation against the 200 labeled specimens"
```

- [ ] **Step 6 (manual — the actual deliverable): run against the real 200-row dev set**

Requires Task 1's regenerated `predictions.csv` to already be in place (Task 1 Step 8).

Run: `conda run -n museumscat-challenge python -m src.evaluate_local`
Expected: prints `verbatimDate: mean_ned=... aurc=...`, `verbatimLocality: mean_ned=... aurc=...`, and an overall AURC. Review these numbers with the user — this is what decides whether to iterate further on OCR/relevance-classification, pursue confidence calibration (see "Deferred" below), or make a trial Kaggle submission. Don't call `kaggle competitions submit` — that's a separate, explicit, quota-costing action.

---

## Verification (end-to-end)

1. `conda run -n museumscat-challenge python -m pytest tests/ -v` — all unit tests green (new: Task 1 = 3, Task 2 = 4, Task 3 = 8, Task 4 = 6, Task 5 = 3 → 24 new, plus all pre-existing tests unaffected).
2. Task 1's Kaggle rerun completed and `data/processed/card_crops/predictions.csv` overwritten with the line-aware schema.
3. `conda run -n museumscat-challenge python -m src.evaluate_local` reviewed with the user — this is the real local AURC the whole plan exists to produce.
4. `conda run -n museumscat-challenge python -m src.generate_submission` — produces `data/processed/submission.csv` with 3300 rows (one per `data/raw/test.csv` row), passing `validate_submission`. Only run this once Step 3's dev-set numbers look reasonable — no Kaggle submission without the user's explicit go-ahead.

---

## Deferred: Confidence calibration

Not part of this plan. `[[project-danish-recognizer-finetune-decision]]` and `[[project-card-word-boxes-danish-finetune-result]]` both note the fine-tuned recognizer's raw confidence is poorly calibrated (confidently low even when correct) — AURC penalizes that. Per the user's own framing (2026-07-29): calibration only makes sense to build *after* Task 5 establishes a real baseline AURC to calibrate against — building it blind risks tuning against noise. Revisit as Task 8 of the original `2026-07-24-museumscat-ocr-pipeline.md` plan (isotonic regression via `sklearn.isotonic.IsotonicRegression`, already in `environment.yml`) once Task 5's numbers are in.
