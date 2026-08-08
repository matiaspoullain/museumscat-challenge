# Confidence Calibration Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** test four independent ways to improve the Gemini pipeline's confidence
*calibration* (not the transcription itself, which is already good — dev-set mean
NED 0.037/0.169) — each alone, then every valid combination — and pick whichever
scores best on honest (leakage-free, out-of-fold) dev-set AURC.

**Architecture:** each of the four ideas is a small, independently-testable unit
(a scoring function, a calibrator family, a feature builder, a deterministic rule)
that plugs into the existing `src.calibrate_gemini_confidence`/
`src.evaluate_gemini_dev` OOF-evaluation pattern. A final grid-search script
(`src.experiment_confidence_grid`) composes all four into every valid
configuration and ranks them by overall dev-set AURC — this is what answers
"which one (or combination) wins." Nothing here calls the Gemini API: proposal 1
(soft agreement) is recomputed from `raw_samples_json`, already stored in
`data/processed/gemini/dev_card_predictions.csv` from the original per-card run.

**Tech Stack:** Python, scikit-learn (`IsotonicRegression`, `LogisticRegression`,
`SimpleImputer`, `KFold`), OpenCV (existing `src.eda_card_visual_features.
extract_features`), pandas, pytest. No new dependencies.

## Global Constraints

- No Gemini API calls anywhere in this plan — everything reuses
  `data/processed/gemini/dev_card_predictions.csv` and
  `data/processed/gemini/dev_wholeimage_predictions.csv`, already on disk.
- No new dependencies (ask before adding any — CLAUDE.md).
- Run tests after every task: `conda run -n museumscat-challenge python -m pytest`.
- `conda run -n museumscat-challenge python ...` for every script invocation — no
  assumed activated shell.
- All new/changed Python: snake_case names, spaces around `=` in assignments,
  `os.path` for paths (project Python style rules).
- Functions get a docstring only where the *why* isn't obvious from the code —
  no restating what the code already says.
- Ask before committing to git is normally standing policy, but this repo's own
  session memory already established "just commit, don't ask first" here — so
  each task's commit step runs without a separate confirmation prompt.

---

## Background: the four proposals and what "best" means

Current production dev-set result (`src.evaluate_gemini_dev`, "raw confidence +
cleanup + whole-image fallback"): **overall AURC 0.075** (`verbatimDate` 0.029,
`verbatimLocality` 0.121). This is the baseline every experiment below is
measured against. Isotonic-regression calibration *already in the codebase*
currently makes AURC **worse** (0.146 vs 0.124 pre-fallback) — small-dev-set
(200 rows) isotonic regression overfitting a 4-valued discrete signal
(`self_consistency_scores` only returns 0/3, 1/3, 2/3, 3/3 with `n_samples=3`).
That's the motivating problem for proposals 1 and 2.

1. **Soft agreement** (`src.gemini_transcribe.soft_consistency_scores`) — replace
   the exact-match fraction with average sample-to-modal-answer similarity
   (`1 - normalized_edit_distance`), turning a 4-valued signal into a continuous
   one, from data already collected.
2. **Calibrator family** (`isotonic` | `platt` | `shrinkage`, generalizing
   `src.calibrate_gemini_confidence.fit_calibrators`/`oof_calibrated_confidences`)
   — isotonic is the current (underperforming) default; `platt` is genuine Platt
   scaling (logistic regression on a binarized correct/incorrect label, the
   textbook 2-parameter low-variance alternative); `shrinkage` blends the
   isotonic fit halfway back toward the raw score (fixed `alpha=0.5`, not tuned).
3. **Multi-feature calibrator** (`src.gemini_calibration_features`) — instead of
   calibrating on the single self-consistency scalar, fit an L2-regularized
   logistic regression on self-consistency plus five structural features
   (text length, cards contributing, cross-card agreement, whole-image-fallback
   origin, "has visual features") plus the 13 hand-crafted visual features
   `src.eda_card_visual_features.extract_features` already computes for the
   card-relevance classifier (color/texture/border — legibility proxies for the
   "older cards are harder to read" hypothesis).
4. **Deterministic penalty rule** (`src.confidence_penalty_rules`) — mined from
   the dev set's actual high-confidence-wrong cases (`src.analyze_confidence_errors`):
   when a specimen has ≥2 Gemini-processed card crops but at least one didn't
   contribute a value to a given field, that field's wrong-rate among
   confidence≥0.9 predictions is 34.3% (12/35) vs. 18.1% (31/171) otherwise —
   self-consistency can't see this because it only resamples one card, never
   checks coverage across cards. Fixed `confidence *= 0.7` when flagged.

"Best" = lowest **overall AURC** (mean of the two fields' AURC), computed the
same leakage-free way `src.evaluate_gemini_dev` already does: `KFold`
out-of-fold calibration, never fit and evaluated on the same rows.

---

## File Structure

- `src/gemini_transcribe.py` — **modify**: add `soft_consistency_scores`.
- `src/metrics.py` — **modify**: add `CORRECT_THRESHOLD` constant (shared by
  Platt-family binarization and the error-mining script, instead of duplicating
  the literal `0.05` in two places).
- `src/calibrate_gemini_confidence.py` — **modify**: add
  `rescored_predictions_by_crop`; generalize `fit_calibrators`/
  `oof_calibrated_confidences` with a `family` parameter.
- `src/gemini_calibration_features.py` — **new**: structural + visual feature
  extraction and the multi-feature calibrator (proposal 3). One responsibility:
  turn dev-set rows into a feature matrix and a fitted/OOF calibrator.
- `src/analyze_confidence_errors.py` — **new**: mining script for
  high-confidence-wrong cases (already run once manually for this plan's
  Background section — this task makes it a permanent, tested, re-runnable tool,
  matching `src.analyze_gemini_errors`'s existing role for text-diff mining).
- `src/confidence_penalty_rules.py` — **new**: the deterministic penalty
  (proposal 4), kept separate from the learned multi-feature calibrator so the
  two approaches are comparable, not conflated.
- `src/experiment_confidence_grid.py` — **new**: the grid harness — enumerates
  every valid combination, evaluates each via OOF AURC, prints a ranked table.
- Tests: `tests/test_gemini_transcribe.py`, `tests/test_metrics.py`,
  `tests/test_calibrate_gemini_confidence.py` (extended); `tests/test_gemini_calibration_features.py`,
  `tests/test_analyze_confidence_errors.py`, `tests/test_confidence_penalty_rules.py`,
  `tests/test_experiment_confidence_grid.py` (new).

---

### Task 1: Soft-agreement self-consistency scoring

**Files:**
- Modify: `src/gemini_transcribe.py`
- Test: `tests/test_gemini_transcribe.py`

**Interfaces:**
- Consumes: `src.metrics.ned(pred, truth, is_date=False)` (existing).
- Produces: `soft_consistency_scores(samples) -> {field: (modal_text, confidence)}`
  — same shape as the existing `self_consistency_scores`, so it's a drop-in
  alternative everywhere that function is used.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gemini_transcribe.py -- add near the existing self_consistency_scores tests
from src.gemini_transcribe import soft_consistency_scores  # add to the existing import block


def test_soft_consistency_scores_unanimous_is_full_confidence():
    samples = [{"verbatimDate": "27.5.2022", "verbatimLocality": "Kb"}] * 3
    scores = soft_consistency_scores(samples)
    assert scores["verbatimDate"] == ("27.5.2022", 1.0)
    assert scores["verbatimLocality"] == ("Kb", 1.0)


def test_soft_consistency_scores_gives_partial_credit_for_near_miss():
    # "Kob" is one edit from the modal "Kb" (ned=1/3 -> similarity=2/3), so the
    # average similarity across all 3 samples is (1 + 1 + 2/3) / 3 = 8/9 --
    # strictly higher than self_consistency_scores' exact-match fraction (2/3)
    # would give the same samples, since that scores "Kob" as zero credit.
    samples = [
        {"verbatimDate": "MISSING", "verbatimLocality": "Kb"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Kb"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Kob"},
    ]
    scores = soft_consistency_scores(samples)
    text, confidence = scores["verbatimLocality"]
    assert text == "Kb"
    assert confidence == 8 / 9


def test_soft_consistency_scores_all_disagree_is_still_above_zero():
    samples = [
        {"verbatimDate": "MISSING", "verbatimLocality": "Kb"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Ti"},
        {"verbatimDate": "MISSING", "verbatimLocality": "Rude Skov"},
    ]
    scores = soft_consistency_scores(samples)
    _, confidence = scores["verbatimLocality"]
    # "Kb" vs itself is similarity 1.0; "Ti" and "Rude Skov" share no
    # characters with "Kb" at all so their similarity is 0.0 -- average is 1/3,
    # identical to self_consistency_scores here since there's no partial overlap
    assert confidence == 1 / 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_gemini_transcribe.py -k soft_consistency -v`
Expected: FAIL with `ImportError: cannot import name 'soft_consistency_scores'`

- [ ] **Step 3: Implement `soft_consistency_scores`**

```python
# src/gemini_transcribe.py -- add import at top
from src.metrics import ned

# add directly after self_consistency_scores
def soft_consistency_scores(samples):
    """Like self_consistency_scores, but each field's confidence is the average
    similarity (1 - normalized edit distance) between every sample and the modal
    answer, instead of the fraction of samples that match it exactly. A sample
    that's a one-character OCR slip away from the modal answer ("1897" vs
    "1887") still counts as partial agreement here, rather than the zero credit
    self_consistency_scores' exact-match count would give it."""
    scores = {}
    for field in FIELDS:
        texts = [sample[field] for sample in samples]
        counts = collections.Counter(_normalize_for_matching(t) for t in texts)
        modal_normalized, _ = counts.most_common(1)[0]
        modal_text = next(t for t in texts if _normalize_for_matching(t) == modal_normalized)
        is_date = field == "verbatimDate"
        similarities = [1.0 - ned(t, modal_text, is_date=is_date) for t in texts]
        scores[field] = (modal_text, sum(similarities) / len(similarities))
    return scores
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_gemini_transcribe.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add src/gemini_transcribe.py tests/test_gemini_transcribe.py
git commit -m "feat: add soft (edit-distance) self-consistency scoring"
```

---

### Task 2: Shared correctness threshold + recompute confidence from stored samples

**Files:**
- Modify: `src/metrics.py`
- Modify: `src/calibrate_gemini_confidence.py`
- Test: `tests/test_metrics.py`, `tests/test_calibrate_gemini_confidence.py`

**Interfaces:**
- Consumes: `soft_consistency_scores`/`self_consistency_scores` (Task 1 and
  existing) as the injectable `scorer_fn`.
- Produces: `src.metrics.CORRECT_THRESHOLD` (float constant); `rescored_predictions_by_crop(path, scorer_fn) -> {crop_filename: {field: (text, confidence)}}`,
  same shape `load_predictions_by_crop` already returns — later tasks (3, 8) can
  swap it in anywhere `load_predictions_by_crop`'s output is consumed.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py -- add
from src.metrics import CORRECT_THRESHOLD  # add to existing import line


def test_correct_threshold_is_a_small_tolerance():
    assert 0.0 < CORRECT_THRESHOLD < 0.2
```

```python
# tests/test_calibrate_gemini_confidence.py -- add
import json

from src.calibrate_gemini_confidence import rescored_predictions_by_crop  # add to import block
from src.gemini_transcribe import self_consistency_scores, soft_consistency_scores


def test_rescored_predictions_by_crop_matches_direct_scorer_call():
    samples = [
        {"verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
        {"verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
        {"verbatimDate": "1.7.2000", "verbatimLocality": "Kob"},
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "predictions.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "crop_filename", "source_image_file", "verbatimDate", "verbatimDate_confidence",
                "verbatimLocality", "verbatimLocality_confidence", "models_used", "raw_samples_json",
            ])
            writer.writeheader()
            # the stored confidence columns are deliberately wrong (0.1) --
            # rescored_predictions_by_crop must ignore them and use raw_samples_json
            writer.writerow({
                "crop_filename": "a_card00.jpeg", "source_image_file": "a.jpeg",
                "verbatimDate": "1.7.2000", "verbatimDate_confidence": "0.1",
                "verbatimLocality": "Kb", "verbatimLocality_confidence": "0.1",
                "models_used": "gemini-3.5-flash-lite", "raw_samples_json": json.dumps(samples),
            })
        hard = rescored_predictions_by_crop(path, self_consistency_scores)
        soft = rescored_predictions_by_crop(path, soft_consistency_scores)

    assert hard["a_card00.jpeg"] == self_consistency_scores(samples)
    assert soft["a_card00.jpeg"] == soft_consistency_scores(samples)
    assert soft["a_card00.jpeg"]["verbatimLocality"][1] > hard["a_card00.jpeg"]["verbatimLocality"][1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py tests/test_calibrate_gemini_confidence.py -k "correct_threshold or rescored" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# src/metrics.py -- add near the top, after the imports
# An NED at or below this counts as "correct" wherever a binary correct/incorrect
# label is needed (Platt-style calibration, high-confidence-error mining) --
# tolerates trivial punctuation/whitespace noise the scorer itself doesn't
# already normalize away, without being loose enough to call a real mistake right.
CORRECT_THRESHOLD = 0.05
```

```python
# src/calibrate_gemini_confidence.py -- add "import json" to the import block, then add:

def rescored_predictions_by_crop(path, scorer_fn):
    """Re-derives {crop_filename: {field: (text, confidence)}} from each row's
    stored raw_samples_json using scorer_fn (self_consistency_scores or
    soft_consistency_scores from src.gemini_transcribe), instead of the
    confidence columns already written to the CSV -- lets a new confidence-
    scoring function be evaluated against the existing dev predictions without
    re-calling the Gemini API."""
    by_crop = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples = json.loads(row["raw_samples_json"])
            by_crop[row["crop_filename"]] = scorer_fn(samples)
    return by_crop
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_metrics.py tests/test_calibrate_gemini_confidence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py src/calibrate_gemini_confidence.py tests/test_metrics.py tests/test_calibrate_gemini_confidence.py
git commit -m "feat: recompute confidence from stored samples without new API calls"
```

---

### Task 3: Generalize calibrator family (isotonic / platt / shrinkage)

**Files:**
- Modify: `src/calibrate_gemini_confidence.py`
- Test: `tests/test_calibrate_gemini_confidence.py`

**Interfaces:**
- Consumes: `src.metrics.CORRECT_THRESHOLD` (Task 2).
- Produces: `fit_calibrators(merged, ground_truth_df, family="isotonic")`,
  `oof_calibrated_confidences(merged, ground_truth_df, family="isotonic")` —
  both existing functions, now accepting `family in ("isotonic", "platt",
  "shrinkage")`; all three return objects exposing `.predict(x) -> array`, so
  `apply_calibration` and `save_calibrators`/`load_calibrators` need no changes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibrate_gemini_confidence.py -- add
def test_fit_calibrators_platt_family_stays_in_unit_range():
    merged, ground_truth_df = _synthetic_dev_set()
    calibrators = fit_calibrators(merged, ground_truth_df, family="platt")
    for calibrator in calibrators.values():
        for raw in (0.0, 0.3, 0.7, 1.0):
            calibrated = calibrator.predict([raw])[0]
            assert 0.0 <= calibrated <= 1.0


def test_fit_calibrators_shrinkage_blends_toward_raw_score():
    merged, ground_truth_df = _synthetic_dev_set()
    isotonic_calibrators = fit_calibrators(merged, ground_truth_df, family="isotonic")
    shrinkage_calibrators = fit_calibrators(merged, ground_truth_df, family="shrinkage")
    raw = 0.2
    isotonic_pred = isotonic_calibrators["verbatimLocality"].predict([raw])[0]
    shrinkage_pred = shrinkage_calibrators["verbatimLocality"].predict([raw])[0]
    expected = 0.5 * isotonic_pred + 0.5 * raw
    assert abs(shrinkage_pred - expected) < 1e-9


def test_fit_calibrators_platt_handles_all_correct_fold_without_crashing():
    # every prediction exactly right -- the binarized label is all-1s, which a
    # real LogisticRegression can't fit a decision boundary for
    merged = {
        "a.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("Kb", 0.9)},
        "b.jpeg": {"verbatimDate": ("1.7.2000", 0.8), "verbatimLocality": ("Kb", 0.8)},
    }
    ground_truth_df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
        {"image_file": "b.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
    ])
    calibrators = fit_calibrators(merged, ground_truth_df, family="platt")
    assert calibrators["verbatimLocality"].predict([0.5])[0] == 1.0


def test_oof_calibrated_confidences_platt_family_covers_every_specimen():
    merged, ground_truth_df = _synthetic_dev_set()
    result = oof_calibrated_confidences(merged, ground_truth_df, family="platt")
    for values in result.values():
        assert len(values) == len(ground_truth_df)
        assert all(v is not None and 0.0 <= v <= 1.0 for v in values)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_calibrate_gemini_confidence.py -k "platt or shrinkage" -v`
Expected: FAIL — `fit_calibrators() got an unexpected keyword argument 'family'`

- [ ] **Step 3: Implement**

```python
# src/calibrate_gemini_confidence.py
# add to imports:
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.metrics import CORRECT_THRESHOLD  # add to the existing "from src.metrics import ..." line

# Blend weight toward the fitted isotonic curve for the "shrinkage" family;
# fixed rather than tuned -- oof_calibrated_confidences already showed plain
# isotonic (equivalent to alpha=1.0) can hurt AURC on this dev set's 200 rows,
# so a fixed halfway blend toward the raw score is a deliberately simple middle
# ground to test, not a fitted hyperparameter.
_SHRINKAGE_ALPHA = 0.5


class _ShrinkageCalibrator:
    def __init__(self, isotonic, alpha=_SHRINKAGE_ALPHA):
        self.isotonic = isotonic
        self.alpha = alpha

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return self.alpha * self.isotonic.predict(x) + (1 - self.alpha) * x


class _PlattWrapper:
    def __init__(self, logistic):
        self.logistic = logistic

    def predict(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return self.logistic.predict_proba(x)[:, 1]


class _ConstantCalibrator:
    """Fallback for a degenerate fold (every label the same) where
    LogisticRegression has no decision boundary to fit -- predicts the
    observed correct-rate for every input."""

    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(np.asarray(x).reshape(-1)), self.value)


def _fit_1d_calibrator(confidences, correctness, family):
    if family == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(confidences, correctness)
        return calibrator
    if family == "platt":
        labels = [1 if (1.0 - c) <= CORRECT_THRESHOLD else 0 for c in correctness]
        if len(set(labels)) < 2:
            return _ConstantCalibrator(sum(labels) / len(labels))
        x = np.asarray(confidences, dtype=float).reshape(-1, 1)
        logistic = LogisticRegression()
        logistic.fit(x, labels)
        return _PlattWrapper(logistic)
    if family == "shrinkage":
        isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        isotonic.fit(confidences, correctness)
        return _ShrinkageCalibrator(isotonic)
    raise ValueError(f"unknown calibrator family: {family!r}")


# replace fit_calibrators' body:
def fit_calibrators(merged, ground_truth_df, family="isotonic"):
    by_field = _confidences_and_correctness(merged, ground_truth_df)
    return {
        field: _fit_1d_calibrator(data["confidences"], data["correctness"], family)
        for field, data in by_field.items()
    }


# replace oof_calibrated_confidences' inner fit/predict loop body (the function
# signature gains `family="isotonic"`, and the two IsotonicRegression() lines
# inside the fold loop become one _fit_1d_calibrator(..., family) call):
def oof_calibrated_confidences(merged, ground_truth_df, family="isotonic"):
    by_field = _confidences_and_correctness(merged, ground_truth_df)
    result = {}
    for field, data in by_field.items():
        confidences, correctness = data["confidences"], data["correctness"]
        calibrated = [None] * len(confidences)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, test_idx in kf.split(confidences):
            calibrator = _fit_1d_calibrator(
                [confidences[i] for i in train_idx], [correctness[i] for i in train_idx], family
            )
            for i in test_idx:
                calibrated[i] = float(calibrator.predict([confidences[i]])[0])
        result[field] = calibrated
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_calibrate_gemini_confidence.py -v`
Expected: PASS (all tests in the file, including the pre-existing isotonic ones —
`family` defaults to `"isotonic"`, so the old call sites are unaffected)

- [ ] **Step 5: Commit**

```bash
git add src/calibrate_gemini_confidence.py tests/test_calibrate_gemini_confidence.py
git commit -m "feat: generalize gemini confidence calibrator to isotonic/platt/shrinkage families"
```

---

### Task 4: Structural + visual feature builder

**Files:**
- Create: `src/gemini_calibration_features.py`
- Test: `tests/test_gemini_calibration_features.py`

**Interfaces:**
- Consumes: `src.eda_card_visual_features.{FEATURE_NAMES, extract_features, IMAGES_DIR}`
  (existing); the `merged`/`predictions_by_crop`/`manifest_by_image` shapes
  already used throughout `src.calibrate_gemini_confidence`.
- Produces: `contributing_crops_by_specimen(manifest_by_image, predictions_by_crop, image_files) -> {image_file: {field: [crop_filename, ...]}}`;
  `build_feature_rows(image_files, merged, contributing_crops, predictions_by_crop, images_dir=IMAGES_DIR) -> {field: [row_dict, ...]}`,
  aligned to `image_files`' order, one `row_dict` per specimen with keys
  `FEATURE_ORDER` (module constant, defined in this task, consumed by Task 5).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gemini_calibration_features.py -- new file
import os
import tempfile

import cv2
import numpy as np

from src.gemini_calibration_features import (
    FEATURE_NAMES,
    FEATURE_ORDER,
    build_feature_rows,
    contributing_crops_by_specimen,
)


def test_contributing_crops_by_specimen_only_counts_non_missing_fields():
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.9)},
        "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.7), "verbatimLocality": ("Kb", 0.7)},
    }
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}
    result = contributing_crops_by_specimen(manifest_by_image, predictions_by_crop, ["a.jpeg"])
    assert result["a.jpeg"]["verbatimDate"] == ["a_card00.jpeg"]
    assert result["a.jpeg"]["verbatimLocality"] == ["a_card01.jpeg"]


def test_contributing_crops_by_specimen_skips_crops_never_sent_to_gemini():
    predictions_by_crop = {"a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.9)}}
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg"]}  # card01 filtered as confound, never called
    result = contributing_crops_by_specimen(manifest_by_image, predictions_by_crop, ["a.jpeg"])
    assert result["a.jpeg"]["verbatimDate"] == ["a_card00.jpeg"]


def _write_tiny_crop(images_dir, crop_filename, fill_value):
    image = np.full((40, 60, 3), fill_value, dtype=np.uint8)
    cv2.imwrite(os.path.join(images_dir, crop_filename), image)


def test_build_feature_rows_single_card_has_visual_features():
    with tempfile.TemporaryDirectory() as images_dir:
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)
        merged = {"a.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.5)}}
        predictions_by_crop = {"a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.9)}}
        contributing_crops = {"a.jpeg": {"verbatimDate": ["a_card00.jpeg"], "verbatimLocality": []}}

        rows = build_feature_rows(["a.jpeg"], merged, contributing_crops, predictions_by_crop, images_dir=images_dir)

        date_row = rows["verbatimDate"][0]
        assert date_row["has_visual_features"] == 1.0
        assert date_row["n_cards"] == 1.0
        assert date_row["cross_card_agreement"] == 0.5  # not applicable with <2 cards
        for name in FEATURE_NAMES:
            assert not np.isnan(date_row[name])

        locality_row = rows["verbatimLocality"][0]
        assert locality_row["has_visual_features"] == 0.0
        assert locality_row["n_cards"] == 0.0
        assert locality_row["text_length"] == 0.0  # "MISSING" contributes zero length
        for name in FEATURE_NAMES:
            assert np.isnan(locality_row[name])


def test_build_feature_rows_cross_card_agreement_and_disagreement():
    with tempfile.TemporaryDirectory() as images_dir:
        _write_tiny_crop(images_dir, "a_card00.jpeg", fill_value=200)
        _write_tiny_crop(images_dir, "a_card01.jpeg", fill_value=50)
        predictions_by_crop = {
            "a_card00.jpeg": {"verbatimDate": ("MISSING", 0.9), "verbatimLocality": ("Kb", 0.9)},
            "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.8), "verbatimLocality": ("Kb", 0.8)},
        }
        merged = {"a.jpeg": {"verbatimDate": ("MISSING", 0.5), "verbatimLocality": ("Kb | Kb", 0.85)}}
        contributing_crops = {"a.jpeg": {"verbatimDate": [], "verbatimLocality": ["a_card00.jpeg", "a_card01.jpeg"]}}
        rows = build_feature_rows(["a.jpeg"], merged, contributing_crops, predictions_by_crop, images_dir=images_dir)
        assert rows["verbatimLocality"][0]["cross_card_agreement"] == 1.0

        predictions_by_crop["a_card01.jpeg"]["verbatimLocality"] = ("Ti", 0.8)
        merged["a.jpeg"]["verbatimLocality"] = ("Kb | Ti", 0.85)
        rows = build_feature_rows(["a.jpeg"], merged, contributing_crops, predictions_by_crop, images_dir=images_dir)
        assert rows["verbatimLocality"][0]["cross_card_agreement"] == 0.0


def test_build_feature_rows_from_wholeimage_fallback_flag():
    merged = {"a.jpeg": {"verbatimDate": ("22.5.1977", 0.7), "verbatimLocality": ("MISSING", 0.5)}}
    contributing_crops = {"a.jpeg": {"verbatimDate": [], "verbatimLocality": []}}
    rows = build_feature_rows(["a.jpeg"], merged, contributing_crops, {}, images_dir="unused")
    # 0 contributing card crops but a non-MISSING date -- must have come from the
    # whole-image fallback, since the per-card merge alone would give "MISSING"
    assert rows["verbatimDate"][0]["from_wholeimage_fallback"] == 1.0
    assert rows["verbatimLocality"][0]["from_wholeimage_fallback"] == 0.0


def test_feature_order_starts_with_self_consistency():
    assert FEATURE_ORDER[0] == "self_consistency"
    assert set(FEATURE_ORDER) - {"self_consistency"} == set(
        ["text_length", "n_cards", "cross_card_agreement", "from_wholeimage_fallback", "has_visual_features"]
        + FEATURE_NAMES
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_gemini_calibration_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.gemini_calibration_features'`

- [ ] **Step 3: Implement**

```python
# src/gemini_calibration_features.py -- new file
"""Builds a per-(specimen, field) feature vector for a multi-feature confidence
calibrator. Self-consistency alone is a single scalar; this adds structural
signals already implicit in the merge (how many cards contributed, whether they
agreed, whether the value came from the whole-image fallback) and per-card
visual features (src.eda_card_visual_features -- color/texture/border, built
for the card-relevance classifier but equally relevant to legibility: an older,
faded card should look different from a crisp modern one, and that difference
should predict how trustworthy its transcription is)."""
import os

import cv2
import numpy as np

from src.eda_card_visual_features import FEATURE_NAMES, IMAGES_DIR, extract_features

FIELDS = ("verbatimDate", "verbatimLocality")
STRUCTURAL_FEATURE_NAMES = [
    "text_length", "n_cards", "cross_card_agreement", "from_wholeimage_fallback", "has_visual_features",
]
FEATURE_ORDER = ["self_consistency"] + STRUCTURAL_FEATURE_NAMES + FEATURE_NAMES


def contributing_crops_by_specimen(manifest_by_image, predictions_by_crop, image_files):
    """image_file -> {field: [crop_filename, ...]} for crops whose text for that
    field isn't "MISSING" -- the same filter src.calibrate_gemini_confidence.
    merge_predictions_by_specimen applies when assembling merged text, but this
    keeps the individual crop filenames instead of collapsing them into one
    string, since build_feature_rows needs the source crop images."""
    result = {}
    for image_file in image_files:
        per_field = {field: [] for field in FIELDS}
        for crop_filename in manifest_by_image.get(image_file, []):
            card = predictions_by_crop.get(crop_filename)
            if card is None:
                continue
            for field in FIELDS:
                text, _ = card[field]
                if text != "MISSING":
                    per_field[field].append(crop_filename)
        result[image_file] = per_field
    return result


def _visual_features_for_crops(crop_filenames, images_dir=IMAGES_DIR):
    """Mean of extract_features across crop_filenames. Returns (None, False)
    when there's nothing to average, so the caller fills NaN for the pipeline's
    SimpleImputer rather than guessing a default value."""
    per_crop = []
    for crop_filename in crop_filenames:
        image_bgr = cv2.imread(os.path.join(images_dir, crop_filename))
        if image_bgr is None:
            continue
        per_crop.append(extract_features(image_bgr))
    if not per_crop:
        return None, False
    means = {name: float(np.mean([f[name] for f in per_crop])) for name in FEATURE_NAMES}
    return means, True


def build_feature_rows(image_files, merged, contributing_crops, predictions_by_crop, images_dir=IMAGES_DIR):
    """One row per (image_file, field), as {feature_name: value}, aligned to
    image_files' order -- callers zip it against a same-order correctness list.
    "self_consistency" is read straight from merged[image_file][field][1]: the
    confidence merge_predictions_by_specimen already averaged across cards,
    computed from whichever scorer (hard or soft) produced the underlying
    per-card predictions -- this module doesn't need to know which."""
    rows = {field: [] for field in FIELDS}
    for image_file in image_files:
        for field in FIELDS:
            text, self_consistency = merged[image_file][field]
            crops = contributing_crops.get(image_file, {}).get(field, [])
            n_cards = len(crops)
            if n_cards >= 2:
                crop_texts = [predictions_by_crop[c][field][0] for c in crops]
                cross_card_agreement = 1.0 if len({t.strip().lower() for t in crop_texts}) == 1 else 0.0
            else:
                cross_card_agreement = 0.5  # not applicable with 0-1 contributing cards
            visual, has_visual = _visual_features_for_crops(crops, images_dir=images_dir)
            row = {
                "self_consistency": self_consistency,
                "text_length": float(len(text)) if text != "MISSING" else 0.0,
                "n_cards": float(n_cards),
                "cross_card_agreement": cross_card_agreement,
                # 0 contributing card crops but a non-MISSING value can only have
                # come from the whole-image fallback -- apply_wholeimage_fallback
                # only ever fills a field the per-card merge left "MISSING".
                "from_wholeimage_fallback": 1.0 if (n_cards == 0 and text != "MISSING") else 0.0,
                "has_visual_features": 1.0 if has_visual else 0.0,
            }
            for name in FEATURE_NAMES:
                row[name] = visual[name] if visual is not None else float("nan")
            rows[field].append(row)
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_gemini_calibration_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gemini_calibration_features.py tests/test_gemini_calibration_features.py
git commit -m "feat: build structural + visual features for a multi-feature confidence calibrator"
```

---

### Task 5: Multi-feature calibrator (fit + out-of-fold)

**Files:**
- Modify: `src/gemini_calibration_features.py`
- Test: `tests/test_gemini_calibration_features.py`

**Interfaces:**
- Consumes: `FEATURE_ORDER`, `build_feature_rows`'s row shape (Task 4);
  `src.metrics.CORRECT_THRESHOLD` (Task 2).
- Produces: `fit_multi_feature_calibrator(rows, correctness_values) -> pipeline_or_None`;
  `predict_multi_feature(pipeline, rows) -> [confidence, ...]`;
  `oof_multi_feature_calibrated_confidences(rows, correctness_values, n_folds=N_FOLDS, random_state=RANDOM_STATE) -> [confidence, ...]`,
  aligned to `rows`' order — consumed directly by Task 8's grid harness.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gemini_calibration_features.py -- add
from src.gemini_calibration_features import (
    fit_multi_feature_calibrator,
    oof_multi_feature_calibrated_confidences,
    predict_multi_feature,
)


def _synthetic_rows(n=20):
    # self_consistency and correctness move together (with noise from n_cards)
    # so a fitted logistic regression has real signal to learn, not just noise
    rows = []
    correctness = []
    for i in range(n):
        confident = i % 2 == 0
        row = {name: 0.0 for name in FEATURE_ORDER}
        row["self_consistency"] = 0.95 if confident else 0.4
        row["n_cards"] = 1.0
        row["cross_card_agreement"] = 0.5
        row["has_visual_features"] = 0.0
        for name in FEATURE_NAMES:
            row[name] = float("nan")
        rows.append(row)
        correctness.append(1.0 if confident else 0.2)
    return rows, correctness


def test_fit_multi_feature_calibrator_predicts_higher_confidence_for_higher_self_consistency():
    rows, correctness = _synthetic_rows()
    pipeline = fit_multi_feature_calibrator(rows, correctness)
    assert pipeline is not None
    high_row = dict(rows[0])   # self_consistency=0.95 case
    low_row = dict(rows[1])    # self_consistency=0.4 case
    predictions = predict_multi_feature(pipeline, [high_row, low_row])
    assert predictions[0] > predictions[1]
    assert all(0.0 <= p <= 1.0 for p in predictions)


def test_fit_multi_feature_calibrator_degenerate_labels_returns_none():
    rows, _ = _synthetic_rows(n=4)
    all_correct = [1.0, 1.0, 1.0, 1.0]  # binarizes to a single class -- no boundary to fit
    assert fit_multi_feature_calibrator(rows, all_correct) is None


def test_oof_multi_feature_calibrated_confidences_covers_every_row():
    rows, correctness = _synthetic_rows()
    result = oof_multi_feature_calibrated_confidences(rows, correctness, n_folds=4, random_state=0)
    assert len(result) == len(rows)
    assert all(v is not None and 0.0 <= v <= 1.0 for v in result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_gemini_calibration_features.py -k multi_feature_calibrator -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# src/gemini_calibration_features.py -- add to imports:
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eda_card_visual_features import N_FOLDS, RANDOM_STATE
from src.metrics import CORRECT_THRESHOLD

# add at the end of the file:

def _feature_matrix(rows, feature_order=FEATURE_ORDER):
    return np.array([[row[name] for name in feature_order] for row in rows], dtype=float)


def fit_multi_feature_calibrator(rows, correctness_values, feature_order=FEATURE_ORDER):
    """L2-regularized logistic regression on feature_order's columns, predicting
    P(correct). NaN visual features (crops with 0 contributing cards) are mean-
    imputed inside the pipeline, fit only on the training fold -- leakage-free
    the same way StandardScaler's fit is."""
    labels = [1 if (1.0 - c) <= CORRECT_THRESHOLD else 0 for c in correctness_values]
    if len(set(labels)) < 2:
        return None
    pipeline = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(), LogisticRegression(max_iter=1000)
    )
    pipeline.fit(_feature_matrix(rows, feature_order), labels)
    return pipeline


def predict_multi_feature(pipeline, rows, feature_order=FEATURE_ORDER):
    return list(pipeline.predict_proba(_feature_matrix(rows, feature_order))[:, 1])


def oof_multi_feature_calibrated_confidences(rows, correctness_values, n_folds=N_FOLDS, random_state=RANDOM_STATE):
    calibrated = [None] * len(rows)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(rows):
        train_rows = [rows[i] for i in train_idx]
        train_correctness = [correctness_values[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]
        pipeline = fit_multi_feature_calibrator(train_rows, train_correctness)
        if pipeline is None:
            # degenerate fold (every training label the same) -- predict the
            # observed correct-rate for every held-out row, same fallback
            # _ConstantCalibrator gives the 1-D "platt" family
            fallback_rate = sum(
                1 for c in train_correctness if (1.0 - c) <= CORRECT_THRESHOLD
            ) / len(train_correctness)
            predictions = [fallback_rate] * len(test_rows)
        else:
            predictions = predict_multi_feature(pipeline, test_rows)
        for i, p in zip(test_idx, predictions):
            calibrated[i] = float(p)
    return calibrated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_gemini_calibration_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gemini_calibration_features.py tests/test_gemini_calibration_features.py
git commit -m "feat: fit multi-feature confidence calibrator with out-of-fold evaluation"
```

---

### Task 6: High-confidence-error mining script

**Files:**
- Create: `src/analyze_confidence_errors.py`
- Test: `tests/test_analyze_confidence_errors.py`

**Interfaces:**
- Consumes: `src.metrics.{CORRECT_THRESHOLD, best_pipe_ned}` (existing/Task 2).
- Produces: `high_confidence_errors(merged, ground_truth_df, confidence_threshold=0.9, error_threshold=CORRECT_THRESHOLD) -> [row_dict, ...]`,
  sorted by confidence descending, each `row_dict` has keys `image_file`,
  `field`, `confidence`, `prediction`, `truth`, `error`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyze_confidence_errors.py -- new file
import pandas as pd

from src.analyze_confidence_errors import high_confidence_errors


def test_high_confidence_errors_filters_and_sorts_by_confidence_descending():
    merged = {
        "a.jpeg": {"verbatimDate": ("1.7.2000", 0.95), "verbatimLocality": ("Kb", 0.95)},   # correct, excluded
        "b.jpeg": {"verbatimDate": ("2.7.2000", 0.99), "verbatimLocality": ("Kb", 0.99)},   # date wrong+confident
        "c.jpeg": {"verbatimDate": ("1.7.2000", 0.92), "verbatimLocality": ("Ti", 0.92)},   # locality wrong+confident
        "d.jpeg": {"verbatimDate": ("9.9.9999", 0.5), "verbatimLocality": ("Rude Skov", 0.5)},  # wrong but not confident
    }
    ground_truth_df = pd.DataFrame([
        {"image_file": "a.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
        {"image_file": "b.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Kb"},
        {"image_file": "c.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Rude Skov"},
        {"image_file": "d.jpeg", "verbatimDate": "1.7.2000", "verbatimLocality": "Rude Skov"},
    ])
    rows = high_confidence_errors(merged, ground_truth_df, confidence_threshold=0.9)
    assert [(r["image_file"], r["field"]) for r in rows] == [
        ("b.jpeg", "verbatimDate"),
        ("c.jpeg", "verbatimLocality"),
    ]
    assert rows[0]["confidence"] >= rows[1]["confidence"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_analyze_confidence_errors.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/analyze_confidence_errors.py -- new file
"""Mining companion to src.analyze_gemini_errors: instead of text-diff
patterns, finds (specimen, field) pairs where confidence is high but the
transcription is still wrong -- the cases AURC punishes hardest, and the
evidence behind src.confidence_penalty_rules' fixed penalty.

Usage:
    conda run -n museumscat-challenge python -m src.analyze_confidence_errors
"""
import os

import pandas as pd

from src.calibrate_gemini_confidence import (
    apply_wholeimage_fallback,
    load_predictions_by_crop,
    load_wholeimage_predictions,
    merge_predictions_by_specimen,
)
from src.clean_gemini_predictions import clean_merged_predictions
from src.metrics import CORRECT_THRESHOLD, best_pipe_ned
from src.pipeline import load_manifest

TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}
CONFIDENCE_THRESHOLD = 0.9


def high_confidence_errors(merged, ground_truth_df, confidence_threshold=CONFIDENCE_THRESHOLD,
                            error_threshold=CORRECT_THRESHOLD):
    rows = []
    for _, r in ground_truth_df.iterrows():
        for field, is_date in FIELD_IS_DATE.items():
            text, confidence = merged[r["image_file"]][field]
            error = best_pipe_ned(text, str(r[field]), is_date=is_date)
            if confidence >= confidence_threshold and error > error_threshold:
                rows.append({
                    "image_file": r["image_file"], "field": field, "confidence": confidence,
                    "prediction": text, "truth": str(r[field]), "error": error,
                })
    rows.sort(key=lambda row: -row["confidence"])
    return rows


def main():
    ground_truth_df = pd.read_csv(TRAIN_CSV_PATH)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    predictions_by_crop = load_predictions_by_crop()
    merged = merge_predictions_by_specimen(
        predictions_by_crop, manifest_by_image, ground_truth_df["image_file"].tolist()
    )
    cleaned = clean_merged_predictions(merged)
    wholeimage_predictions = load_wholeimage_predictions()
    with_fallback = clean_merged_predictions(apply_wholeimage_fallback(cleaned, wholeimage_predictions))

    rows = high_confidence_errors(with_fallback, ground_truth_df)
    print(f"{len(rows)} high-confidence (>={CONFIDENCE_THRESHOLD}) wrong cases")
    for row in rows:
        print(f"{row['image_file']} {row['field']} conf={row['confidence']:.2f} "
              f"pred={row['prediction']!r} truth={row['truth']!r} err={row['error']:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_analyze_confidence_errors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analyze_confidence_errors.py tests/test_analyze_confidence_errors.py
git commit -m "feat: add high-confidence-error mining script for gemini dev predictions"
```

---

### Task 7: Deterministic confidence penalty rule

**Files:**
- Create: `src/confidence_penalty_rules.py`
- Test: `tests/test_confidence_penalty_rules.py`

**Interfaces:**
- Consumes: `manifest_by_image`/`predictions_by_crop` shapes (existing).
- Produces: `count_processed_and_contributing(image_file, field, manifest_by_image, predictions_by_crop) -> (n_processed, n_contributing)`;
  `apply_confidence_penalty(confidence, n_processed, n_contributing, penalty_factor=PENALTY_FACTOR) -> float`
  — both consumed directly by Task 8's grid harness.

This task's rule was already mined for this plan's Background section by running
`src.analyze_confidence_errors` against `data/processed/gemini/dev_card_predictions.csv`
and cross-referencing the flagged pattern's wrong-rate: **34.3% (12/35) flagged
vs. 18.1% (31/171) unflagged**, among dev predictions with confidence ≥ 0.9.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_confidence_penalty_rules.py -- new file
from src.confidence_penalty_rules import apply_confidence_penalty, count_processed_and_contributing


def test_count_processed_and_contributing_counts_only_gemini_processed_crops():
    predictions_by_crop = {
        "a_card00.jpeg": {"verbatimDate": ("1.7.2000", 0.9), "verbatimLocality": ("MISSING", 0.9)},
        "a_card01.jpeg": {"verbatimDate": ("MISSING", 0.7), "verbatimLocality": ("Kb", 0.7)},
    }
    # card02 is in the manifest (a detected crop) but never sent to Gemini
    # (filtered as a confound by the relevance classifier) -- must not count
    manifest_by_image = {"a.jpeg": ["a_card00.jpeg", "a_card01.jpeg", "a_card02.jpeg"]}
    n_processed, n_contributing = count_processed_and_contributing(
        "a.jpeg", "verbatimDate", manifest_by_image, predictions_by_crop
    )
    assert (n_processed, n_contributing) == (2, 1)


def test_apply_confidence_penalty_flags_incomplete_multi_card_coverage():
    # 2 processed cards, only 1 contributed to this field -- flagged
    penalized = apply_confidence_penalty(1.0, n_processed=2, n_contributing=1)
    assert penalized == 0.7


def test_apply_confidence_penalty_all_cards_contributed_not_flagged():
    unpenalized = apply_confidence_penalty(1.0, n_processed=2, n_contributing=2)
    assert unpenalized == 1.0


def test_apply_confidence_penalty_single_card_missing_field_not_flagged():
    # this is the ordinary "this card just has no date" case, not a coverage risk
    unpenalized = apply_confidence_penalty(0.9, n_processed=1, n_contributing=0)
    assert unpenalized == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_confidence_penalty_rules.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/confidence_penalty_rules.py -- new file
"""Deterministic confidence penalty for a structural risk pattern mined via
src.analyze_confidence_errors on the dev set: when a specimen has >=2 Gemini-
processed card crops but at least one of them didn't contribute a value to a
given field, that field's transcription is at real risk of having missed a
second card's value entirely -- self-consistency alone can't see this, since it
only measures agreement across resamples of the SAME card, never coverage
across cards. Among dev-set predictions with confidence >= 0.9, this pattern's
wrong-rate is 34.3% (12/35) vs. 18.1% (31/171) when it doesn't apply -- roughly
1.9x (mined 2026-08-07, see docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md).

Kept as a fixed multiplicative penalty rather than folded into
src.gemini_calibration_features' learned multi-feature calibrator, so the two
approaches (hardcoded rule vs. fit-on-dev-data) can be compared head to head."""
PENALTY_FACTOR = 0.7


def count_processed_and_contributing(image_file, field, manifest_by_image, predictions_by_crop):
    n_processed = 0
    n_contributing = 0
    for crop_filename in manifest_by_image.get(image_file, []):
        card = predictions_by_crop.get(crop_filename)
        if card is None:
            continue
        n_processed += 1
        text, _ = card[field]
        if text != "MISSING":
            n_contributing += 1
    return n_processed, n_contributing


def apply_confidence_penalty(confidence, n_processed, n_contributing, penalty_factor=PENALTY_FACTOR):
    flagged = n_processed >= 2 and n_contributing < n_processed
    return confidence * penalty_factor if flagged else confidence
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_confidence_penalty_rules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/confidence_penalty_rules.py tests/test_confidence_penalty_rules.py
git commit -m "feat: add deterministic confidence penalty for incomplete multi-card coverage"
```

---

### Task 8: Grid-search harness

**Files:**
- Create: `src/experiment_confidence_grid.py`
- Test: `tests/test_experiment_confidence_grid.py`

**Interfaces:**
- Consumes: everything from Tasks 1-7 — `soft_consistency_scores`/`self_consistency_scores`
  (Task 1), `rescored_predictions_by_crop` (Task 2), `oof_calibrated_confidences(..., family=...)`
  (Task 3), `contributing_crops_by_specimen`/`build_feature_rows`/`oof_multi_feature_calibrated_confidences`
  (Tasks 4-5), `count_processed_and_contributing`/`apply_confidence_penalty` (Task 7).
- Produces: `enumerate_configs() -> [config_dict, ...]` (16 entries);
  `evaluate_config(config, context) -> (overall_aurc, per_field_aurc_dict)`;
  `main()` prints the full ranked table — this is the script Task 9 runs for real.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_experiment_confidence_grid.py -- new file
import os
import tempfile

import cv2
import numpy as np
import pandas as pd

from src.experiment_confidence_grid import enumerate_configs, evaluate_config
from src.gemini_calibration_features import contributing_crops_by_specimen


def test_enumerate_configs_has_sixteen_valid_combinations():
    configs = enumerate_configs()
    assert len(configs) == 16
    # every (agreement, feature_set, penalty) triple appears, and calibrator_family
    # is only meaningful (i.e. varies) when feature_set == "scalar"
    scalar_families = {c["calibrator_family"] for c in configs if c["feature_set"] == "scalar"}
    assert scalar_families == {"isotonic", "platt", "shrinkage"}
    multi_families = {c["calibrator_family"] for c in configs if c["feature_set"] == "multi"}
    assert multi_families == {"logistic_multi"}


def _tiny_context(tmp_dir):
    images_dir = os.path.join(tmp_dir, "images")
    os.makedirs(images_dir)
    manifest_by_image = {}
    predictions_by_crop = {"hard": {}, "soft": {}}
    rows = []
    for i in range(12):
        image_file = f"s{i}.jpeg"
        crop = f"s{i}_card00.jpeg"
        cv2.imwrite(os.path.join(images_dir, crop), np.full((30, 40, 3), 100 + i, dtype=np.uint8))
        manifest_by_image[image_file] = [crop]
        correct = i % 2 == 0
        text = "1.7.2000" if correct else "9.9.9999"
        predictions_by_crop["hard"][crop] = {
            "verbatimDate": (text, 0.9 if correct else 0.3),
            "verbatimLocality": ("MISSING", 0.5),
        }
        predictions_by_crop["soft"][crop] = predictions_by_crop["hard"][crop]
        rows.append({"image_file": image_file, "verbatimDate": "1.7.2000", "verbatimLocality": "MISSING"})
    ground_truth_df = pd.DataFrame(rows)
    image_files = ground_truth_df["image_file"].tolist()
    contributing_crops = contributing_crops_by_specimen(manifest_by_image, predictions_by_crop["hard"], image_files)
    return {
        "ground_truth_df": ground_truth_df, "manifest_by_image": manifest_by_image,
        "image_files": image_files, "predictions_by_crop": predictions_by_crop,
        "wholeimage_predictions": {}, "contributing_crops": contributing_crops,
        "images_dir": images_dir,
    }


def test_evaluate_config_runs_for_every_combination():
    with tempfile.TemporaryDirectory() as tmp_dir:
        context = _tiny_context(tmp_dir)
        for config in enumerate_configs():
            overall, per_field = evaluate_config(config, context)
            assert 0.0 <= overall <= 1.0
            assert set(per_field) == {"verbatimDate", "verbatimLocality"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_experiment_confidence_grid.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/experiment_confidence_grid.py -- new file
"""Grid-evaluates every combination of this repo's four confidence-calibration
experiments (docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md)
against honest (leakage-free, out-of-fold) dev-set AURC, to pick which
combination (if any) beats the current production config -- agreement="hard",
feature_set="scalar", calibrator_family="isotonic", penalty=False is NOT the
production default (production actually skips calibration's isotonic step in
favor of raw confidence, per src.evaluate_gemini_dev's best-performing block) --
this grid's job is to find the best config among ones that DO calibrate.

Usage:
    conda run -n museumscat-challenge python -m src.experiment_confidence_grid
"""
import os

import pandas as pd

from src.calibrate_gemini_confidence import (
    apply_wholeimage_fallback,
    load_predictions_by_crop,
    load_wholeimage_predictions,
    merge_predictions_by_specimen,
    oof_calibrated_confidences,
    rescored_predictions_by_crop,
)
from src.clean_gemini_predictions import clean_merged_predictions
from src.confidence_penalty_rules import apply_confidence_penalty, count_processed_and_contributing
from src.eda_card_visual_features import IMAGES_DIR
from src.gemini_calibration_features import (
    build_feature_rows,
    contributing_crops_by_specimen,
    oof_multi_feature_calibrated_confidences,
)
from src.gemini_transcribe import self_consistency_scores, soft_consistency_scores
from src.metrics import aurc, best_pipe_ned
from src.pipeline import load_manifest

TRAIN_CSV_PATH = os.path.join("data", "raw", "train.csv")
MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
DEV_PREDICTIONS_PATH = os.path.join("data", "processed", "gemini", "dev_card_predictions.csv")
FIELD_IS_DATE = {"verbatimDate": True, "verbatimLocality": False}
SCORERS = {"hard": self_consistency_scores, "soft": soft_consistency_scores}
SCALAR_CALIBRATOR_FAMILIES = ("isotonic", "platt", "shrinkage")
BASELINE_AURC = 0.075  # src.evaluate_gemini_dev's current best (raw confidence + cleanup + fallback)


def enumerate_configs():
    """Every valid (agreement, feature_set, calibrator_family, penalty)
    combination: 16 total. calibrator_family only varies when feature_set ==
    "scalar" (isotonic/platt/shrinkage are all 1-D fits over the self-
    consistency scalar); feature_set == "multi" always uses the multi-feature
    logistic regression from src.gemini_calibration_features, so
    calibrator_family is fixed to "logistic_multi" there instead."""
    configs = []
    for agreement in ("hard", "soft"):
        for penalty in (False, True):
            for family in SCALAR_CALIBRATOR_FAMILIES:
                configs.append({
                    "agreement": agreement, "feature_set": "scalar",
                    "calibrator_family": family, "penalty": penalty,
                })
            configs.append({
                "agreement": agreement, "feature_set": "multi",
                "calibrator_family": "logistic_multi", "penalty": penalty,
            })
    return configs


def _label(config):
    return (f"agreement={config['agreement']:<4} feature_set={config['feature_set']:<6} "
            f"family={config['calibrator_family']:<14} penalty={config['penalty']}")


def _prepare_context(images_dir=IMAGES_DIR):
    ground_truth_df = pd.read_csv(TRAIN_CSV_PATH)
    manifest_by_image = load_manifest(MANIFEST_PATH)
    image_files = ground_truth_df["image_file"].tolist()
    predictions_by_crop = {
        "hard": load_predictions_by_crop(DEV_PREDICTIONS_PATH),
        "soft": rescored_predictions_by_crop(DEV_PREDICTIONS_PATH, soft_consistency_scores),
    }
    contributing_crops = contributing_crops_by_specimen(manifest_by_image, predictions_by_crop["hard"], image_files)
    return {
        "ground_truth_df": ground_truth_df, "manifest_by_image": manifest_by_image,
        "image_files": image_files, "predictions_by_crop": predictions_by_crop,
        "wholeimage_predictions": load_wholeimage_predictions(), "contributing_crops": contributing_crops,
        "images_dir": images_dir,
    }


def _with_fallback(predictions_by_crop_variant, context):
    merged = merge_predictions_by_specimen(
        predictions_by_crop_variant, context["manifest_by_image"], context["image_files"]
    )
    cleaned = clean_merged_predictions(merged)
    return clean_merged_predictions(apply_wholeimage_fallback(cleaned, context["wholeimage_predictions"]))


def _errors_by_field(with_fallback, ground_truth_df):
    errors = {field: [] for field in FIELD_IS_DATE}
    for _, r in ground_truth_df.iterrows():
        for field, is_date in FIELD_IS_DATE.items():
            text, _ = with_fallback[r["image_file"]][field]
            errors[field].append(best_pipe_ned(text, str(r[field]), is_date=is_date))
    return errors


def evaluate_config(config, context):
    predictions_by_crop_variant = context["predictions_by_crop"][config["agreement"]]
    with_fallback = _with_fallback(predictions_by_crop_variant, context)
    ground_truth_df = context["ground_truth_df"]
    image_files = context["image_files"]
    errors_by_field = _errors_by_field(with_fallback, ground_truth_df)

    confidences_by_field = {}
    if config["feature_set"] == "scalar":
        confidences_by_field = oof_calibrated_confidences(
            with_fallback, ground_truth_df, family=config["calibrator_family"]
        )
    else:
        for field in FIELD_IS_DATE:
            rows = build_feature_rows(
                image_files, with_fallback, context["contributing_crops"], predictions_by_crop_variant,
                images_dir=context["images_dir"],
            )[field]
            correctness = [1.0 - e for e in errors_by_field[field]]
            confidences_by_field[field] = oof_multi_feature_calibrated_confidences(rows, correctness)

    if config["penalty"]:
        for field in FIELD_IS_DATE:
            penalized = []
            for i, image_file in enumerate(image_files):
                n_proc, n_contrib = count_processed_and_contributing(
                    image_file, field, context["manifest_by_image"], predictions_by_crop_variant
                )
                penalized.append(apply_confidence_penalty(confidences_by_field[field][i], n_proc, n_contrib))
            confidences_by_field[field] = penalized

    aurcs = {field: aurc(errors_by_field[field], confidences_by_field[field]) for field in FIELD_IS_DATE}
    return sum(aurcs.values()) / len(aurcs), aurcs


def main():
    context = _prepare_context()
    results = []
    for config in enumerate_configs():
        overall, per_field = evaluate_config(config, context)
        results.append((overall, per_field, config))
    results.sort(key=lambda r: r[0])

    print(f"{'overall_aurc':>12}  {'date_aurc':>10}  {'locality_aurc':>14}  config")
    for overall, per_field, config in results:
        print(f"{overall:12.4f}  {per_field['verbatimDate']:10.4f}  "
              f"{per_field['verbatimLocality']:14.4f}  {_label(config)}")

    best_overall, best_per_field, best_config = results[0]
    print(f"\nbest: {_label(best_config)} -> overall {best_overall:.4f} "
          f"(current best baseline {BASELINE_AURC:.4f}, delta {best_overall - BASELINE_AURC:+.4f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge python -m pytest tests/test_experiment_confidence_grid.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `conda run -n museumscat-challenge python -m pytest -v`
Expected: PASS (every test in the repo, not just this plan's new/changed files)

- [ ] **Step 6: Commit**

```bash
git add src/experiment_confidence_grid.py tests/test_experiment_confidence_grid.py
git commit -m "feat: add grid harness to compare confidence-calibration experiment combinations"
```

---

### Task 9: Run the real grid on the dev set and record the outcome

**Files:**
- Modify: `docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md` (this file — append an "Outcome" section)

**Interfaces:**
- Consumes: `src.experiment_confidence_grid.main` (Task 8), the real
  `data/processed/gemini/dev_card_predictions.csv` and
  `data/processed/gemini/dev_wholeimage_predictions.csv`.
- Produces: nothing new for later tasks — this is the plan's terminal
  "get the real answer" step.

- [ ] **Step 1: Run the grid for real and capture output**

Run: `conda run -n museumscat-challenge python -m src.experiment_confidence_grid | tee /tmp/confidence_grid_results.txt`

This takes longer than the unit tests (real OpenCV feature extraction over up to
~382 dev-set card crops, ×2 for hard/soft, ×however many "multi" configs need
it — still well under a minute, no network calls).

- [ ] **Step 2: Append an "## Outcome" section to this plan file**

Read `/tmp/confidence_grid_results.txt` and add a section at the very end of
this file with:
- The full ranked table (or its top 5 and bottom 3 rows if very long).
- Which single-toggle change helped most/least in isolation (compare each
  config that differs from the all-off baseline — `agreement=hard
  feature_set=scalar family=isotonic penalty=False` — by exactly one field).
- The winning combination and its delta vs. the 0.075 current-best baseline.
- Whether the winner beats 0.075 at all — if not, say so plainly; a negative
  result (none of these four ideas help) is still the honest answer to "which
  is best," and this repo's own convention (`src.evaluate_gemini_dev`'s
  reports, `clean_gemini_predictions.py`'s docstring) is to report deltas
  against baseline either way rather than only when they're favorable.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md
git commit -m "docs: record confidence-calibration experiment grid results"
```

---

## What this plan deliberately doesn't do

Wiring the winning configuration into `src.generate_submission_gemini` (the
actual submission-generation path) is a follow-up step, not part of this plan —
matches this repo's established gate (`docs/superpowers/plans/2026-08-07-gemini-wholeimage-fallback-design.md`:
"only wired to a real value once evaluate_gemini_dev confirms it helps"). Decide
that once Task 9's real numbers are in.
