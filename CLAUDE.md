# museumscat-challenge

Kaggle competition: https://www.kaggle.com/competitions/museumscat-specimen-collection-annotation-task
(MuseumSCAT: Specimen Collection Annotation Task@CVNH ECCV26, hosted by the Natural History Museum of Denmark)

## The task

Transcribe two fields from photos of pinned Danish Dung Beetle specimen labels (1800s-present):
`verbatimDate` and `verbatimLocality`. Only 200 of the 3500 images have ground truth
(`data/raw/train.csv`); the other 3300 are `data/raw/test.csv`.

Full task description, data format, and scoring details: `kaggle_meta/pages/*.md`
(fetched verbatim from the competition pages — `data-description.md` and `evaluation.md`
are the most important). Read those before assuming anything about edge cases
(multi-card `|`-separated specimens, `"MISSING"` vs `"NULL"`, date punctuation
normalization, danish letters æøå, "Dania" not being a locality, etc.) — they're full
of load-bearing detail that's easy to get wrong from memory.

Scoring: AURC (Area Under Risk-Coverage curve) over Normalized Edit Distance, averaged
across the two fields. Lower is better. Submissions need calibrated per-field confidence,
not just the transcription — a confident wrong answer is penalized more than an unconfident
one. Submission columns, exact order:
`image_file,verbatimDate,verbatimDate_confidence,verbatimLocality,verbatimLocality_confidence`.

## Status: work stopped here (2026-07-30)

The user decided to stop working on this competition. The full end-to-end pipeline
(detection → OCR → field classification → multi-card merge → visual relevance filter →
submission) is built and passes its test suite. Local dev-set numbers (200 labeled
specimens, see `docs/superpowers/plans/2026-07-29-submission-pipeline.md`'s "Outcome"
section for the full breakdown):

- Baseline: overall AURC 0.617 (`verbatimDate` 0.536, `verbatimLocality` 0.698).
- With the visual card-relevance filter (`src/card_relevance_visual.py`, added after the
  submission-pipeline plan): overall AURC **0.576** (`verbatimDate` 0.538, `verbatimLocality`
  0.613) — the pipeline's best local result, and what `src/generate_submission.py` uses
  by default.

`data/processed/submission.csv` (3300 rows, `validate_submission`-clean) was generated
locally but **never submitted to Kaggle** — `kaggle competitions submit` is a deliberate,
user-triggered action per this repo's conventions, and work stopped before that step.
Confidence calibration (isotonic regression, see that same doc's "Deferred" section) is
the highest-leverage next step if this resumes — both fields' AURC is well above their
NED, meaning raw confidence is poorly calibrated relative to actual correctness.

## Repo structure

Single repo, no git submodules (unlike `../bayesian-kaggle`, which this repo's scaffolding
conventions were borrowed from — same `download_kaggle_data.py` pattern, same
`kaggle_meta/pages/` convention, but flattened since this is one competition, not a
submodule-per-competition setup).

- `src/` — pipeline code, see the plan below for what lives where.
- `data/raw/`, `data/processed/`, `models/` — **gitignored**. Images (6.9GB, 3500 files)
  and anything derived from them never go to GitHub, per explicit instruction for this
  competition (unlike bayesian-kaggle, which does commit `data/raw/`). Re-download with
  `conda run -n museumscat-challenge python src/download_kaggle_data.py`.
- `kaggle_meta/` — committed. Small text: competition rules/description pages + file listing.
- `tests/` — pytest, mirrors `src/` module names.
- `docs/superpowers/plans/` — implementation plans, in chronological/supersession order:
  `2026-07-24-museumscat-ocr-pipeline.md` (original architecture, tasks 6-8 superseded) →
  `2026-07-24-danish-recognizer-finetune.md` (first recognizer fine-tune, superseded — see
  its own status note) → `2026-07-28-card-label-matching-app.md` →
  `2026-07-29-card-word-boxes-danish-finetune.md` (current recognizer) →
  `2026-07-29-submission-pipeline.md` — **read this one first**, it's the current
  end-to-end pipeline (card-crop manifest → per-word OCR → line reconstruction → field
  classification → multi-card merge → generate submission) and has the final results
  in its "Outcome" section.
- `tables_figures/eda/` — EDA outputs (grids, notes). Small/committed unless it's a
  derived-from-raw-images preview, in which case gitignore it.

## Environment

Conda env `museumscat-challenge` (`environment.yml`): Python 3.11, pandas, numpy, pillow,
scikit-learn, `kaggle`. OCR/CV libraries (opencv, doctr/easyocr) get added as the plan's
detection/OCR tasks are implemented — ask before adding, they're non-trivial (torch-based)
dependencies.

Run anything with `conda run -n museumscat-challenge python ...` rather than assuming an
activated shell.

## Constraints specific to this competition

- CC BY-NC 4.0 data license — non-commercial use only.
- No paid VLM/LLM APIs in the pipeline for now (decided 2026-07-24). The only key available
  is Anthropic (Claude Pro subscription, not necessarily API billing) — a possible Phase-2
  fallback for low-confidence crops only, kept swappable via `src/ocr_labels.py`'s
  `run_ocr(crop) -> (text, confidence)` interface, not built until the classical pipeline's
  dev-set numbers say it's needed.
- 10 submissions/day, 2 final submissions — don't call `kaggle competitions submit` from a
  script; it's a deliberate, quota-costing action the user should trigger explicitly.
- Competition ends 2026-08-31.
