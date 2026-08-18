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

## Status (2026-08-07)

**The classical OCR pipeline (detection → doctr → field classification → multi-card
merge → visual relevance filter, best local AURC 0.576) has been superseded by a
Gemini VLM per-card transcription pipeline**, which is now the primary approach and
performs an order of magnitude better. The classical pipeline's code/tests/docs
(`src/pipeline.py`, `src/classify_field.py`, `src/card_relevance_visual.py`,
`src/generate_submission.py`, `docs/superpowers/plans/2026-07-29-submission-pipeline.md`
and earlier) still exist and pass their tests, but aren't the active path — read the
Gemini-specific docs below for current work, not the classical-pipeline plans.

**Current best result:** overall dev-set AURC **0.0715** (`verbatimDate` ≈0.026,
`verbatimLocality` ≈0.117), from raw self-consistency confidence (no fitted
calibrator) plus a fixed deterministic coverage penalty. Full history and the
reasoning behind that specific config: `docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md`
(read this one first for anything confidence/calibration-related).

How the pipeline got here, in order:
1. **Per-card Gemini transcription**: `src/gemini_transcribe.py` sends each
   already-cropped card image (from the existing SAM-based detection pipeline,
   `data/processed/card_crops/`) to Gemini, sampled 3x at temperature>0, scored by
   self-consistency (fraction of samples agreeing) rather than the model's own
   self-reported confidence. Multi-card merge is unchanged (`src/merge_records.py`).
   Runners: `src/run_gemini_dev.py` (200 labeled specimens, bundles both the
   per-card pass and the whole-image fallback below in one script since the
   dev set finishes within a single quota budget), `src/run_gemini_test.py`
   (3300-image test set per-card pass — the two are split into separate
   scripts for test because this one alone spans multiple days of quota).
2. **Whole-image fallback** (`docs/superpowers/specs/2026-08-07-gemini-wholeimage-fallback-design.md`):
   for specimens where both fields came back `"MISSING"` after the per-card pass
   (failed detection, an over-eager confound filter, or a genuinely-missing label),
   send the original uncropped photo to Gemini instead. Took dev AURC from 0.124
   (per-card raw confidence alone) to 0.075 (+ cleanup + fallback).
3. **Confidence-calibration experiments** (`docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md`):
   tried soft (edit-distance) self-consistency scoring, 1-D recalibration
   (isotonic/Platt/shrinkage), a 19-feature structural+visual multi-feature
   calibrator (L2/L1 logistic regression, PCA+random-forest), and a deterministic
   penalty for specimens with under-covered cards. **Key finding, not just an
   empirical result but a mathematical one**: AURC (`aurc()` in `src/metrics.py`)
   depends only on confidence *rank order*, never on absolute values — so any
   strictly monotonic reshaping of a single scalar (which is all isotonic/Platt/
   shrinkage do to self-consistency) is structurally incapable of improving it
   beyond step-function tie-breaking noise. Every 1-D calibrator family lost to
   raw confidence in every one of the 26 grid configs tried. The fitted models
   that *do* see more information (multi-feature calibrator, Lasso, PCA+RF) came
   closer, but their edge over doing nothing was smaller than this 200-row dev
   set's fold-to-fold variance. **Winner: raw self-consistency confidence +
   `src/confidence_penalty_rules.py`'s fixed 0.7x penalty when a specimen has an
   under-covered card — zero fitted parameters, deterministic, AURC 0.0715.**

**Not yet done:** the winning config isn't wired into `src/generate_submission_gemini.py`
(the actual submission-generation path still needs it applied — see "Next steps").
Neither `data/processed/submission.csv` (classical pipeline, AURC 0.576, generated but
never submitted) nor a Gemini-pipeline test-set submission have ever been sent to
Kaggle — `kaggle competitions submit` is a deliberate, user-triggered action per this
repo's conventions.

## Status (2026-08-18): test-set transcription complete

Both Gemini transcription passes over the full 3300-image **test set** finished this
session, combining local runs with a second independent Gemini quota source running in
parallel on Kaggle (see "Kaggle quota-source infrastructure" below):

- **Per-card pass** (`src/run_gemini_test.py`): **3267/3267** relevant card crops
  transcribed into `data/processed/gemini/test_card_predictions.csv`.
- **Whole-image fallback** (`src/run_gemini_test_wholeimage.py`): **1041/1041**
  both-MISSING-after-per-card specimens transcribed into `data/processed/gemini/
  test_wholeimage_predictions.csv`.

Both files are complete and gitignored (`data/processed/`) — they exist only on this
machine (and as dataset versions on Kaggle, see below). **Not yet done:** merging these
into a submission. See "Next steps" below.

### Bug fixed this session: `AllModelsExhaustedError`

`src/gemini_batch_runner._call_with_fallback` used to `raise` the raw `ClientError`
when the *last* model in the rotation also hit its daily quota, which propagated
unhandled and crashed `run_gemini_test.py`/`run_gemini_test_wholeimage.py` with a full
traceback instead of stopping cleanly (the resumable design the rest of this file
describes was never actually reached in that case). Fixed: `_call_with_fallback` now
raises `AllModelsExhaustedError` instead, which `run_batch`/`run_wholeimage_batch` catch
and turn into a `[done] every model in the rotation hit its daily quota after N/M ...`
message before returning normally. Covered by
`tests/test_gemini_batch_runner.py::test_call_with_fallback_raises_all_models_exhausted_when_last_model_hits_daily_quota`.

### Kaggle quota-source infrastructure (built 2026-08-14/18)

To run a second, independent Gemini quota source in parallel with local runs, two
Kaggle notebooks were built and pushed as private kernels, plus three private Kaggle
Datasets (all owner `matiaspoullain`) feeding them:

- **`museumscat-card-crops`** (877MB, created 2026-08-13) — the SAM-detection card crop
  *images* (`card_crops/`) only now. Essentially static; only re-uploaded if the crops
  themselves change.
- **`museumscat-gemini-code`** (~350KB, created 2026-08-15) — this project's `src/`
  (zipped as `src.zip`; Kaggle auto-extracts a single-zip dataset *flat*, no `src/`
  subfolder, see the notebooks' Cell 4/5 comments) — split out of `museumscat-card-crops`
  specifically so a code fix doesn't require re-uploading 877MB of images. Re-version
  with `kaggle datasets version -p data/processed/gemini/kaggle_code_dataset -r zip -m "..."`
  whenever `src/` changes and a Kaggle run needs it.
- **`museumscat-gemini-progress`** (~1-2MB, created 2026-08-14) — the small, fast-moving
  progress CSVs: `test_card_predictions.csv`, `test_relevance_flags.csv`,
  `test_wholeimage_predictions.csv`. Re-version with `kaggle datasets version -p
  data/processed/gemini/kaggle_progress_dataset -m "..."` before every Kaggle run to
  seed it with the latest local (or previously-downloaded Kaggle) progress.

  **`kaggle datasets version -p <folder>` replaces a dataset's *entire* contents with
  whatever is in `<folder>` at push time** — any file missing from that folder is
  dropped from the new version. This is *why* code/progress live in their own datasets
  separate from the 877MB crops: refreshing either one only ever touches its own small
  folder, never risks the crops.

  `kaggle_code_dataset/src/` is a full copy of this repo's `src/`, gitignored under
  `data/processed/` but living on disk alongside real code — pytest's default discovery
  picked it up as a second `src` package and collided with the real one on files whose
  names end in `_test.py` (e.g. `run_gemini_test.py`), breaking `pytest` collection
  entirely. Fixed by adding `pytest.ini` (`testpaths = tests`) so collection never
  leaves `tests/` in the first place — keep that file if these staging folders keep
  existing.

- **`matiaspoullain/museumscat-run-gemini-test-kaggle-quota-source`** (kernel) — runs
  `src.run_gemini_test`. Notebook source: `notebooks/kaggle_run_gemini_test.ipynb`.
- **`matiaspoullain/museumscat-wholeimage-kaggle-quota-source`** (kernel, note the
  shorter slug — the natural `museumscat-run-gemini-test-wholeimage-...` name was
  rejected by Kaggle for exceeding the kernel slug length limit) — runs
  `src.run_gemini_test_wholeimage`. Notebook source:
  `notebooks/kaggle_run_gemini_test_wholeimage.ipynb`.

  Both kernels' `kernel-metadata.json` (staged under `data/processed/gemini/
  kaggle_kernel_push/` and `kaggle_kernel_push_wholeimage/` respectively, gitignored)
  list `dataset_sources` **without a version pin**, so a fresh run always resolves to
  each dataset's *latest* version — but an **already-open Kaggle editor tab** can stay
  pinned to whatever version it had when first opened; re-open/refresh the tab (or
  re-add the input) before relying on a freshly-pushed dataset version being used.

  **The loop for every Kaggle run, each direction:**
  - *Before* triggering a Kaggle run: copy the current local `test_card_predictions.csv`
    / `test_relevance_flags.csv` / `test_wholeimage_predictions.csv` into
    `kaggle_progress_dataset/` and `kaggle datasets version` it, so Kaggle resumes from
    the latest local progress instead of redoing already-transcribed work.
  - *After* a Kaggle run finishes (`kaggle kernels status <kernel>` shows `COMPLETE`):
    `kaggle kernels output <kernel> -p <dir>` to download it, then merge the downloaded
    CSV into the local one (union keyed by `crop_filename`/`source_image_file`, drop
    duplicates — both files should agree on overlapping rows since they're the same
    underlying transcription) *before* the next local run or the next progress-dataset
    upload, or Kaggle's contribution silently gets clobbered by a stale re-upload.
  - **This round-trip is manual and easy to skip a step of** — it's exactly what went
    wrong mid-session (a Kaggle run's 331-row contribution sat undownloaded while a
    stale 472-row progress version could have been re-uploaded over it, which would
    have silently discarded it). Always check `kaggle kernels status` before assuming
    "ran with no changes" resumes from the *latest* Kaggle-side state — Kaggle never
    remembers its own prior run's `/kaggle/working` (wiped every run); continuity only
    happens via this manual dataset round-trip.
  - Kaggle kernel pushes auto-trigger a run using whatever secrets/inputs are already
    configured on that kernel; a push-triggered auto-run has intermittently errored
    with a transient `ConnectionError`/`HTTPError` in this project's experience, while
    the user manually clicking "Save & Run All" in the editor has been reliable — prefer
    that for the actual data-producing run once the kernel exists and its `GEMINI_API_KEY`
    secret is set (secrets are per-kernel, not shared, and can't be pushed via the API).

## Next steps (updated 2026-08-18)

1. **Wire the winning confidence config into `src/generate_submission_gemini.py`**:
   `agreement="hard"` (`self_consistency_scores`, not `soft_consistency_scores`), no
   fitted calibrator (skip `src.calibrate_gemini_confidence`/`src.gemini_calibration_features`
   entirely), raw confidence + `src.confidence_penalty_rules.apply_confidence_penalty`.
   Then re-run `src.evaluate_gemini_dev` (extend it with the penalty step if it
   doesn't have it yet) to confirm the *actual submission-generation path* — not just
   `src.experiment_confidence_grid`'s harness — reproduces ~0.0715 dev AURC. This step
   was never started (see the 2026-08-07 Status section above for the calibration
   research that led to this recommended config).
2. **Generate the actual test-set submission** now that both `data/processed/gemini/
   test_card_predictions.csv` (3267/3267) and `test_wholeimage_predictions.csv`
   (1041/1041) are complete — this is new since 2026-08-07, when neither existed yet.
   Run `src/merge_records.py`-style merge + `src/generate_submission_gemini.py` (once
   step 1 is done) to produce `data/processed/submission_gemini.csv`, matching the
   exact column order from this file's top ("The task" section).
3. **`verbatimLocality` is still the dominant error source** (~0.117 dev AURC vs.
   `verbatimDate`'s ~0.026, a ~4.5x gap as of 2026-08-07) and confidence calibration is
   exhausted as a lever (rank-invariant past a point, see 2026-08-07 Status). Further
   gains need transcription/matching quality work on `verbatimLocality` specifically.
   `src/analyze_gemini_errors.py` and `src/analyze_confidence_errors.py` are the
   existing error-mining tools — this repo's convention is mining real errors first,
   not designing fixes speculatively (see `src/clean_gemini_predictions.py`'s
   docstring). With the full test set now transcribed, this analysis could also be run
   against test-set predictions, not just the 200-row dev set — more data, though no
   ground truth to score against there.
4. **`kaggle competitions submit` stays a deliberate, user-triggered action** — 10
   submissions/day, 2 final, competition ends 2026-08-31. Never call it from a script
   or without explicit instruction. Neither the classical pipeline's `submission.csv`
   nor a Gemini-pipeline submission has ever been sent to Kaggle as of 2026-08-18.

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
- `docs/superpowers/plans/` — implementation plans, in chronological/supersession order.
  Classical pipeline (superseded, see Status above, but still the record of how the
  card-detection/crop infrastructure the Gemini pipeline reuses was built):
  `2026-07-24-museumscat-ocr-pipeline.md` (original architecture, tasks 6-8 superseded) →
  `2026-07-24-danish-recognizer-finetune.md` (first recognizer fine-tune, superseded — see
  its own status note) → `2026-07-28-card-label-matching-app.md` →
  `2026-07-29-card-word-boxes-danish-finetune.md` (recognizer, no longer used for
  transcription itself but `data/processed/card_crops/` still comes from this era's
  detection work) → `2026-07-29-submission-pipeline.md` (classical end-to-end pipeline,
  final AURC 0.576 in its "Outcome" section).
  Current (Gemini VLM) pipeline — **read these for anything active**:
  `docs/superpowers/specs/2026-08-07-gemini-wholeimage-fallback-design.md` (whole-image
  fallback design) → `2026-08-07-confidence-calibration-experiments.md` (**read this one
  first** — confidence calibration experiments and the current recommended config, full
  "Outcome"/"Addendum" sections with final numbers).
- `tables_figures/eda/` — EDA outputs (grids, notes). Small/committed unless it's a
  derived-from-raw-images preview, in which case gitignore it.

## Environment

Conda env `museumscat-challenge` (`environment.yml`): Python 3.11, pandas, numpy, pillow,
scikit-learn, `kaggle`, `opencv-python-headless`, `python-doctr[torch]`, `easyocr`,
`segment-anything` (classical pipeline), and `google-genai` (current Gemini VLM pipeline).

The Gemini runners (`src/run_gemini_dev.py`, `src/run_gemini_test.py`) need
`GEMINI_API_KEY` in the environment (or `--api-key`) — a free-tier Google AI Studio key,
not Vertex AI. `src/gemini_transcribe.MODEL_FALLBACK_ORDER` lists the `-flash-lite`
models tried in order; `src/gemini_batch_runner.ModelRotation` falls forward across them
on a 429, since free-tier daily quotas are the scarce resource, not money.

Both models' free-tier quota is 15 requests/minute, on top of the daily quota —
two separate limits, and only the daily one should ever cause a model switch.
`--rpm` (default 10) drives `src.gemini_transcribe.RateLimiter`'s preemptive
per-minute throttle and is hard-capped at `src.gemini_transcribe.MAX_SAFE_RPM`
(14, a safety margin below 15) by `src.gemini_batch_runner._validate_rpm` — passing
`--rpm 15` or higher raises immediately rather than risking a 429. If a per-minute
429 slips through anyway (clock skew, a burst at the minute boundary),
`src.gemini_batch_runner.is_daily_quota_exhausted` tells it apart from a genuine
daily-quota 429 (via the error's structured `quotaId`, e.g. `...PerDay...` vs
`...PerMinute...`, falling back to message-text sniffing): a daily 429 advances
`ModelRotation` to the next model, a per-minute 429 backs off 60s and retries the
*same* model instead.

Individual requests have been observed to occasionally hang (a transport-level
stall, not a 429). `src.gemini_transcribe.REQUEST_TIMEOUT_MS` (120s) bounds
this via `HttpOptions.timeout`, so httpx cancels a stuck request instead of
blocking the batch run indefinitely; `_call_with_fallback` retries the same
model on the resulting `httpx.TimeoutException`/`ConnectError`.

Gemini also occasionally returns a transient `google.genai.errors.ServerError`
(503 UNAVAILABLE, "Unable to process input image" -- observed intermittently
during the test-set run, 2026-08-09). This isn't a quota signal, so
`_call_with_fallback` doesn't advance `ModelRotation` for it -- it backs off
`src.gemini_batch_runner.SERVER_ERROR_BACKOFF_SECONDS` (5s) and retries the
same model, same as the timeout/connect-error case above but with a short
pause first since Gemini's own error message says to retry.

Run anything with `conda run -n museumscat-challenge python ...` rather than assuming an
activated shell.

## Constraints specific to this competition

- CC BY-NC 4.0 data license — non-commercial use only.
- **Gemini API quota is the scarce resource for the VLM pipeline** (superseded the
  original 2026-07-24 "no paid VLM/LLM APIs" decision once dev-set numbers showed it was
  worth it — see Status above). Free-tier daily quotas, not billing, are the constraint:
  conserve real API calls (`src/run_gemini_dev.py`/`run_gemini_test.py`) for actual
  transcription work, not experimentation — confidence-calibration/analysis work should
  reuse already-collected `raw_samples_json` (`src.calibrate_gemini_confidence.
  rescored_predictions_by_crop`) instead of re-querying the API. Both Gemini runners are
  resumable (`already_done`, flush-per-row) if a run gets interrupted by quota exhaustion.
- 10 submissions/day, 2 final submissions — don't call `kaggle competitions submit` from a
  script; it's a deliberate, quota-costing action the user should trigger explicitly.
- Competition ends 2026-08-31.
