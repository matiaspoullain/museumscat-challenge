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

## Next steps (proposed 2026-08-07, none started yet)

1. **Wire the winning confidence config into `src/generate_submission_gemini.py`**:
   `agreement="hard"` (`self_consistency_scores`, not `soft_consistency_scores`), no
   fitted calibrator (skip `src.calibrate_gemini_confidence`/`src.gemini_calibration_features`
   entirely), raw confidence + `src.confidence_penalty_rules.apply_confidence_penalty`.
   Then re-run `src.evaluate_gemini_dev` (extend it with the penalty step if it
   doesn't have it yet) to confirm the *actual submission-generation path* — not just
   `src.experiment_confidence_grid`'s harness — reproduces ~0.0715 dev AURC.
2. **`verbatimLocality` is now the dominant error source** (~0.117 dev AURC vs.
   `verbatimDate`'s ~0.026, a ~4.5x gap) and confidence calibration is exhausted as a
   lever (see Status above — it's provably rank-invariant past a point). Further gains
   need to come from transcription/matching quality on `verbatimLocality` specifically,
   not more calibration variants. `src/analyze_gemini_errors.py` (text-diff pattern
   mining) and `src/analyze_confidence_errors.py` (high-confidence-wrong-case mining)
   are the existing tools for finding concrete, evidence-backed error patterns before
   building anything new — this repo's established convention is mining real errors
   first, not designing fixes speculatively (see `src/clean_gemini_predictions.py`'s
   docstring for why).
3. **Once (1) is done and dev numbers hold, run `src/run_gemini_test.py`** on the full
   3300-image test set — this is where free-tier Gemini quota usage becomes real.
   `src/gemini_batch_runner.ModelRotation` already handles falling forward across
   `MODEL_FALLBACK_ORDER` models on daily-quota 429s, and both runners are resumable
   (`already_done`, flush-per-row) if a run gets interrupted. Once `run_gemini_test.py`
   prints `[done] nothing to do` (every relevant test card crop transcribed), run
   `src/run_gemini_test_wholeimage.py` for the phase-2 whole-image fallback pass —
   it refuses to start early (`src.gemini_batch_runner.pending_crop_count` gate) since
   a specimen with untranscribed cards looks identical to a genuinely both-MISSING one,
   and running the fallback on it early would waste quota on a wrong classification.
4. **`kaggle competitions submit` stays a deliberate, user-triggered action** — 10
   submissions/day, 2 final, competition ends 2026-08-31. Never call it from a script
   or without explicit instruction.

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
