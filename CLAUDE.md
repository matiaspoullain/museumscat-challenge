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

**Current best *submitted* result:** public Kaggle AURC **0.05073** (`submission_gemini_rank_refined.csv`,
ref `55648362`, submitted 2026-08-20), Track A confidence rank refinement — text byte-identical
to the prior best (`submission_gemini.csv`, ref `55607015`, 0.06007), only the confidence
ranking changed. Public delta (0.06007 → 0.05073, −15.6%) confirms the dev-set signal
(0.0715 → 0.0575, −19.6%) held on the real leaderboard. Full history and reasoning:
`docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md` (base pipeline) and
`docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md` (Track A).

**Track A3 (local-VLM within-tier signal) — resumed 2026-08-20 with a cheaper sampling
config.** Would take dev AURC 0.0575 → 0.0493 once extended to the full test set. The first
Kaggle T4 GPU attempt (`matiaspoullain/museumscat-vlm-test-set-run`) was user-cancelled after
~10h at ~565/3267 items (both GPUs combined; ~29 items/hr/GPU) — at that rate full completion
was multi-day. Before relaunching, `src.analyze_local_vlm_sample_count` (zero GPU, re-derives
the majority vote from the 10 raw samples already collected for 152 dev crops, truncated to
1/3/5/10) confirmed the within-tier agree/disagree NED gradient
`src.local_vlm_signal.local_vlm_agreement_score` actually consumes — which only ever reads the
majority-vote *text*, never Qwen's self-consistency confidence — is already ~fully present at
n_samples=1 for `verbatimDate` (0.005 vs 0.037, unchanged through n=10) and still large for
`verbatimLocality` (0.039 vs 0.151 at n=1, vs 0.039 vs 0.202 at n=10 — smaller but real).
**Relaunched with `N_SAMPLES=1`, greedy decoding (`do_sample=False`, no temperature)** instead
of 10 stochastic samples at temperature 0.7 — a ~10x per-item cost cut, turning a multi-day job
into a single-session one. The 565 already-collected 10-sample items were downloaded from the
cancelled run (272+293 across the two GPUs, zero overlap), merged into
`previous_card_results.csv`, and re-uploaded to the `museumscat-vlm-test-subset` dataset so the
resumed run only spends GPU time on the ~2,700 remaining pending crops — those 565 keep their
original 10 samples (harmless; `self_consistency_scores` works on any sample-list length, and
having more samples for that subset only helps). Kernel re-pushed and running as of 2026-08-20
(status tracked via `kaggle kernels status matiaspoullain/museumscat-vlm-test-set-run`).

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

The winning config is wired into `src/generate_submission_gemini.py` (was already done
2026-08-08, re-verified 2026-08-18 — see "Next steps"), and `data/processed/
submission_gemini.csv` (3300 rows, full test set) has been generated as of 2026-08-18.
**Submitted to Kaggle 2026-08-18 (user-triggered): public leaderboard AURC 0.06007**
(vs. the classical pipeline's `data/processed/submission.csv`, submitted 2026-07-30,
public AURC 0.60501) — Kaggle submission ref `55607015`.

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
machine (and as dataset versions on Kaggle, see below). Merged into a submission on
2026-08-18. See "Next steps" below.

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

## Status (2026-08-19): local VLM second-opinion spike — not adopted

Explored whether a genuinely independent model — Qwen2.5-VL-7B-Instruct,
4-bit quantized, run entirely on Kaggle's free T4x2 GPU, no Gemini quota spent
— could serve as a second confidence "vote" alongside Gemini's self-consistency
(whose 3 samples come from one model, so its errors are correlated with each
other). Full writeup: `docs/superpowers/plans/2026-08-19-local-vlm-second-opinion-spike.md`.
Branch `spike/kaggle-vlm-second-opinion` (kept for reference, not merged).

**Result: the correlation is real but doesn't translate into a usable AURC
improvement.** At full dev-set scale (124 single-card + 71 whole-image
specimens — bigger than the 30-crop pilot that first suggested this was
promising), the local VLM's confidence is consistently informative about
whether *Gemini* is correct (Spearman -0.17 to -0.24 across both fields/both
passes) — genuinely mined error cases confirm this isn't noise (19 concrete
cases where the local VLM caught a real Gemini mistake: handwritten digit
misreads, the exact æ/ø-vs-ö confusion the prompt already warns against,
short 1-3-letter locality abbreviations Gemini discarded as `MISSING`). But
five different ways of combining the two confidences (unconditional 50/50
blend, `min()`, an æøå-triggered text-level swap, blending only where Gemini
isn't already at max confidence, blending only where Gemini is at its
*minimum* confidence) each either lost to Gemini's own confidence alone or
matched it — never beat it. The unconditional blend was actively harmful in
whole-image `verbatimLocality` (AURC 0.079 → 0.146, nearly doubling) because
most of its damage came from overriding rows where Gemini was already
maximally confident and already correct.

Not integrated anywhere in `src/`. If revisited, the working (resumable,
subprocess-isolated) Kaggle pipeline could cheaply re-run against the full
3300-image test set, or a *fitted* (not hand-picked) combiner could be tried
against the already-collected `raw_samples_json` — see the doc's "Outcome"
section for specifics. Neither is currently planned.

## Status (2026-08-20): partial-MISSING whole-image fallback spike — closed, not merged

Explored extending the whole-image fallback (which today only fires when a
specimen has *both* fields `MISSING` after the per-card pass) to specimens
with *exactly one* field `MISSING` — 514/3300 test specimens qualify. Full
writeup: `docs/superpowers/plans/2026-08-19-partial-missing-wholeimage-
fallback-spike.md`. Branch `experiment/wholeimage-partial-missing` (kept
for reference, not merged).

**Result: dev-set and test-set MISSING-rate signals were positive, but two
different confidence designs both made the real Kaggle submission worse —
progressively worse, in fact.** Dev-set AURC improved 0.0715 → 0.069 (4/4
genuine per-card misses recovered correctly on a 24-slot sample, 0
hallucinations). At full test-set scale, `verbatimLocality` recovered
47/50 (94%) of its missing slots, `verbatimDate` only 11/464 (2.4%).

1. **Hard-override design** (whole-image's own confidence taken outright):
   submitted 2026-08-20, **public AURC 0.06007 → 0.06105** (ref
   `55636785`). Diffing old vs. new submissions confirmed only the
   expected 58 rows changed (no collateral bug); manually checking 6 of
   those 58 against the actual specimen photos found 5 exact matches and
   1 partially-right-at-lower-confidence, no hallucinations.
2. **Pooled design** (whole-image's raw samples pooled with the
   specimen's card crops' raw samples, discarding "MISSING" samples
   before voting instead of taking whole-image's 3-sample confidence
   alone — architecturally sounder, since a crop's MISSING answers "no
   text in this crop", not "no text on the specimen"): initial dev-set
   evaluation on the 24-slot subset in isolation suggested a large
   improvement (AURC 0.0281 → 0.0166), but that measurement was flawed —
   AURC is a global rank metric, and a subset scored alone doesn't
   reflect how those predictions rank once embedded in the full
   submission. Re-measured correctly against the complete 200-row dev
   set: negligible difference (0.06922 → 0.06925, +0.00004). Submitted
   anyway to confirm empirically: **public AURC got worse again, by
   more — 0.06105 → 0.06263** (ref `55637181`).

Two submissions regressing in the same direction (1.6%, then 4.3% relative
vs. the 0.06007 baseline) rules out public-leaderboard sampling noise as
the likely sole explanation — something about this fallback (in both
confidence designs tried) hurts the real submission for reasons the
dev-set signal and a 6-sample manual photo audit didn't surface. Plausible
mechanism, not confirmed: the pooled design recovers *more* non-MISSING
values than the hard override (`verbatimLocality` MISSING 477 → 474 — it
can override cases where whole-image's own 3-sample majority was itself
MISSING), taking on more opportunities to be wrong exactly where AURC
penalizes hardest.

**Decision: closed, not merged, not pursued further.**
`data/processed/submission_gemini.csv` (0.06007, submission ref
`55607015`) remains the best known submission. The ~1542 Gemini calls'
raw transcriptions (with full `raw_samples_json`) are archived to a
dedicated Kaggle private dataset, `matiaspoullain/museumscat-partial-
wholeimage-archive`, in case this is revisited without re-spending quota.

## Status (2026-08-20): confidence rank refinement (Track A) — shipped, dev AURC 0.0715 → 0.0575/0.0493

Full writeup: `docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md`. Branch
`experiment/confidence-rank-refinement`.

The 2026-08-07 "calibration is exhausted" finding (AURC only depends on confidence *rank*, so
any monotone reshaping of a single scalar is provably inert) was correct but incomplete: with
`n_samples=3`, raw self-consistency takes only ~4 distinct values, so **~45-51% of all rows sit
in one tie block at confidence 1.0**, resolved by Python's stable sort — i.e. arbitrary CSV row
order. Measured: oracle reordering **within existing ties alone** (zero text changes) takes dev
AURC 0.0715 → 0.0456 (−36%), and the current tie order is statistically indistinguishable from
random. Rank order was never fully set.

**Track A (confidence-only re-ranking, shipped):**
1. `src/confidence_rank_rules.STRUCTURAL_MISSING_CONFIDENCE` — elevates `MISSING_CONFIDENCE`
   (0.5, an unjustified magic number in `src/merge_records.py`) to 0.95 specifically for
   specimens where `<2` card crops were Gemini-processed and none found a value (mean NED 0.018,
   *better* than the confidence=1.0 tier's 0.021, previously ranked below several worse tiers).
   Deliberately excludes the `>=2`-processed-all-MISSING case, which is
   `src.confidence_penalty_rules`' own risk pattern. Effect: 0.0715 → 0.0639.
2. `src.confidence_rank_rules.within_tier_score` — a bounded within-tier tiebreak
   (`combine_tier_and_within_tier_score`, `TIER_EPSILON=0.01`, can never cross a tier boundary):
   soft consistency (`src.gemini_transcribe.soft_consistency_scores`, previously unused) for
   `verbatimDate`; `avg(soft_consistency, corpus_support_score)` for `verbatimLocality`
   (`src/locality_corpus.py`: frequency of a locality string across all ~15,000 raw samples ±
   edit-distance-2 fuzzy neighbours). `date_plausibility_score` (parses as a real date) was
   built and evaluated but **rejected** — inverted risk gradient, net-negative AURC. Combined
   with (1): **dev AURC 0.0575**, verified through the real production code path.
3. `src/local_vlm_signal.py` — re-purposes the closed local-VLM spike's already-collected
   152/197 dev crops as a within-tier *agreement* signal (does Qwen2.5-VL's text match Gemini's
   final text?) instead of the spike's blend/gate designs, which both failed because the local
   VLM's information "concentrates in exactly the rows Gemini is already sure about" — i.e.
   inside the tie the spike never tested reordering. Within Gemini's top tier: agree n=21/25
   mean NED 0.000/0.039 vs. disagree n=20/23 mean NED 0.033/0.193 (date/locality) — a real, 5x
   gradient for locality. Combined with (1)+(2): **dev AURC 0.0493**.

Text is **byte-identical** to `submission_gemini.csv` throughout Track A — confirmed by direct
diff, not just by construction — since every step operates on confidence alone.
`data/processed/submission_gemini_rank_refined.csv` (A1+A2) is generated and validated
(`validate_submission` passes, confidence range checked, 1809/3300 date and 2154/3300 locality
rows changed confidence, distinct-value count jumped from 10-13 to 64-500). **Submitted to
Kaggle 2026-08-20 (user-triggered): public leaderboard AURC 0.05073** (ref `55648362`), down
from the prior best 0.06007 (ref `55607015`) — confirms the dev-set gain (0.0715 → 0.0575)
held on the real leaderboard, at a very similar relative magnitude (−15.6% public vs −19.6%
dev). This is now the best submitted result.

**A3 test-set extension: launched 2026-08-20, user-cancelled same day — paused, not
abandoned.** Dataset `matiaspoullain/museumscat-vlm-test-subset` (3,267 test card crops) +
kernel `matiaspoullain/museumscat-vlm-test-set-run` (copy of the dev pilot's resumable/
subprocess-isolated notebook) were set up and the run was started (zero Gemini quota, T4x2
GPU). After ~10h it had reached only 293/1633 items (~29 items/hr) — full completion at that
rate was multi-day, the same shape as the original Gemini test-set run but without the
Gemini-side justification of "the whole submission depends on this." User cancelled the run
2026-08-20 rather than block on it. The 293 items processed so far are preserved in the
Kaggle progress dataset (`kaggle_vlm_test_subset_dataset`) via the usual round-trip pattern —
not lost, just paused. **Not currently planned, but explicitly a candidate for a future
session**: options if revisited are (a) resume in several shorter sessions across days,
(b) scope down to only re-rank within Gemini's top confidence tier specifically (where A3's
measured dev-set gradient is strongest — 5x for `verbatimLocality`), rather than the full
1,633-item population. Once/if local-VLM test-set data exists, generate the third candidate
via `python -m src.generate_submission_gemini --local-vlm-predictions-path <downloaded CSV>`.

**Track B (text/NED reduction via MBR sample selection, HTML-entity decoding, corpus-based
locality snapping — `src/consensus_text.py`) was built, tested, and evaluated but NOT adopted.**
MBR text selection is statistically confirmed *harmful* when layered on the full confidence
stack (paired-bootstrap 90% CI excludes 0 in the harmful direction). HTML-decode + corpus-snap
trends positive but its CI includes 0, and a manual audit of the 7 dev rows it changed found 2
confirmed regressions (a correct rare spelling snapped to a more frequent wrong neighbour,
e.g. `Harboøre`→`Harboør`) alongside 5 genuine fixes — exactly the gazetteer-style risk
`src.clean_gemini_predictions` already flagged, now empirically confirmed even for a same-corpus
snap. Kept as tested dead ends so they aren't re-proposed.

Every evaluation in this round scored the **full** 200-row dev set, never a subset — the
partial-MISSING spike's isolated-24-row measurement (which looked like a 0.0281→0.0166 win but
was actually +0.00004 on the full set) is the explicit reason why.

## Next steps (updated 2026-08-18)

1. ~~Wire the winning confidence config into `src/generate_submission_gemini.py`~~
   **DONE — was actually already wired back on 2026-08-08** (commits `7bd3f36`
   "add deterministic confidence penalty for incomplete multi-card coverage" and
   `953f781` "wire coverage-penalty confidence into the Gemini submission pipeline"),
   before this "Next steps" section was last written — the "never started" claim above
   was stale. `src/generate_submission_gemini.py` already uses raw self-consistency
   confidence (no calibrator), whole-image fallback, and
   `src.confidence_penalty_rules.apply_confidence_penalty`. Re-verified 2026-08-18:
   `conda run -n museumscat-challenge python -m src.evaluate_gemini_dev`'s
   "production config" line reproduces overall dev AURC **0.071**
   (verbatimDate 0.026, verbatimLocality 0.117), matching the 2026-08-07 result.
2. **Generate the actual test-set submission** — **DONE 2026-08-18.**
   `data/processed/gemini/test_card_predictions.csv` (3267/3267, verified no
   duplicate `crop_filename`) and `test_wholeimage_predictions.csv` (1041/1041,
   verified no duplicate `source_image_file`) were complete, so
   `conda run -n museumscat-challenge python -m src.generate_submission_gemini` was
   run directly (it already does the merge internally via
   `src.calibrate_gemini_confidence.merge_predictions_by_specimen` +
   `apply_wholeimage_fallback`). Output: `data/processed/submission_gemini.csv`,
   3300 rows, exact required column order, 0 image_files missing/extra vs
   `data/raw/test.csv`, 0 null confidences. `verbatimDate` MISSING in 31.0% of rows,
   `verbatimLocality` MISSING in 15.9%; confidence range 0.233–1.000 both fields.
   **Submitted to Kaggle 2026-08-18 (user-triggered)** — public AURC **0.06007**
   (submission ref `55607015`), vs. the classical pipeline's 0.60501. 9 of 10
   daily submissions remained after this one.
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
   or without explicit instruction. As of 2026-08-18, both the classical pipeline's
   `submission.csv` (0.60501) and the Gemini pipeline's `submission_gemini.csv`
   (0.06007) have been submitted; **neither has been marked as one of the 2 final
   submissions yet** — that's a separate, irreversible action on Kaggle's side.

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
  "Outcome"/"Addendum" sections with final numbers) →
  `2026-08-19-local-vlm-second-opinion-spike.md` (local Qwen2.5-VL-7B on Kaggle GPU as an
  independent second confidence signal — not adopted *as a blend/gate*, see its "Outcome"; kept
  as a record of a real-but-then-unusable correlation and the Kaggle multi-GPU debugging journey
  to get there, on branch `spike/kaggle-vlm-second-opinion`) →
  `2026-08-20-confidence-rank-refinement.md` (**read this one for anything current** — the
  tie-breaking headroom finding, the shipped confidence rank refinement taking dev AURC
  0.0715 → 0.0575/0.0493, re-purposes the local-VLM spike above as a within-tier signal
  instead of a blend, and the Track B text-consensus techniques that were evaluated but not
  adopted; branch `experiment/confidence-rank-refinement`).
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
