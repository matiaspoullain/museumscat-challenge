# museumscat-challenge

Kaggle competition: https://www.kaggle.com/competitions/museumscat-specimen-collection-annotation-task
(MuseumSCAT: Specimen Collection Annotation Task@CVNH ECCV26, hosted by the Natural History Museum of Denmark)

## The task

Transcribe two fields from photos of pinned Danish Dung Beetle specimen labels (1800s-present):
`verbatimDate` and `verbatimLocality`. Only 200 of the 3500 images have ground truth
(`data/raw/train.csv`); the other 3300 are `data/raw/test.csv`.

**For the pipeline's shape at a glance — inputs, which script writes which file, and where
text and confidence come from — see the Mermaid diagram in `README.md`.** It is kept accurate
against the code: it shows only what the shipped route actually reads.

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

**Competition ended 2026-08-31. Final result: private leaderboard AURC 0.02345, rank 15th of
127 teams.** See the 2026-09-02 Status section near the end of this file for the full result
and how it was read off Kaggle.

**Best *submitted* result during the competition:** public Kaggle AURC **0.02382**
(`submission_gemini_wholeimage_primary.csv`, ref `55722819`, submitted 2026-08-23) — the
**whole-image-primary route** (which replaced the per-card SAM-crop route as the text source)
plus **risk-ordered route tiers**. This is also the submission that ended up scoring best on
the private leaderboard (0.02345) and decided the final rank. Rank 10th on the public board at
submission time (the public board kept moving as other teams submitted through 2026-08-31; see
below for the final private-leaderboard rank). See the 2026-08-23 Status sections below for how
it was built.

Full submission history: `submission.csv` (classical pipeline) 0.60501 → `submission_gemini.csv`
0.06007 → `submission_gemini_rank_refined.csv` (Track A) 0.05073 →
`submission_gemini_rank_refined_local_vlm.csv` (Track A3) 0.04423 →
`submission_gemini_wholeimage_primary.csv` (D4) 0.02530 → same file with risk-ordered route
tiers (Phase 6) **0.02382**. Every step's public delta confirmed
its dev-set signal held on the real leaderboard (Track A: 0.06007→0.05073 public vs 0.0715→0.0575
dev; Track A3: 0.05073→0.04423 public, −12.8%, vs 0.0575→0.0493 dev, −14.3%; D4: 0.04423→0.02530
public, **−42.8%**, vs 0.0487→0.0353 dev, −27.5% — the only step where the public gain *exceeded*
the dev projection, and also the only step that changed the transcribed **text** rather than
confidence rank alone). Full history and reasoning:
`docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md` (base pipeline),
`2026-08-20-confidence-rank-refinement.md` (Track A and A3), and
`2026-08-21-transcription-quality-push.md` (D4, current).

**Track A3 test-set extension — complete.** The first Kaggle T4 GPU attempt
(`matiaspoullain/museumscat-vlm-test-set-run`) was user-cancelled after ~10h at ~565/3267 items
(both GPUs combined, 10 stochastic samples/item at temperature 0.7; ~29 items/hr/GPU) — at that
rate full completion was multi-day. Before relaunching, `src.old.analyze_local_vlm_sample_count`
(zero GPU, re-derives the majority vote from the 10 raw samples already collected for 152 dev
crops, truncated to 1/3/5/10) confirmed the within-tier agree/disagree NED gradient
`src.local_vlm_signal.local_vlm_agreement_score` actually consumes — which only ever reads the
majority-vote *text*, never Qwen's self-consistency confidence — is already ~fully present at
n_samples=1 for `verbatimDate` (0.005 vs 0.037, unchanged through n=10) and still large for
`verbatimLocality` (0.039 vs 0.151 at n=1, vs 0.039 vs 0.202 at n=10 — smaller but real).
Relaunched with `N_SAMPLES=1`, greedy decoding (`do_sample=False`, no temperature) instead of
10 stochastic samples — the 565 already-collected 10-sample items were rescued (downloaded,
merged, re-uploaded as resume state; kept as-is rather than redone, since more samples for that
subset only helps) and the remaining ~2,700 pending crops ran at the new setting. **Finished in
a single ~5.5h session** (19,913s per the kernel log), matching the projected 1/10th-cost
estimate: **3,267/3,267 card crops covered**, staged locally at
`data/processed/gemini/test_local_vlm_card_predictions.csv`. Generated via
`python -m src.generate_submission_gemini --local-vlm-predictions-path
data/processed/gemini/test_local_vlm_card_predictions.csv`, validated (0 errors, text
byte-identical to the base submission, confidence changed in 1408/3300 date and 2339/3300
locality rows vs. the Track A-only candidate), and submitted — see result above. Nothing left
pending on this track.

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
Branch `spike/kaggle-vlm-second-opinion` was deleted 2026-08-24 along with every other
unadopted branch; the writeup above is the record.

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
fallback-spike.md` (rescued onto the main line before its branch,
`experiment/wholeimage-partial-missing`, was deleted 2026-08-24).

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

## Status (2026-08-20): confidence rank refinement (Track A + A3) — shipped, public AURC 0.06007 → 0.04423

Full writeup: `docs/superpowers/plans/2026-08-20-confidence-rank-refinement.md`. Branch
`experiment/confidence-rank-refinement` (since deleted; merged into main via
`research/improve-predictions-2026-08-21`).

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

**A3 test-set extension — complete, submitted, and now the best result.** First attempt
(dataset `matiaspoullain/museumscat-vlm-test-subset`, kernel
`matiaspoullain/museumscat-vlm-test-set-run`, 10 stochastic samples/item at temperature 0.7)
was user-cancelled after ~10h at ~565/3267 items (~29 items/hr/GPU) — full completion at that
rate was multi-day. Before relaunching, `src.old.analyze_local_vlm_sample_count` (new, zero GPU)
re-derived the majority vote from the 152 already-collected dev crops' raw samples truncated to
1/3/5/10, and found the agree/disagree NED gradient `src.local_vlm_signal.
local_vlm_agreement_score` actually consumes (only the majority-vote *text*, never Qwen's own
self-consistency confidence) is already ~fully present at n_samples=1 for `verbatimDate` (0.005
vs 0.037, unchanged through n=10) and still large for `verbatimLocality` (0.039 vs 0.151 at
n=1, vs 0.039 vs 0.202 at n=10). User chose **`N_SAMPLES=1`, greedy decoding**
(`do_sample=False`) over a more conservative `N_SAMPLES=3` — a ~10x per-item cost cut. The 565
already-collected 10-sample items were rescued (downloaded from the cancelled run, merged,
re-uploaded as resume state) so the relaunch only spent GPU time on the ~2,700 remaining
crops. **Finished in a single ~5.5h session, 3,267/3,267 covered** — no multi-day/multi-session
work needed after all. Generated the third candidate
(`submission_gemini_rank_refined_local_vlm.csv`), validated clean, and **submitted to Kaggle
2026-08-20 (user-triggered): public AURC 0.04423** (ref `55654639`), down from 0.05073 — a
−12.8% delta closely matching the −14.3% dev-set delta despite A3's dev evaluation only having
partial crop coverage. This is now the best submitted result; nothing left pending on Track A3.

**Track B (text/NED reduction via MBR sample selection, HTML-entity decoding, corpus-based
locality snapping — `src/old/consensus_text.py`) was built, tested, and evaluated but NOT adopted.**
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

## Status (2026-08-23): whole-image-primary route — SHIPPED, public AURC 0.04423 → 0.02530

Full writeup: `docs/superpowers/plans/2026-08-21-transcription-quality-push.md`. Branch
`research/improve-predictions-2026-08-21`, merged to `main` 2026-08-24.

**The finding that motivated this round.** Replacing confidence with a *perfect* (oracle) global
ranking over the shipped text gives dev AURC 0.0127 — but the public leaderboard's top five were
already at 0.005–0.009. **Our floor with perfect confidence sat above what the leaders already
scored**, so no further confidence work could reach them. The binding constraint was text, and
`verbatimLocality` specifically (dev mean NED 0.169 vs `verbatimDate`'s 0.037).

**What changed: the text source.** Every specimen is now transcribed from its **original
uncropped 45 MP photo** at `media_resolution=MEDIA_RESOLUTION_HIGH`, greedy (`temperature=0.0`,
`n_samples=1`), replacing the per-card SAM-crop route. Dev results (full 200 rows, never a
subset):

| route | date NED | loc NED | loc exact | loc MISSING |
|---|---|---|---|---|
| per-card only | 0.1919 | 0.3185 | 97 | 74 |
| production (per-card + fallback, shipped at 0.04423) | 0.0372 | 0.1655 | 122 | 34 |
| **whole-image primary** | **0.0329** | **0.1224** | **129** | 38 |

Locality NED −26% relative, 90% paired-bootstrap CI **[−0.0707, −0.0159]** excludes 0. The
oracle-rank floor moves 0.0121 → **0.0077**, i.e. *below* the leaderboard's top-5 band — the
structural unblock the round was looking for. Whole-image wins on **every stratum**, including
the one where bias was most plausible (specimens where per-card *did* find a locality:
0.1781 → 0.1222).

**Why D1 was falsified on crops but D4 won on whole images:** a card crop is ~1117×450 px,
already legible at default tokenization, so raising media resolution had nothing to recover
(three arms, all bootstrap CIs including 0). The whole-image path sends the raw 8192×5464 photo,
which default media resolution downsamples to near-illegibility. Same knob, opposite outcome,
because the input sizes differ by ~90x in pixels.

**Confidence had to be rebuilt**, since `n_samples=1` removes self-consistency entirely.
`src/route_agreement.py`: **cross-route agreement** (does whole-image text match per-card text?)
as the coarse tier, with locality corpus support and local-VLM agreement averaged into a bounded
within-tier tiebreak via the existing `combine_tier_and_within_tier_score`. Five designs were
graded on dev; all five beat shipped Track A3 (0.0487), best 0.0353.

**Test-set collection took 3 days** (2026-08-21/22/23), two independent API keys walking
`test.csv` from opposite ends — Kaggle kernel `matiaspoullain/museumscat-wholeimage-primary`
forward, a local invocation in reverse — meeting in the middle with no coordination beyond the
progress-dataset round-trip. **3,300/3,300 covered, 0 field-level conflicts across 3,064
overlapping keys**: the two keys never disagreed on a single field, which is what greedy
`temperature=0.0` should buy and is worth having confirmed at that scale.

**Submitted 2026-08-23 (user-triggered): public AURC 0.02530** (ref `55721990`), from 0.04423 —
**−42.8%**, the largest relative gain since the classical→Gemini switch, and the only step whose
public delta *exceeded* its dev projection (−27.5%). Consistent with this being the first change
to touch **text** rather than confidence rank: better text lowers per-row NED, which improves
AURC through both risk and rank at once. Rank ~11th of the public leaderboard.

**Throughput note for any future full pass:** ~19 s per whole-image call (45 MP upload), so
~190 rows/hour/key — bounded by request latency, not quota or `--rpm`. A day's ~1,000-call budget
needs ~5.5 h of wall clock, so Kaggle's 12 h session cap has never been the binding constraint.

**Techniques evaluated and rejected this round** (kept so they aren't re-proposed): `thinking_config`
(`thinking_level="low"`) was the *worst* arm on both fields; a corpus candidate-list rerank is
capped at 34% recall because the corpus is self-poisoning (our own wrong predictions are corpus
entries at distance 0 from themselves) and frequency filtering removes rare *true* localities
faster than it removes errors.

## Next steps (updated 2026-08-23, after Phase 6)

The earlier list is fully discharged: the confidence config was wired (2026-08-08), the test-set
submission generated (2026-08-18), `verbatimLocality` text quality attacked (the 2026-08-21 push,
D4), and the rank layer re-measured on the new route (Phase 6, same day). Current state:

1. **~~Mark the 2 final submissions~~ — done 2026-08-24 (user-performed, on Kaggle's site).**
   Eight submissions exist (0.60501, 0.06007, 0.06105, 0.06263, 0.05073, 0.04423, 0.02530,
   0.02382); the final selection is now locked in. Nothing about it is reversible or checkable
   from the CLI, so do not attempt to re-do or verify it.
2. **~~Ranking levers~~ — swept 2026-08-24, exhausted.** Both unexplored levers
   (`src.route_agreement.ROUTE_TIERS`' `partial_scale` and the equal-weight averaging of the
   within-tier signals) were measured on the full dev set on branch
   `experiment/rank-param-sweep-2026-08-24`, **since deleted** — the numbers survive in the plan
   doc, the code does not. Best combination: dev 0.03450 → 0.03330, 90% CI [−0.00266, +0.00028] —
   **not adopted**; the submission regenerated through the swept code was byte-identical to the
   submitted 0.02382 file, so nothing was lost by dropping it. The finding
   that matters is the negative one: reshaping these constants recovers only **4.5%** of the
   0.0265 dev gap to the oracle floor, so the remaining ranking headroom is information the
   current signals do not carry, not a weighting problem. Do not re-propose tuning this layer.
   See the plan doc's Phase 7 for the tables (including two hypotheses it falsified).
3. **~~Text quality on the partial-agreement group~~ — arbitrated 2026-08-24, population closed.**
   A third Gemini pass adjudicated the 63 disputed dev specimens (spec
   `docs/superpowers/specs/2026-08-24-partial-agreement-arbitration-design.md`). Locality NED on
   the mined population fell 15.3%, but full-dev AURC moved only −0.00044 (CI includes 0), and
   **transcribing every disputed field perfectly is worth just −0.00280** — the whole population's
   ceiling, measured. Not adopted, and the 1,041-call test pass was never run.
   **The generalizable lesson: "worst mean NED" is the wrong way to pick a text target under
   AURC.** These rows are already ranked at the bottom, so the risk-coverage curve already excuses
   them. A text target has to be wrong *and currently trusted* — which is why D4 paid −42.8%
   (it improved text everywhere, including the high-confidence rows that dominate the curve) and
   this paid nothing.
4. **`kaggle competitions submit` stays a deliberate, user-triggered action** — 10/day, 2 final,
   ends 2026-08-31. Never from a script or without explicit instruction.
5. **Two standing methodology rules, both learned the expensive way.** Score the **full** 200-row
   dev set, never a subset (the partial-MISSING spike's isolated 24-row measurement looked like
   0.0281→0.0166 and was actually +0.00004 on the full set; AURC is a global rank metric). And
   treat a dev delta as a **direction-and-significance test, not a magnitude estimate** — D4's
   public gain was 1.6x its dev projection while Phase 6's was 0.6x it.

## Status (2026-08-24): repo cleanup — `src/old`, docstrings, pipeline diagram

Branch `chore/src-cleanup-2026-08-24`. No behaviour change: the submission generated after the
cleanup is byte-identical, and all 335 tests pass.

**What was done, and the criterion used.** The import graph of `src/` was computed and the
closure reachable from the live entry points (`src.generate_submission_gemini`, the three Gemini
runners, `src.evaluate_gemini_dev`, `src.download_kaggle_data`, `src.extract_card_crops`) was
taken as the active set: **28 modules in, 29 out**. The 29 moved to `src/old/` (classical OCR
pipeline, Danish recognizer fine-tune tooling, one-off EDA/inspection scripts, the manual
annotation apps, and the closed confidence experiments — `experiment_confidence_grid`,
`consensus_text`, the `analyze_*` miners). Tests stayed in `tests/` and now import
`src.old.<module>`, so the moved code keeps its safety net.

**Reachability, not age, was the criterion** — and that distinction is the part worth
remembering. `pipeline.py`, `classify_field.py`, `merge_records.py`, `reconstruct_lines.py` and
`generate_submission.py` are all classical-era code that **stayed in `src/`**, because the
Gemini route still imports them (`validate_submission`, the manifest/prediction loaders, the
multi-card merge). Deleting or moving them by their 2026-07 timestamps would have broken the
active path.

Also: the 14 modules that never had a module docstring got one (`metrics`, `pipeline`,
`merge_records`, `detect_labels`, `classify_field`, ... — the oldest ones);
`src.generate_submission_gemini`'s docstring was rewritten, since it still described the
2026-08-18 per-card default as though it were the current config and never mentioned the
whole-image-primary route that actually ships; and `README.md` gained a **Mermaid pipeline
diagram** (inputs → scripts → outputs) plus the submission-history table. The diagram was
checked against the code, not written from the narrative: it deliberately excludes
`src/run_gemini_test_wholeimage.py` and `src/confidence_penalty_rules.py`, because
`generate_wholeimage_primary_submission` reads neither — both belong to the earlier variants
only.

## Status (2026-09-02): competition ended — final result

MuseumSCAT ended 2026-08-31. Final standings read via `kaggle competitions leaderboard -c
museumscat-specimen-collection-annotation-task -s` (this command returns the **private**
leaderboard once a competition is over — its per-team `score` matches the `privateScore`
column `kaggle competitions submissions` had already been showing all along, e.g. exactly
0.02345 for ref `55722819`; `kaggle competitions leaderboard -d` instead always downloads the
`publicleaderboard.csv` artifact, i.e. the pre-deadline public split, which is a different,
now-frozen number (0.02382, rank 17th) — don't confuse the two files):

**Final private-leaderboard AURC: 0.02345 — rank 15th of 127 teams.**

The score comes from submission ref `55722819` (`submission_gemini_wholeimage_primary.csv`,
whole-image-primary route + risk-ordered route tiers, public 0.02382) — one of the two
submissions marked final on 2026-08-24. Its private score (0.02345) beat the other marked
final's (ref `55721990`, private 0.02414), so Kaggle's "best of your two finals" rule picked it
automatically; no action needed on our end. Notably this is the *exploratory* Phase 6 submission
whose dev-set 90% CI had included 0 (not a confirmed win at submission time) — it turned out to
be the better of the two on the real private split anyway.

Nothing further is actionable here: submissions are closed, the leaderboard is final, and this
section is the closing record of the competition for this repo.

## Branches (cleaned up 2026-08-24)

**Everything adopted lives on `main`.** `research/improve-predictions-2026-08-21` — which carried
the whole-image-primary route (D4, 0.02530) and the risk-ordered route tiers (Phase 6, 0.02382) —
was fast-forwarded into `main` on 2026-08-24, and every unadopted branch was deleted the same day:
`experiment/rank-param-sweep-2026-08-24`, `experiment/partial-agreement-arbitration`,
`experiment/wholeimage-partial-missing`, `spike/kaggle-vlm-second-opinion`,
`experiment/confidence-rank-refinement`, and the stale remote `gemini-vlm-transcription`.

**The findings survive, the code does not.** Every deleted branch's writeup was rescued onto the
main line first (`docs/superpowers/plans/` and `specs/`), because what stops a dead end from being
re-proposed is the measurement, not the implementation. Four rounds now have ended in "measured,
not adopted" — the partial-MISSING fallback, the local-VLM blend, the constant sweep, and
arbitration — and in each case the numbers are the deliverable.

The practical consequence: `src/` on `main` is exactly the code that produced the best submitted
result (public AURC 0.02382). Anything a doc describes that is not in `src/` was deliberately
dropped, not lost.

## Repo structure

Single repo, no git submodules (unlike `../bayesian-kaggle`, which this repo's scaffolding
conventions were borrowed from — same `download_kaggle_data.py` pattern, same
`kaggle_meta/pages/` convention, but flattened since this is one competition, not a
submodule-per-competition setup).

- `src/` — pipeline code, see the plan below for what lives where. **Only the modules in
  `src/` itself are on the active path** (the Gemini whole-image-primary route that produced
  public AURC 0.02382).
- `src/old/` — code that is no longer on that path: the classical OCR pipeline, the Danish
  recognizer fine-tune tooling, one-off EDA/inspection scripts, the manual annotation apps, and
  the closed confidence experiments (`experiment_confidence_grid`, `consensus_text`, the
  `analyze_*` miners). Moved there 2026-08-24, nothing deleted, tests still run against
  `src.old.<module>`. `src/old/README.md` says what each group was for and how it ended.
  **Caveat for Kaggle runs:** the `museumscat-gemini-code` dataset is built from `src/*.py`
  flattened, so `src/old/` is *not* shipped — a notebook that needs one of those modules has to
  copy it in explicitly.
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
  to get there; its branch has since been deleted) →
  `2026-08-20-confidence-rank-refinement.md` (**read this one for anything current** — the
  tie-breaking headroom finding, the shipped confidence rank refinement taking dev AURC
  0.0715 → 0.0575/0.0493, re-purposes the local-VLM spike above as a within-tier signal
  instead of a blend, and the Track B text-consensus techniques that were evaluated but not
  adopted; its branch has since been deleted, the work is on `main`).
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
