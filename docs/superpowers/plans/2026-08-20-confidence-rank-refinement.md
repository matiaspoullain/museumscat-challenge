# Confidence rank refinement + text consensus (2026-08-20)

## Context

Prior work (`docs/superpowers/plans/2026-08-07-confidence-calibration-experiments.md`)
concluded confidence calibration was "exhausted as a lever" because AURC (`src.metrics.aurc`)
depends only on confidence *rank order*, so any monotone reshaping of a single scalar is
provably inert. That conclusion is correct but incomplete: with `n_samples=3`, raw
self-consistency takes only ~4 distinct values, so roughly half of all submission rows sit in
one tie block at confidence 1.0, and Python's stable sort resolves ties by arbitrary CSV row
order. **Rank order was never fully set** — it was partly determined by dataset row order.

Goal: improve on the shipped submission (`data/processed/submission_gemini.csv`, public Kaggle
AURC 0.06007, dev AURC 0.0715) **without spending further Gemini API quota**, using the ~15,000
already-paid Gemini generations (`raw_samples_json` in every prediction CSV), the 6,853 card
crop images, the 3,500 original photos, and 200 dev ground-truth labels already on disk.

Measurements run at the start of this work (full 200-row dev set, production config):

| Measurement | Result |
|---|---|
| Production baseline | **0.0715** |
| Oracle ordering *within existing ties* (no text change) | **0.0456** (−36%) |
| Anti-oracle within ties | 0.1246 |
| Random tie-break, 200 draws | date p05 0.0248/median 0.0283; loc p05 0.1013/median 0.1145 |
| Oracle best-of-3-sample text selection | ~0.0679 (−0.0036) |
| Global OOF refit of all confidence cells, 10 seeds | **0.0961 — much worse** |

Two consequences: (1) the current tie ordering is statistically indistinguishable from random —
real, large headroom exists; (2) refitting the *whole* ranking reproduces the failure mode that
killed the 19-feature calibrator (mean 0.0723 vs. deterministic 0.0715) — any new rule must be
strictly additive within existing ties, never a global re-rank.

## Approach: two independent tracks

- **Track A — confidence re-ranking.** Text stays byte-identical to the 0.06007 submission;
  only the two `*_confidence` columns change. Per-row NED cannot change, so this track can only
  move AURC through ranking.
- **Track B — NED reduction.** Reduce per-row NED using the samples already collected.

Both were fully implemented and evaluated on the full 200-row dev set (never a subset — see
"Anti-pattern" below); only Track A cleared the adoption bar.

## Track A — SHIPPED (dev AURC 0.0715 → 0.0493)

Reproducible via `conda run -n museumscat-challenge python -m src.evaluate_gemini_dev`'s last
two report blocks.

### A0. Diagnostic harness — `src/analyze_rank_headroom.py`

Oracle/anti-oracle/random-tiebreak bounds, a risk table per (confidence tier × candidate
signal), and a paired bootstrap (resample dev specimens with replacement, score candidate vs.
baseline on the *same* resample) as the adoption gate: a delta whose 90% CI includes 0 is not
adoptable. Also caught a real methodology bug mid-analysis: checking only the *largest* tie
group made `soft_consistency` look like it had zero signal, because that tier (confidence=1.0)
is exactly where hard and soft consistency coincide by construction (all 3 samples matched
exactly). Fixed to check every eligible tier — the real gradient was in the 0.667/0.333 tiers,
right where the mechanism predicts it should be.

### A1. Structural-MISSING tier — `src.confidence_rank_rules.STRUCTURAL_MISSING_CONFIDENCE`

`MISSING_CONFIDENCE = 0.5` (`src/merge_records.py`) is an unjustified magic number. But
"`<2` card crops were Gemini-processed for this specimen, and none of them (nor the whole-image
fallback) found a value" is genuinely low-ambiguity: on dev this population's mean NED for
`verbatimDate` (0.018) is *better* than the confidence=1.0 text tier's (0.021), yet it ranked
below the 0.667/0.7 tiers purely because 0.5 is a smaller number. Deliberately excludes `>=2`
processed-crops-all-MISSING — that's `src.confidence_penalty_rules`' own under-coverage risk
pattern (a second card's real value may have been missed), left to that existing penalty.

Elevated to **0.95** (a fixed placement between the 1.0 and 0.7 tiers). A placement sweep found
values *above* 1.0 (invalid — confidence must be ≤1) or *tied at* 1.0 score even better on this
particular dev CSV's row order (0.0560–0.0608 vs. 0.95's 0.0639) — but that's exactly the
arbitrary-tie-break dependency this whole track exists to remove, so it was rejected on
principle even though it scores better here. 0.85–0.99 all give the *identical* 0.0639, a
plateau that's real evidence of robustness, not a fragile point estimate.

Effect: **0.0715 → 0.0639** (verbatimDate 0.0258→0.0196, bootstrap 90% CI [−0.0114,−0.0007];
verbatimLocality 0.1172→0.1082, CI [−0.0157,−0.0004] — both exclude 0).

### A2. Within-tier signals — `src.confidence_rank_rules.within_tier_score`

`combine_tier_and_within_tier_score(tier_confidence, within_tier_score)` returns
`base + TIER_EPSILON·clip(score,0,1)` where `base = min(tier_confidence, 1 − TIER_EPSILON)` —
the cap matters only for the tier=1.0 ceiling (no tier above it to borrow headroom from), caught
by regenerating the real submission and hitting `validate_submission`'s `[0,1]` check (confidence
1.01) — `aurc()` itself doesn't validate range, so the pre-cap unit tests didn't catch it.

Candidates evaluated, each gated by a per-tier risk-gradient check plus paired bootstrap:

| Signal | Field | Verdict | Evidence |
|---|---|---|---|
| `soft_consistency` (reuse of `gemini_transcribe.soft_consistency_scores`, previously unused) | both | **adopt** | Monotone in the 0.667/0.333 tiers (date: n=27, 0.240→0.037; locality: n=39, 0.374→0.131), both bootstrap CIs exclude 0 |
| `date_plausibility_score` (new: parses as a real calendar date, targets catalog-number-as-date confounds like "608.", "2704") | date | **reject** | Inverted gradient in the largest (1.0) tier; net AURC effect +0.0011, CI includes 0 |
| `corpus_support_score` (`src.locality_corpus`: frequency of a locality string across all ~15,000 raw samples, ± edit-distance-2 fuzzy neighbours) | locality | **adopt** | Monotone in 3 of 5 eligible tiers incl. the two largest (n=87: 0.132→0.061; n=39: 0.326→0.177); AURC 0.1082→0.0968, CI [−0.0252,−0.0004] |

Shipped combiner: `within_tier_score` = soft consistency alone for `verbatimDate`;
`avg(soft_consistency, corpus_support_score)` for `verbatimLocality` (beats corpus support alone
on its own paired bootstrap: 90% CI [−0.0012,−0.0001]).

Combined Track A (A1 + A2), verified through the real production code path
(`src.generate_submission_gemini.build_submission_df(apply_rank_refinement=True)`, not just the
ad hoc measurement script): **dev AURC 0.0715 → 0.0575** (verbatimDate 0.026→0.019,
verbatimLocality 0.117→0.096).

### A3. Local-VLM agreement — `src.local_vlm_signal`

The closed spike (`2026-08-19-local-vlm-second-opinion-spike.md`) found Qwen2.5-VL's signal
"concentrates in exactly the rows Gemini is already sure about", then only tested blending
across confidence levels (harmful: locality AURC 0.079→0.146) and gating at low confidence (no
effect) — it never tested reordering *within* the confidence=1.0 tie, which is exactly where
this track's headroom measures.

Re-analyzed the already-collected 152/197 dev crops
(`data/processed/gemini/kaggle_vlm_spike_dataset/previous_card_results.csv`) purely as a
within-tier agreement signal (does the local VLM's specimen-level text match Gemini's final
text?), zero additional cost. Within Gemini's top confidence tier:

| Field | Agree (n, mean NED) | Disagree (n, mean NED) |
|---|---|---|
| verbatimDate | 21, 0.000 | 20, 0.033 |
| verbatimLocality | 25, 0.039 | 23, 0.193 |

A real, mechanistically clear gradient — 5x for locality. Wired into `within_tier_score` as an
optional `local_vlm_agreement` parameter (equal-weight blend on top of the existing base score;
`None` — no local VLM data for this split — leaves behavior unchanged). Combined with A1+A2,
verified through the real production code path: **dev AURC 0.0575 → 0.0493**
(verbatimDate CI [−0.0089,−0.0009], clean; verbatimLocality CI [−0.0285,+0.0054], directionally
strong but not as clean — partial 152/197 coverage dilutes both effect and bootstrap power).

**Test-set extension launched 2026-08-20, user-cancelled same day — paused, not abandoned.**
(User-authorized, no Gemini quota — a second, independent model on Kaggle's free T4 GPU):
dataset `matiaspoullain/museumscat-vlm-test-subset` (3,267 test card crops, reusing the
existing `matiaspoullain/museumscat-card-crops` image dataset), kernel
`matiaspoullain/museumscat-vlm-test-set-run` (copy of the dev pilot's
`notebooks/kaggle_local_vlm_spike.ipynb`, same subprocess-isolated/sub-batched/resumable design
that took 9 debug iterations to get right — see the spike doc's Outcome). Whole-image scope
excluded (the original spike's whole-image half never actually ran, 0 rows collected — no
validated mechanism to extend). At 3,267 items × 10 samples on a single T4x2, this turned out
to be a multi-day effort in practice too, not just in projection: after ~10h the run had only
reached 293/1633 items (~29 items/hr — the "1,633" item count reflects the notebook's
sub-batching, not the raw 3,267 crop count). User cancelled it 2026-08-20 rather than block on
a multi-day background job. The 293 items processed are preserved in the Kaggle progress
dataset (`kaggle_vlm_test_subset_dataset`), so nothing already computed is lost — this is
paused, resumable, and not currently planned rather than a dead end. If revisited: either
resume across several shorter sessions, or scope down to only Gemini's top confidence tier
(where the A3 dev-set signal's gradient is strongest, 5x for `verbatimLocality`) instead of
the full item population.

### Submission candidates

`data/processed/submission_gemini_rank_refined.csv` — A1+A2 only, generated via
`python -m src.generate_submission_gemini --apply-rank-refinement`. Verified end-to-end:
text columns byte-identical to `submission_gemini.csv` (confirmed via direct comparison, not
just "should be" — see Track A gate below), confidence range valid, `validate_submission` passes
clean, 1809/3300 (date) and 2154/3300 (locality) rows got a different confidence value, distinct
confidence value count jumped from 10–13 to 64–500. **Submitted to Kaggle 2026-08-20: public
AURC 0.05073** (ref `55648362`), down from the prior best 0.06007 (ref `55607015`) — the
dev-set gain held on the real leaderboard at a similar relative magnitude (−15.6% public vs
−19.6% dev). Now the best submitted result.

A third candidate (`submission_gemini_rank_refined_local_vlm.csv`, A1+A2+A3) would need the
paused Kaggle test-set run above to complete or be rescoped; not pursued further this session.

## Track B — evaluated, NOT adopted

`src.consensus_text.py`: MBR (minimum-Bayes-risk) sample selection for the crop-level
no-majority tie-break, HTML-entity decoding, and corpus-based near-neighbour snapping. All
built, tested, and evaluated on the **full** 200-row dev set with the same paired-bootstrap
protocol as Track A (never a subset — see "Anti-pattern" below).

**MBR text selection: net HARMFUL when layered on the full Track A confidence stack.**
Overall AURC 0.0575 → 0.0583 (worse); verbatimLocality's bootstrap 90% CI [+0.0002,+0.0015]
excludes 0 in the *harmful* direction — statistically confirmed, not noise. Isolated (locality
text only, no confidence layer): mean NED 0.1692 → 0.1719, also worse. Rejected outright.

**HTML-decode + corpus-snap: directionally positive but does not clear the adoption bar.**
Layered on the full stack: verbatimLocality AURC 0.0960 → 0.0946 (−0.0014), but bootstrap 90%
CI [−0.0032,+0.0003] includes 0. A manual audit of the 7 dev rows the snap actually changed found
5 genuine corrections to NED=0 (e.g. `Neksel\x00 S`→`Nekselø S.`, `B&#248;rstinge Mose`→
`Børstinge Mose`) but **2 confirmed regressions** where an already-correct rare spelling got
snapped to a more frequent wrong neighbour (`Harboøre`→`Harboør`, 0.000→0.125 NED; `Hjorte`→
`Holte`, 0.000→0.333 NED) — exactly the risk `src.clean_gemini_predictions`' docstring already
flagged for gazetteer-style corrections, now empirically confirmed even for a same-corpus (not
external-gazetteer) snap.

**Decision: Track B is not wired into production.** Kept as tested, documented dead ends
(`src/consensus_text.py`, `tests/test_consensus_text.py`) so they aren't re-proposed. The oracle
ceiling for text-only improvement was small to begin with (only −0.0036 dev AURC at best), and
neither concrete technique tried actually captured it safely.

## Anti-pattern this work deliberately avoided

The partial-MISSING whole-image fallback spike (`2026-08-19-partial-missing-wholeimage-fallback-spike.md`)
first measured its pooled design on just the 24 affected dev rows in isolation and saw an
apparent 0.0281→0.0166 win; re-measured on the full 200-row dev set the true effect was
+0.00004 (noise), and the real Kaggle submission regressed. Every evaluation in this document —
Track A's per-signal gates and Track B's — scores the **full** 200-row dev set, never a subset,
specifically because of that precedent.

## Explicitly out of scope / already dead

Any 1-D calibrator (isotonic/Platt/shrinkage), global refits of the ranking, soft consistency as
a *replacement* for hard consistency, L1/L2 multi-feature calibrators as a global map, PCA+RF,
t-SNE, the partial-MISSING whole-image fallback in either confidence design, pooling crop
`MISSING` votes with whole-image votes, æøå-triggered text swapping, external gazetteers,
unconditional local-VLM blending, gating local VLM at low confidence.

## Files

**New:** `src/analyze_rank_headroom.py`, `src/confidence_rank_rules.py`, `src/locality_corpus.py`,
`src/consensus_text.py` (evaluated, not wired in), `src/local_vlm_signal.py`, matching
`tests/test_*.py` (all new modules fully covered), this doc,
`data/processed/gemini/kaggle_vlm_test_subset_dataset/` (gitignored),
`data/processed/gemini/kaggle_kernel_push_vlm_test/` (gitignored).

**Modified:** `src/evaluate_gemini_dev.py` (Track A + Track A3 report blocks),
`src/generate_submission_gemini.py` (`apply_rank_refinement`/`local_vlm_predictions_path`
opt-in flags, three separate output paths so candidates can be diffed/submitted independently),
`CLAUDE.md`.

**Reused unchanged:** `src/metrics.py`, `src/calibrate_gemini_confidence.py`,
`src/confidence_penalty_rules.py`, `src/clean_gemini_predictions.py`,
`src/gemini_transcribe.soft_consistency_scores`, `src/pipeline.py`.
