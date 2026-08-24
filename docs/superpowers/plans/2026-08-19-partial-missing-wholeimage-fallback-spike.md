# Partial-MISSING Whole-Image Fallback Spike

**Branch:** `experiment/wholeimage-partial-missing` (kept separate from
`main` per explicit instruction — not merged as of this writing).

**Goal:** production's whole-image fallback
(`docs/superpowers/specs/2026-08-07-gemini-wholeimage-fallback-design.md`)
only fires when a specimen has **both** fields `MISSING` after the per-card
pass (`src.gemini_batch_runner.wholeimage_pending`'s `all()` gate). That
leaves a gap: specimens where the per-card pass found **one** field but
missed the other never get a second look at the full photo for the missing
one. This spike asks whether extending the fallback to that "exactly one
field MISSING" population is worth the extra Gemini quota.

## Method

`src.gemini_batch_runner.run_wholeimage_batch` gained an optional
`pending_override` parameter (default `None`, existing both-MISSING
callers unaffected) so a specimen list can be supplied directly instead of
computed via `wholeimage_pending`.

New script `src.run_gemini_dev_partial_wholeimage`
(`partial_missing_specimens()`) selects dev specimens where **exactly one**
of `verbatimDate`/`verbatimLocality` is `MISSING` after the per-card merge —
the complement of the production gate — and runs the existing
`WHOLE_IMAGE_PROMPT`/self-consistency machinery against them, writing to a
separate output file (`dev_wholeimage_partial_predictions.csv`) so it never
collides with production's `dev_wholeimage_predictions.csv`.
`apply_wholeimage_fallback` already only fills a field when the per-card
side is `MISSING`, so no changes were needed there — the same overlay
function works unmodified for this population too (verified: 0 specimens
where it overwrote an already-found per-card value).

Of the 200 dev specimens, **24** have exactly one field `MISSING`. Ran all
24 (72 Gemini calls, 24 × 3 samples) on 2026-08-19.

## Result

| ground truth for the missing slot | whole-image outcome | n |
|---|---|---|
| genuinely `MISSING` | correctly said `MISSING` | 20 |
| genuinely `MISSING` | hallucinated text | 0 |
| real text | recovered correctly (NED<0.5) | 4 |
| real text | missed / wrong | 0 |

Of the 24 "one field MISSING" cases, 20 were never actually failures — the
per-card pass had already correctly determined nothing was there. Only 4
were genuine per-card misses, and whole-image recovered **all 4**, each at
confidence 1.0, with **zero** hallucinations on the 20 truly-empty slots.

Layering this on top of production's config (raw self-consistency + cleanup
+ both-MISSING fallback + coverage penalty, `src.evaluate_gemini_dev`'s
current best) on the full 200-specimen dev set:

| config | verbatimDate AURC | verbatimLocality AURC | overall AURC |
|---|---|---|---|
| production (both-MISSING fallback only) | 0.0258 | 0.1172 | 0.0715 |
| + partial-MISSING fallback (this spike) | 0.0245 | 0.1140 | **0.0692** |

Delta **-0.0023** (lower is better), both fields improved, on the leftover
120% of true dev AURC 0.0715 documented in `2026-08-07-confidence-
calibration-experiments.md`.

**Caveat: n=4 recovered cases.** The 100%-recovery / 0%-hallucination result
is clean but comes from a tiny sample — not something to treat as a stable
rate. The reasonable next step (not a re-derivation, an actual larger test)
is running the same pass against the analogous **test-set** population
before deciding whether this belongs in production.

## Test-set scope

Same split computed against `test_card_predictions.csv` (3300 test
specimens, per-card pass 100% complete as of 2026-08-18):

| category | n (of 3300) |
|---|---|
| 0 fields MISSING | 1745 |
| **exactly 1 field MISSING (this spike's target)** | **514** |
| both fields MISSING (already covered by production's fallback) | 1041 |

514 specimens × 3 samples ≈ 1542 Gemini calls — run via the Kaggle quota-
source infrastructure (see CLAUDE.md's "Kaggle quota-source infrastructure"
section) rather than locally, to avoid competing with local quota used for
other work. New script `src.run_gemini_test_partial_wholeimage` mirrors
`src.run_gemini_test_wholeimage`'s shape (same per-card-complete gate,
same resumability) but selects the partial-MISSING population via
`pending_override` instead of the both-MISSING gate, writing to
`test_wholeimage_partial_predictions.csv` (kept separate from production's
`test_wholeimage_predictions.csv`).

## Test-set result

Run to completion 2026-08-19: 332/514 via the Kaggle kernel (stopped when
both models in the rotation hit their daily quota), remaining 182 finished
locally the same day (resumable — no work lost, no duplicate rows). All
514/514 done, 0 conflicts merging Kaggle's output into the local progress
file.

Unlike the dev set, `test.csv` has no ground truth, so this can only report
what whole-image *found*, not whether it's correct:

| field | missing slots | filled by whole-image | still MISSING |
|---|---|---|---|
| verbatimDate | 464 | 11 (2.4%) | 453 |
| verbatimLocality | 50 | 47 (**94.0%**) | 3 |

A sharp asymmetry the dev set's small sample (24 slots) didn't surface:
90% of this spike's "one field MISSING" population is specifically a
missing *date* (464/514), and whole-image almost never finds one the
per-card pass didn't (2.4%) — consistent with a lot of these being genuine
period-appropriate missing dates, same as the dev-set pattern where most
"partial MISSING" turned out to already be correctly MISSING. But the
minority with a missing *locality* (50/514) gets rescued **94% of the
time**, mostly at confidence 1.0 (36/47) — this is the same failure mode
the dev spike's 4/4 recoveries hinted at: `verbatimLocality` is more often
a genuine per-card miss (bad crop, cut-off text) than a genuinely absent
field, and the whole photo recovers it easily.

**58/514 specimens (11.3%) gained at least one previously-MISSING field.**
Zero specimens had an already-found per-card value overwritten (same
sanity check as the dev spike, still holds at test scale).

Effect on the full 3300-row test submission if layered on top of
production's existing both-MISSING fallback:

| | verbatimDate MISSING | verbatimLocality MISSING |
|---|---|---|
| current production submission | 1022/3300 (31.0%) | 524/3300 (15.9%) |
| + partial-MISSING fallback (this spike) | 1011/3300 (30.6%) | **477/3300 (14.5%)** |

Consistent with the per-field table above: barely moves `verbatimDate`
(-11 rows), meaningfully reduces `verbatimLocality` MISSING (-47 rows,
~9% relative reduction in that field's MISSING rate) — and
`verbatimLocality` is already documented (`2026-08-07-confidence-
calibration-experiments.md`) as this pipeline's dominant error source, so
a targeted reduction there is disproportionately valuable even though the
specimen count looks modest.

## Rejected: pooling crop votes with whole-image votes for confidence

Before integrating, considered whether the confidence for a field recovered
by whole-image should combine with the per-card crop's votes (which were
mostly voting `MISSING`) rather than just taking whole-image's own
self-consistency outright. Tested on the 24-slot dev subset using only
already-collected `raw_samples_json` (no API cost): pooling the crop's
`MISSING` votes with whole-image's real-text votes and recomputing modal
agreement over the combined set. Even restricted to only the 4 slots where
whole-image found real text (leaving the 20 genuinely-MISSING slots
untouched), pooling made AURC **~4.6x worse** (0.0281 hard-override vs
0.1304 pooled) — it flipped one of the 4 genuine recoveries back to
`MISSING` (a 3-3 tie between the crop's 3 `MISSING` votes and whole-image's
3 real-text votes) and dropped confidence on the other 3 correct recoveries
from 1.0 to 0.5 for no benefit. Root cause: a crop's `MISSING` vote answers
"is there legible text in *this card crop*", not "is there text on the
specimen at all" — pooling treats it as if it were a repeated trial of the
same question whole-image is answering, which it isn't. Consistent with
`2026-08-19-local-vlm-second-opinion-spike.md`'s finding that blending two
confidence sources tends to override rows that were already correct.
**Decision: keep the hard-override design** (`apply_wholeimage_fallback`
already does this, unchanged) — whole-image's own confidence is used
outright when it finds real text; the crop's original `(MISSING,
confidence)` is kept untouched when whole-image agrees nothing's there.

## Status

Dev-set spike complete (2026-08-19). Test-set run complete (2026-08-19,
via Kaggle + local, see above). Cost: ~1542 Gemini calls (514 × 3
samples) split across a third Kaggle quota-source kernel
(`matiaspoullain/museumscat-partial-wholeimage-kaggle-quota-source`,
notebook `notebooks/kaggle_run_gemini_test_partial_wholeimage.ipynb`) and
a local run, no quota taken from the per-card pass or production
both-MISSING fallback.

**Wired into `src.generate_submission_gemini` on this branch 2026-08-20**
(not merged to `main` -- see decision below): the partial-MISSING fallback
(`apply_wholeimage_fallback`, same hard-override function as the
both-MISSING pass, called a second time with
`test_wholeimage_partial_predictions.csv`) applied after the existing
both-MISSING fallback -- the two overlays never touch the same specimen
(mutually exclusive by construction, see `build_submission_df`'s docstring
in `src/generate_submission_gemini.py`), so applying both in either order
is safe. `src.evaluate_gemini_dev`'s "production config" line updated to
match: dev AURC **0.0715 -> 0.069** (verbatimDate 0.026->0.024,
verbatimLocality 0.117->0.114). (Later swapped for the pooled design
below, then closed entirely -- this hard-override wiring is what
submission `55636785` used.)

**Submitted to Kaggle 2026-08-20 (user-triggered) -- public AURC
regressed: 0.06007 -> 0.06105** (submission ref `55636785`), despite the
dev-AURC improvement and the reduced test-set MISSING rates. Investigated
before deciding what to do with this branch:
- Diffed the old vs. new submission programmatically: exactly 58 rows
  changed, matching the expected count precisely -- no unrelated bug
  touching other specimens.
- Manually inspected 6 of the 58 recovered values against the actual
  specimen photos (`data/raw/images/`): 5/6 matched the photo exactly
  (`1708366.jpeg` "Vamdrup", `1706187.jpeg` "Tiis Sø", `1706792.jpeg`
  "Asminderød", `1706002.jpeg` date "26/V 2004", all typewritten or clear
  handwriting), 1/6 partially right at appropriately-lowered confidence
  (`1706462.jpeg` "Jydske As" correct, second word "Hørndal" likely a
  misread of ambiguous cursive, confidence 0.67 not 1.0), 0/6 outright
  hallucinations. This spot check does not support a systematic quality
  problem with the recovered values.
- The public leaderboard is documented (`kaggle_meta/pages/foundational-
  rules.md`) as "a representative sample" of the test set, not the full
  set, and final standing is decided by the *private* leaderboard --
  so a small regression (+0.00098, ~1.6% relative, from a change
  touching 58/3300 = 1.76% of rows) on the public sample is plausibly
  noise rather than a real quality regression, though this can't be
  confirmed without the private score.

Considered whether the public-leaderboard regression above was just
sampling noise, and tried a second confidence design to check: instead of
taking whole-image's own 3-sample confidence outright
(`apply_wholeimage_fallback`, "hard override"), pool whole-image's raw
samples with every one of the specimen's card crops' raw samples for the
recovered field, **discarding any "MISSING" sample before voting**
(`src.pooled_wholeimage_fallback.apply_pooled_wholeimage_fallback`) --
architecturally more defensible, since a crop's MISSING vote answers "no
text in this crop", not "no text on the specimen", so it shouldn't compete
with (or dilute the denominator of) a source that actually found
something. Implemented with tests (`tests/test_pooled_wholeimage_fallback.py`)
and wired into `src.generate_submission_gemini` in place of the hard
override for the partial-MISSING overlay only.

**First measurement was methodologically flawed and overstated the
improvement.** Scored in isolation on just the 24 dev-set partial-MISSING
slots, pooling looked like a large win (AURC 0.0281 -> 0.0166). But AURC
is a global rank metric -- scoring a 24-row subset by itself doesn't
reflect how those same predictions rank once embedded in the full
submission alongside the other 176/3276 rows they actually compete
against for confidence rank. Re-measured correctly, against the complete
200-row dev set: **the difference was negligible** (overall AURC 0.06922
-> 0.06925, +0.00004 -- noise).

**Submitted to Kaggle anyway, to confirm empirically 2026-08-20 -- got
worse again, by more this time: public AURC 0.06105 -> 0.06263**
(submission ref `55637181`), making pooling *worse* than the hard-override
design it was meant to improve on, despite being architecturally sounder
and showing ~zero dev-set difference. Two submissions in a row, in the
same direction, ~1.6% then ~4.3% relative regression vs. the 0.06007
baseline -- past the point where "public leaderboard sampling noise"
remains a plausible explanation on its own; something about extending the
whole-image fallback to partial-MISSING specimens (in both of the two
confidence designs tried) makes the real submission worse, for reasons the
dev-set signal and a 6-sample manual photo audit did not surface. Plausible
mechanism, not confirmed: pooling recovers *more* non-MISSING values than
the hard override (`verbatimLocality` MISSING 477 -> 474 -- pooling can
override a case where whole-image's own 3-sample majority was itself
MISSING, which the hard override never touches), so it takes on more
opportunities to be wrong exactly where AURC penalizes hardest.

**Decision: closed, not merged, not pursued further.** Both confidence
designs regressed the real submission, monotonically worsening
(0.06007 -> 0.06105 -> 0.06263), which outweighs a clean-looking manual
spot check and dev-set signals that turned out not to predict real
submission behavior at all in this case. Kept on this branch as a record,
same treatment as `2026-08-19-local-vlm-second-opinion-spike.md`'s
unmerged `spike/kaggle-vlm-second-opinion` branch.

**Raw transcriptions archived** (Kaggle private dataset
`matiaspoullain/museumscat-partial-wholeimage-archive`, created
2026-08-20) so the ~1542 Gemini calls this spike spent aren't lost if
revisited: `dev_wholeimage_partial_predictions.csv` (24 dev specimens) and
`test_wholeimage_partial_predictions.csv` (514 test specimens), both with
full `raw_samples_json` per specimen -- any future confidence-scoring
attempt can reuse this data without re-querying the API. Deliberately a
dedicated dataset, not folded into the mutable `museumscat-gemini-progress`
dataset (whose contents get wholesale-replaced on every re-version, see
CLAUDE.md's "Kaggle quota-source infrastructure" section -- that's exactly
the kind of accident this archive avoids).

If revisited: the private leaderboard score (visible after the
competition ends 2026-08-31) would help settle whether the earlier public
regression was noise or real; if it also regressed, that would rule noise
out more conclusively. A larger manual audit (more than 6 of 58) or
checking whether errors concentrate in the low-hit-rate `verbatimDate`
recoveries (11/464, vs. `verbatimLocality`'s 47/50) would be the next
diagnostic step -- but this isn't currently planned.
