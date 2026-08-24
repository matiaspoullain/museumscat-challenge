# Partial-agreement arbitration — design

**Date:** 2026-08-24
**Branch:** `experiment/partial-agreement-arbitration`, deleted 2026-08-24 once the result
came back negative — this document is the record, and none of the code described below is in
the tree.
**Status:** implemented, measured on dev, **NOT adopted** — see Outcome
**Supersedes nothing.** Purely additive to the whole-image-primary route shipped at public
AURC 0.02382.

## Problem

Phase 6's stage-1 diagnostic split every specimen into four cross-route agreement groups and
found the **partial-agreement** group — both routes read text, but different text — is the worst
of the four in *both* fields. Phase 7 then established that the remaining ranking headroom is not
recoverable by reshaping tier constants (it recovers 4.5% of the dev gap to the oracle floor), so
the lever that is left is **text**.

This is the shape of problem this repo prefers: an already-mined, already-localized error
population, not a speculative fix (`src/clean_gemini_predictions.py`'s docstring).

## Population, measured

| group | dev date | dev locality | test date | test locality |
|---|---|---|---|---|
| exact agreement | 83 | 75 | 1434 | 1323 |
| both-MISSING | 60 | 33 | 1011 | 472 |
| one-MISSING | 32 | 46 | 503 | 655 |
| **partial agreement** | **25** | **46** | **352** | **850** |

**1,041 test specimens** are in partial disagreement in at least one field (the union — many are
partial in only one of the two).

### Why this is worth doing, and where its ceiling is

Mean NED on the dev partial group, comparing the two readings we already have:

| field | shipped (whole-image) | per-card | oracle: pick the better one |
|---|---|---|---|
| verbatimDate | 0.1294 | 0.1811 | **0.0586** (−55%) |
| verbatimLocality | 0.2148 | 0.3332 | **0.1418** (−34%) |

Whole-image wins 25 of 46 locality cases against per-card's 16 (5 ties), so always taking
whole-image — what ships today — is already the better default, but leaves 16 cases on the table.

The load-bearing number is the oracle's **0.1418** for locality. A perfect arbiter restricted to
*choosing between the two existing candidates* still lands there, which means in a large share of
these specimens **both candidates are wrong**. That is what rules out a strict-pick design as the
primary one.

## Design

### 1. Placement — additive only

Nothing that produced the 0.02382 submission is modified. New module `src/arbitrate_partial.py`;
new runners; a **new** builder `build_arbitrated_submission_df` and
`--arbitration-predictions-path` flag alongside the existing ones in
`src/generate_submission_gemini.py`, writing a fifth output file
`data/processed/submission_gemini_arbitrated.csv`. Same one-file-per-route convention every prior
route used, for the same reason: attribute any leaderboard movement to one isolated change.

Acceptance check before anything is submitted: regenerating
`submission_gemini_wholeimage_primary.csv` must still come out **byte-identical** to the submitted
0.02382 file.

### 2. The arbitration call

One call per specimen (1,041, not 1,202 — a specimen partial in both fields is arbitrated once).
Config identical to the D4 whole-image pass, which is the validated one: original uncropped photo,
`media_resolution=MEDIA_RESOLUTION_HIGH`, `temperature=0.0`, `n_samples=1`, model rotation
`gemini-3.5-flash-lite` → `gemini-3.1-flash-lite`.

The prompt is `src.gemini_transcribe.WHOLE_IMAGE_PROMPT`'s rules verbatim (æøå never ö, "Dania"
is not a locality, `" | "` for multi-card, the confound-exclusion list) plus a candidates block
showing **both** fields' two readings — a date often disambiguates a locality, so the arbiter
should see them together.

The arbiter returns free text per field: it may pick either candidate or write a third reading.
This is the approved mandate; a strict-pick design was rejected because of the 0.1418 ceiling
above.

**Only fields that were actually in partial disagreement take the arbitrated text.** A specimen
partial in `verbatimDate` and exact in `verbatimLocality` keeps its shipped locality untouched.
This confines the text change to the mined population and keeps attribution clean.

### 3. Confidence

Arbitration produces three outcomes per field: it **confirms** the whole-image reading, **overturns**
it to the per-card reading, or writes a **third reading**. These become three sub-tiers replacing
the single partial-agreement band.

Their order is **measured, not assumed** — mean NED per outcome on the dev set, ranked by measured
risk, exactly as Phase 6's stage 1 did. Phase 6 is the precedent for why: the pre-Phase-6 design
assumed one-MISSING was the worst group and the measurement showed it was better than partial
agreement in both fields.

### 4. Validation gate, before any test-set quota is spent

Run the 63 dev specimens (~20 min, local). The dev report must break out, separately:

1. **How often a third reading is produced, and whether those rows' NED falls or rises** against
   the shipped text on the same rows. This is the specific risk of the approved free-text mandate
   and is not visible in an aggregate mean.
2. Mean NED per outcome (feeds the tier ordering in §3).
3. Full 200-row dev AURC and a paired bootstrap against the shipped configuration.

**Strict-pick is recoverable from this same data at zero additional API cost** — snap each
arbitrated string to whichever candidate it is closer to. If free text proves net-harmful, that
fallback is evaluated without re-querying.

Adoption bar, unchanged: a real mechanism plus a paired-bootstrap 90% interval over the 200 dev
specimens that excludes 0. Score the **full** dev set, never the partial subset in isolation —
AURC is a global rank metric.

### 5. Test-set collection

1,041 calls at ~19 s each: ~5.5 h on one key, ~3 h split. Kaggle notebook walking the list
forward, a local invocation walking it in reverse, meeting in the middle — the Phase 5 pattern
that closed 3,300/3,300 with zero field-level conflicts.

**The runner lives in `src/`, not inlined in the notebook cell.** The D4 runner existed only in
`notebooks/kaggle_wholeimage_primary.ipynb` Cell 5 and had to be reconstructed twice after
scratchpad losses; the notebook here imports from `src/` instead.

## Interfaces

```python
# src/arbitrate_partial.py
partial_disagreement_fields(image_file, per_card_merged, wholeimage_predictions) -> tuple[str, ...]
partial_disagreement_specimens(image_files, per_card_merged, wholeimage_predictions) -> dict
build_arbitration_prompt(candidates) -> str          # candidates: {field: (whole, per_card)}
arbitration_outcome(arbitrated, whole, per_card, is_date) -> str  # confirms/overturns/third
snap_to_nearest_candidate(arbitrated, whole, per_card, is_date) -> str   # strict-pick fallback
arbitration_tier_score(field, outcome) -> float      # constants filled in after the dev measurement
```

## Risks

- **Hallucinated third readings.** Mitigated by the explicit dev break-out in §4 and the
  zero-cost strict-pick fallback.
- **Dev is n=200 and has disagreed with the real leaderboard twice** (the partial-MISSING spike,
  which improved on dev and regressed publicly, twice). The adoption bar is the safeguard.
- **Anchoring.** Showing the arbiter two candidates may pull it toward them even when both are
  wrong. Measured directly by the third-reading rate: a rate near zero is evidence of anchoring,
  not of agreement.

---

## Outcome (2026-08-24) — measured on dev, not adopted

The 63 disputed dev specimens were arbitrated in 28.3 min (one transient `ServerError`, retried).
Full report: `conda run -n museumscat-challenge python -m src.evaluate_arbitration_dev`.

### What the arbiter did

| field | confirms whole-image | overturns to per-card | third reading |
|---|---|---|---|
| verbatimDate (n=25) | 21 (84%) | 4 (16%) | 0 (0%) |
| verbatimLocality (n=46) | 28 (61%) | 13 (28%) | 5 (11%) |

**It is heavily anchored.** It rubber-stamps the reading it was shown 61–84% of the time, and
those confirmations are not safe: they still carry mean NED 0.1185 (date) and 0.1558 (locality).
The free-text mandate — the whole reason a strict-pick design was rejected — produced 5 third
readings in total, and they were a coin flip: 2 better, 2 worse, 1 unchanged.

### Risk per outcome, which set ARBITRATION_TIERS

| field | outcome | n | shipped NED | arbitrated NED |
|---|---|---|---|---|
| verbatimDate | confirms | 21 | 0.1185 | 0.1185 |
| verbatimDate | overturns | 4 | 0.1866 | **0.1494** |
| verbatimLocality | confirms | 28 | 0.1558 | 0.1558 |
| verbatimLocality | overturns | 13 | 0.3252 | **0.2163** |
| verbatimLocality | third | 5 | 0.2575 | 0.2390 |

The order confirms < overturns < third holds in both fields, so it is a real gradient rather than
one field's noise, and `ARBITRATION_TIERS` was set from it (replacing provisional placeholders —
the first placement had arbitrated date rows at 0.70 against one-MISSING's 0.75, when the
measurement says every arbitrated date row is 4–5x riskier than that group).

**All of arbitration's value is in the overturns.** Confirmations change nothing by construction,
third readings are noise; the working set is 13 locality and 4 date specimens where the per-card
route was right and the shipped route was wrong.

### Text did improve

| field | shipped | per-card | **arbitrated** | strict-pick | oracle-pick |
|---|---|---|---|---|---|
| verbatimDate | 0.1294 | 0.1811 | **0.1234** | 0.1234 | 0.0586 |
| verbatimLocality | 0.2148 | 0.3332 | **0.1820** | 0.1832 | 0.1418 |

Locality NED on the mined population fell 15.3%, capturing 45% of the gap to the pick-oracle.
Free text beat strict-pick, but by 0.0012 — the risk it was chosen for never materialized, and
neither did the benefit.

### And AURC still barely moved

| configuration | overall | date | locality | delta | 90% CI |
|---|---|---|---|---|---|
| shipped (0.02382) | 0.03450 | 0.01247 | 0.05652 | — | — |
| arbitrated (free text) | 0.03405 | 0.01262 | 0.05548 | −0.00044 | [−0.00285, +0.00198] |
| arbitrated (strict pick) | 0.03454 | 0.01268 | 0.05640 | +0.00004 | [−0.00252, +0.00270] |
| **perfect on this population** | **0.03170** | 0.01170 | 0.05170 | **−0.00280** | — |

## Decision: not adopted, and the population is closed

The last row is what settles it. **Transcribing every disputed field perfectly is worth only
−0.00280 on dev** — less than Phase 6's confidence reordering (−0.00426), and barely above
Phase 7's rejected sweep (−0.00119). The arbiter captured −0.00044 of that, 16% of a ceiling that
was never large. No better prompt, model, or arbitration design can be worth more than 0.00280
here, and this dev set's noise band is wider than that.

**The mechanism, which generalizes beyond this round:** the partial-agreement group is already
ranked at the bottom of the submission, so the risk-coverage curve already excuses its errors —
those rows only enter the average at high coverage. Improving text in rows the ranking already
distrusts buys very little AURC. That is the precise reason D4 paid −42.8% while this pays
nothing: D4 improved text *everywhere*, including the high-confidence rows that dominate the
curve at every coverage level.

**Corollary for anything future:** "worst mean NED" is the wrong way to pick a text-quality
target under AURC. The right target is a population that is both wrong *and* currently trusted.
The partial-agreement group is wrong and already distrusted, which is exactly why it looked
attractive in Phase 6's stage-1 table and turned out not to be.

Not submitted. The Phase 6 precedent for submitting despite a failing bar does not transfer:
that change was byte-identical in text (so the downside was one slot and attribution was
unambiguous), all its variants moved the same direction, and its mechanism was an ordering error
confirmed independently in both fields. Here the text does change, strict-pick moved the wrong
way, and the ceiling is measured and small. The partial-MISSING spike — a text change on a mined
population that improved on dev and regressed publicly twice — is the closer precedent.

The code was deleted with its branch — a measured ceiling of 0.00280 means there is nothing left
for it to do, and keeping an unused third route in `src/` would only invite someone to re-run it.
`src/route_agreement.py` and `src/generate_submission_gemini.py` on `main` are therefore exactly
the code that scored 0.02382. The 63 dev arbitrations survive at
`data/processed/gemini/dev_arbitration_predictions.csv` (gitignored, local only), so the tables
above can be re-derived without re-spending quota.

The 1,041-call test pass was **never run**, which is the entire point of putting the gate before
it: the dev pass cost 63 calls and 28 minutes, and it bought the decision.
