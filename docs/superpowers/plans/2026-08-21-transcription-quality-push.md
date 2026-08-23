# Transcription quality push — attacking NED instead of rank (2026-08-21)

Branch: `research/improve-predictions-2026-08-21`. Competition ends **2026-08-31** (9 days).

## Context

The shipped submission (`data/processed/submission_gemini_rank_refined_local_vlm.csv`, public
AURC **0.04423**) is the end of a two-week run of *confidence-ranking* work
(`2026-08-07-confidence-calibration-experiments.md`, `2026-08-20-confidence-rank-refinement.md`).
This plan argues that lever is now spent and the remaining headroom is in **transcription text
quality**, specifically `verbatimLocality`.

### The measurement that settles it

Full 200-row dev set, current production text, confidence replaced by a perfect (oracle) global
ranking:

| Field | mean NED | AURC now (A3) | AURC with *perfect* ranking |
|---|---|---|---|
| verbatimDate | 0.037 | 0.017 | **0.0017** |
| verbatimLocality | 0.169 | 0.082 | **0.0237** |
| overall | — | 0.049 | **0.0127** |

Public leaderboard at the time of writing: 1st **0.00499**, top-5 spread **0.005–0.009**, us
~21st at 0.04423.

**Our floor with perfect confidence ranking (0.0127) sits above what the top five already
score.** No amount of further confidence work can reach them. The binding constraint is text.

For perfectly-ranked errors where a fraction `f` of rows are wrong, `AURC = f + (1-f)·ln(1-f)`.
So AURC 0.005 ≈ 10% of rows wrong; 0.0119 ≈ 15%. We currently have **80/200 (40%) non-exact
localities**. Roughly halving that is the target.

### Error anatomy (full 200-row dev set)

`verbatimDate` — mean NED 0.037, 170/200 exact. Residual is single-digit handwriting misreads
(`22.8.32` vs `22.8.22`, `14-5-1944` vs `14-5-1914`). Not the priority.

`verbatimLocality` — mean NED 0.169, **120/200 exact**, 62 rows with NED > 0.2. Error mass:

| Category | n | NED mass | share |
|---|---|---|---|
| both have a value, single card | 152 | 24.91 | **73.6%** |
| pred MISSING, truth has a value | 5 | 4.82 | 14.2% |
| multi-card | 13 | 3.20 | 9.5% |
| pred has a value, truth MISSING | 1 | 0.90 | 2.7% |

Three quarters of the error is ordinary single-card misreading, not merge logic or MISSING
handling. Typical failures are *plausible-Danish-place-name hallucinations*: `Jægerspris` for
`Øbjerggård`, `Hellebæk` for `Veilø Bugt`, `Charlottenlund` for `Ørholm`.

## Deficiencies found

Ordered by estimated size.

### D1. `media_resolution` and `thinking_config` are never set — FALSIFIED on card crops

`src.gemini_transcribe.call_gemini_once` passes only `system_instruction`, `temperature` and the
response schema to `GenerateContentConfig`. The installed SDK (google-genai 2.17.0) exposes both
`media_resolution` (`MEDIA_RESOLUTION_LOW/MEDIUM/HIGH`) and `thinking_config`
(`thinking_level`, `thinking_budget`), and neither has ever been used in this project.

Card crops have median resolution 1117×450 px. At default media resolution Gemini downsamples
the image before the model sees it, so real detail is being discarded ahead of inference — on a
task that is *entirely* fine-detail handwriting reading. Thinking is off by default on the
flash-lite tier.

**Result (2026-08-21): falsified for card crops.** Three arms over the same 197 dev crops, greedy
(`temperature=0`, one sample) so only the config varied. Scored through the production merge
(`scratchpad/score_arms.py`), with the whole-image fallback applied as in production:

| arm | date NED | loc NED | loc exact | loc MISSING |
|---|---|---|---|---|
| shipped (3 samples @ 0.7) | 0.0372 | 0.1655 | 122 | 34 |
| `baseline` (greedy, default res) | 0.0415 | 0.1545 | 123 | 36 |
| `high` (`MEDIA_RESOLUTION_HIGH`) | 0.0384 | 0.1528 | 125 | 36 |
| `high_thinking` (+ `thinking_level="low"`) | 0.0444 | 0.1629 | 121 | 36 |

`high_thinking` is the worst arm on both fields — thinking does not help read handwriting here,
it slightly hurts. (Its last 91 crops ran on the local key after the Kaggle key hit its daily
quota; same model, same config.)

Paired bootstrap over the 200 dev specimens, 5,000 resamples, `high` vs `baseline`:

- `verbatimDate` delta −0.0030, 90% CI [−0.0135, +0.0048] — **not distinguishable from 0**
- `verbatimLocality` delta −0.0017, 90% CI [−0.0125, +0.0087] — **not distinguishable from 0**

Row level: locality 35 rows changed, 16 better / 19 worse; date 13 changed, 6 better / 7 worse.
A coin flip, not a mechanism.

**Why it failed, and where it might still work.** A card crop is ~1117×450 px, already legible
at default tokenization, so raising media resolution has nothing to recover. The *whole-image*
path is a different story: `src.gemini_batch_runner` sends `img_f.read()` — the raw
**8192×5464 (45 MP)** photo, unscaled — so default media resolution downsamples the cards to
near-illegibility there. D1 is therefore falsified for crops and **still open for whole images**;
see D4.

**Side finding — greedy is as good as self-consistency, for text.** `baseline` (one greedy
sample) vs the shipped 3 samples at temperature 0.7: date delta +0.0043, 90% CI [−0.0041,
+0.0153]; locality delta −0.0110, 90% CI [−0.0347, +0.0125]. Neither is distinguishable from 0.
The three samples buy the *confidence* signal AURC depends on, not better text — so any pass
whose purpose is text only (an ablation, a corpus build, a second opinion) can run at 1/3 the
request cost.

### D2. Self-consistency samples one model, so its errors are correlated

`n_samples=3` at `temperature=0.7` from a single model. The local-VLM spike
(`2026-08-19-local-vlm-second-opinion-spike.md`) already established that a genuinely
independent model carries information this ensemble does not. One sample from each of 3-4
different flash-lite-class models costs the same number of requests and decorrelates both the
text vote and the confidence estimate.

### D3. The collection-wide locality prior is unexploited

Of the 90 truth locality cards we currently get wrong, **53 (59%) already appear elsewhere in
our own corpus** of 2,510 card values / 1,115 distinct — produced correctly on some other
specimen. `Ti` appears 33×, `Ørholm` 14×, `Falster` 14×, `Halk` 7×, `Al` 6×.

Blind text-snapping against this corpus was tried and rejected in Track B (`Harboøre` snapped to
the more frequent `Harboør`). The distinction proposed here is that the model gets the candidate
list **together with the crop image**, so visual evidence — absent in Track B's post-hoc snap —
arbitrates.

**Result (2026-08-21): ceiling measured, and it is low.** For each of the 62 wrong *single-card*
localities (excluding MISSING and multi-card, which a candidate list does not address), rank the
corpus by normalized edit distance to our wrong prediction and ask where the truth lands:

| corpus filter | entries | truth in top-1 | top-3 | top-5 | top-10 |
|---|---|---|---|---|---|
| all values | 1,115 | 11% | 27% | 32% | **34%** |
| seen ≥2× | 282 | 16% | 23% | 24% | 24% |
| seen ≥3× | 168 | 11% | 13% | 13% | 13% |
| seen ≥5× | 92 | 8% | 10% | 10% | 10% |

Recall saturates at k=10 and **never exceeds 34%** — so even a perfect chooser reaches only ~21
of 200 dev rows. Two reasons it is capped:

1. **The corpus is self-poisoning.** Our own wrong predictions are corpus entries, and being
   distance 0 from themselves they top their own candidate list — `Harublave`, `Hnsm.`,
   `Rūde skov` and `Gislev kloster\n Vestermose` are all in the corpus *because we produced
   them*, outranking the truths they should be corrected to.
2. **Frequency filtering makes it worse, not better** (34% → 24%), because many true localities
   are themselves rare. There is no threshold that removes our errors without removing real
   answers.

Blind snap to candidate #1 — the rejected Track B design — re-measured here: 10 better, 14
worse. Consistent with Track B's rejection.

Not dead, but demoted: a bounded ~21-row ceiling, worth revisiting only after D4.

### D3b. `clean_locality`'s trailing-punctuation rule — AUDITED, correctly tuned

8% of ground-truth locality cards genuinely end in punctuation (`Nekselø S.`,
`Ravneholm Grusgr.`, `Vind Hede Jyll.`, `Dybbøl.`), so stripping it is not free. Measured by
neutralizing only that rule: removing it helps 8 dev rows and hurts 10 (mean NED 0.1655 →
0.1675). The rule stays. Recorded so it is not re-proposed.

### D4. The per-card route discards context — CONFIRMED, and it is the big one

Multi-card count/order errors: `Grenaa | Brastrup` vs `Grenaa`, `Rotholme | Rotholme Jyll` vs
`Rotholme Jyll. | Rotholme Jyll.`, dropped `Ti |` and `NS |` prefixes.

Measured directly (2026-08-21), scoring the shipped dev predictions with and without the
whole-image fallback:

| route | date NED | date exact | loc NED | loc exact | loc MISSING |
|---|---|---|---|---|---|
| per-card only | 0.1919 | 143 | **0.3185** | 97 | **74** |
| per-card + whole-image fallback | 0.0372 | 170 | **0.1655** | 122 | **34** |

The whole-image fallback **halves** locality error and cuts MISSING from 74 to 34 — and it only
ever runs on the ~72 specimens where the per-card pass returned both fields MISSING. Whole-image
has never been evaluated as the *primary* route.

Two things compounded here, which is why this became the top candidate:

- The 95 dev specimens that had whole-image predictions were a **biased subset** — precisely the
  ones where per-card failed. Whether whole-image also won on the rest was unmeasured.
- Whole-image ran **crippled**: the raw 45 MP photo at default media resolution (see D1). The one
  place `MEDIA_RESOLUTION_HIGH` should matter is exactly the place it had never been tried.

#### The experiment (2026-08-21, 200 local requests)

Whole-image over **all 200** dev specimens, greedy (`temperature=0.0`, `n_samples=1`, justified by
D1's side finding) at `MEDIA_RESOLUTION_HIGH`. 200/200 covered, no duplicates, all on
`gemini-3.5-flash-lite`, ~52 min at rpm 12 (~22 s/request — a 45 MP upload, not the rate limiter).

| route | date NED | date exact | loc NED | loc exact | loc MISSING |
|---|---|---|---|---|---|
| per-card only | 0.1919 | 143 | 0.3185 | 97 | 74 |
| per-card + whole-image fallback (**shipped**) | 0.0372 | 170 | 0.1655 | 122 | 34 |
| **whole-image as primary** | **0.0329** | **171** | **0.1224** | **129** | 38 |
| per-card primary, whole-image fills gaps | 0.0370 | 170 | 0.1576 | 119 | 33 |
| whole-image primary, per-card fills gaps | 0.0329 | 171 | 0.1241 | 128 | 33 |

Locality mean NED **0.1655 → 0.1224**, −26% relative, 90% paired-bootstrap CI
**[−0.0707, −0.0159]** — excludes 0. Date improves too but its CI crosses 0. Letting per-card
fill whole-image's remaining gaps trades 5 MISSING for slightly worse NED; plain whole-image is
the better text.

#### The bias concern is dead — whole-image wins on every stratum

| stratum | n | per-card loc NED | whole-image loc NED | delta |
|---|---|---|---|---|
| per-card FOUND a locality | 126 | 0.1781 | 0.1222 | **−0.0559** |
| per-card returned MISSING | 74 | 0.5575 | 0.1227 | −0.4348 |
| single-card specimens | 57 | 0.4406 | 0.1027 | −0.3378 |
| multi-card specimens | 143 | 0.2698 | 0.1302 | −0.1396 |

It wins on the 126 specimens where per-card *succeeded*, so the gain is not an artifact of the
biased subset. It also wins on multi-card specimens (−0.1396), where per-card's explicit
card enumeration was supposed to be structurally superior — that advantage does not materialize.
Same pattern for `verbatimDate` on all four strata.

#### The oracle floor moves below the barrier

The measurement that opened this investigation was that production text's oracle-rank AURC
(0.0121) sits *above* the leaderboard top-5's 0.005–0.009 — no confidence work could ever reach
them. Whole-image text changes that:

| text route | date oracle | loc oracle | overall oracle |
|---|---|---|---|
| production (shipped) | 0.0017 | 0.0224 | 0.0121 |
| **whole-image** | 0.0016 | **0.0139** | **0.0077** |

0.0077 is inside the top-5 band. This is the structural unblock the whole push was looking for.

#### Confidence, rebuilt from signals that already exist (zero quota)

A whole-image-primary route loses the per-card structure the entire Track A/A3 confidence stack
is keyed to, and at `n_samples=1` it has no self-consistency of its own — so the text win could
have died on the way to AURC. It does not. Every signal needed is already on disk:

- `agree` — `1 − NED` between the whole-image text and the **independent per-card merge**. This
  is Track A3's own pattern with the roles swapped, and at full 200/200 coverage instead of
  Qwen's partial 152/197. Both routes independently returning MISSING counts as agreement (the
  structural-missing-tier finding: those rows are unusually accurate), one-sided MISSING as
  maximal disagreement.
- `corpus` — `src.locality_corpus.corpus_support_score`, unchanged.
- `vlm` — `src.local_vlm_signal` agreement, unchanged.

Dev AURC, both sides scored through the same code path in one script (the Track A3 side rebuilt
from `src.evaluate_gemini_dev`'s logic, reproducing its recorded 0.0493 to within 0.0006):

| candidate | date | loc | overall |
|---|---|---|---|
| Track A3 — production text (**shipped**, public 0.04423) | 0.0170 | 0.0803 | 0.0487 |
| whole-image + `agree` (both-MISSING=0.5) | 0.0226 | 0.0620 | 0.0423 |
| whole-image + `agree` (both-MISSING=1.0) | 0.0166 | 0.0642 | 0.0404 |
| whole-image + `agree` + `corpus` | 0.0166 | 0.0557 | 0.0362 |
| **whole-image + `agree` + `corpus` + `vlm`** | **0.0145** | **0.0561** | **0.0353** |
| whole-image + `agree` + `vlm` | 0.0145 | 0.0622 | 0.0384 |

**All five** confidence designs beat the shipped Track A3, so the conclusion does not depend on
which one is picked — that is the selection-effect-free reading, and it is the one to trust. The
best (0.0353, −27% vs 0.0487) was chosen by point estimate and then bootstrapped, which inflates
its apparent edge; its paired-bootstrap CI vs Track A3 is **[−0.0279, +0.0002] overall** and
**[−0.0516, +0.0003]** for locality — both land just barely on the inconclusive side of the 90%
bar, i.e. right at the threshold rather than clear of it.

So the strict adoption bar is met on **text** (locality NED CI excludes 0) and on the **oracle
floor**, but not quite on end-to-end dev AURC. Given that the leaderboard is the real arbiter
here, that is a case for spending a test-set pass to get a real read, not for stopping.

#### Unattributed variable

Route (whole-image vs per-card) and `media_resolution` (HIGH vs default) changed together in this
experiment. A control arm would settle which lever does the work, but it would not change what to
run next — the winning config is HIGH either way, and D1 already falsified HIGH on crops — so it
was not spent.

### D5. Text-encoding noise — MEASURED, zero risk

Predictions carry Unicode denormalization, HTML entities and control characters:

```
'Nivå' (NFD) vs 'Nivå' (NFC)      -> NED 0.400 on visually identical strings
'B&#248;rstinge Mose'              -> NED 0.316
'Gislev kloster\n Vestermose'      -> embedded newline
'Neksel\x00 S'                     -> embedded NUL
```

`html.unescape` + `unicodedata.normalize("NFC", …)` + control-char strip + whitespace collapse,
measured on the full dev set: **3 rows improved, 0 worsened**; locality mean NED 0.1692 → 0.1655,
AURC 0.1172 → 0.1153 (base config). Small, but free and strictly non-negative.

## Rejected during this investigation

**Visual near-duplicate clustering of card crops.** Premise was that specimens form collection
series sharing a physically identical label, so a within-cluster vote would fix outliers.
Measured with 256-bit dHash + 24×24 normalized-thumbnail cosine over all 3,464 transcribed
crops, at four thresholds: at the loosest usable setting only **11% of crops cluster at all**,
and intra-cluster locality agreement is only **70%**. The 242 predicted `Dyrehaven` values are
242 physically distinct cards from different collectors and decades, not a repeated print run.
Premise was wrong; not pursued. Probe kept at
`scratchpad/probe_visual_clusters.py` + `probe_cluster_eval.py` (throwaway, not in `src/`).

## Budget

Two independent Gemini API keys.

| Class | Models | Per key/day | Total available (9 days) |
|---|---|---|---|
| flash-lite | `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, … | ~500 **per model** | **~36,000** |
| strong flash | `gemini-3.7-flash`, `gemini-3.6-flash` | 20 each = 40 | **~720** |
| Pro | `gemini-3.1-pro-preview` | 0 (429s on free tier) | — |

**The ~500/day is per *model*, not per key.** The ~2,000/day figure is the sum across four
flash-lite-class models, and only `src.gemini_batch_runner.ModelRotation` can actually spend it
that way. Any experiment that must hold the model fixed — every arm of a config A/B, for
instance — is capped at ~500 requests per key per day, i.e. **two 197-crop dev arms per key per
day**. Confirmed the hard way on 2026-08-21: the Phase 1 sweep was sized at 591 requests against
the 2,000 figure and hit a daily 429 118 crops into its third arm.

A batch that hits a daily 429 must stop, not retry — `is_daily_quota_exhausted` distinguishes it
from a per-minute 429. Retrying a daily 429 four times per remaining item cost ~1.5h of Kaggle
session on that first run.

**Experiments and production passes draw on the same per-model daily budget, so a day spent
measuring is a day not spent transcribing.** Demonstrated on 2026-08-21: Phase 5's first day
yielded ~1,000 rows, not the ~2,000 the table above suggests, because *both* keys' 
`gemini-3.5-flash-lite` had already been drained by that same day's experiments — Phase 1's
media-resolution sweep on the Kaggle key, the D4 dev pass on the local one. Each key had exactly
one fresh model left, and each returned almost exactly 500 rows. Plan the last production pass
for a day with no experiments on it.

One full test pass = **3,267** card crops (or 3,300 whole images). One full dev pass = **~200**.
So: ~20 dev experiments per day, and room for ~4 full test passes at the end.

Strong-flash models cannot carry a production pass (720 requests ≈ 22% of one). They are
measurement and targeted-escalation instruments only.

## Plan

Every evaluation scores the **full 200-row dev set**. Never a subset — see the partial-MISSING
spike, where an isolated 24-row measurement read as 0.0281→0.0166 but was +0.00004 on the full
set. And dev has disagreed with the real leaderboard twice, so dev is a filter for what to
submit, not proof.

### Phase 0 — text sanitation (zero quota)

Add the normalization above to `src.clean_gemini_predictions`. Expected: locality mean NED
0.1692 → 0.1655. Already measured; just needs implementing with tests.

### Phase 1 — `media_resolution` + `thinking_config` A/B (~600 requests) — FIRST

Re-transcribe the 197 dev card crops under each config and compare mean NED against the
flash-lite baseline of 0.037 / 0.169:

1. baseline (as shipped) — re-run for a same-day control
2. `media_resolution=MEDIA_RESOLUTION_HIGH`
3. `HIGH` + thinking enabled

Greedy decoding (`n_samples=1`) so the comparison isolates the config, not sampling noise.
Watch for the free-tier *token* quota (`generate_content_free_tier_input_token_count`) — HIGH
resolution raises input tokens per request, and that quota is separate from requests/day.

**Decision gate:** if HIGH moves locality mean NED materially, it reframes Phases 2-4 and the
full test re-run happens at that config. If it does nothing, proceed to D2/D4 experiments.

### Phase 2 — whole-image as the primary route (D4) — DONE, and it won

200 local requests. Result in D4 above: locality NED 0.1655 → 0.1224 (CI excludes 0), oracle
floor 0.0121 → 0.0077, dev AURC 0.0487 → 0.0353 with a zero-quota confidence rebuild. Wins on
every stratum including where per-card succeeded.

D2 (multi-model ensemble) was **not** run and is now lower priority: it was a way to decorrelate
the *per-card* confidence vote, and the per-card route is no longer the primary one.

### Phase 3 — corpus-augmented second pass (D3) — demoted

D3's ceiling was measured at ~21 rows and its candidate-list recall saturates at 34% (see D3).
Not worth quota while Phase 2's result is unshipped.

### Phase 4 — targeted escalation with the strong-flash budget

First ~40 requests: run `gemini-3.7-flash`/`3.6-flash` against the 20 worst dev locality errors
to establish a ceiling. If the strong model reads them correctly, spend the remaining ~680 on
the lowest-confidence test rows. If it fails too, those rows are genuinely illegible and effort
returns to confidence ranking for them.

### Phase 5 — full test whole-image pass, regenerate, validate, submit — THE NEXT MOVE

3,300 whole-image requests, greedy, `MEDIA_RESOLUTION_HIGH` — exactly the dev config. Cost is
~1.7 days of quota out of 9 remaining, and every confidence signal the design needs already
exists at full test coverage (`test_card_predictions.csv` 3267/3267 for `agree`,
`test_local_vlm_card_predictions.csv` 3267/3267 for `vlm`, corpus rebuilt from predictions).
`n_samples=1` deliberately: D1's side finding says extra samples buy confidence, not text, and
the confidence here comes from cross-route agreement instead.

Wall time, not quota, is the binding constraint: ~22 s/request serial ≈ 20 h, past Kaggle's 12 h
session cap. Split across both keys — the Kaggle kernel on one, local on the other — via the
existing progress-dataset round-trip.

`kaggle competitions submit` stays user-triggered. 10/day, 2 final submissions, and neither the
classical nor any Gemini submission has been marked final yet.

#### Progress (day 1, 2026-08-21): 1,157/3,300

Both keys ran in parallel, walking `test.csv` from opposite ends so they need no coordination
until they meet: the Kaggle kernel `matiaspoullain/museumscat-wholeimage-primary` forward (497
rows, stopped cleanly on quota), a local invocation in reverse (660 rows, same). Zero overlap,
zero field-level conflicts on merge into
`data/processed/gemini/test_wholeimage_primary_predictions.csv`.

Both stopped at ~500 because both keys had only one fresh model left — see the Budget section's
note on experiments spending the same day's production budget. Days 2 and 3 have all four
model-days available, so ~2,000 then the remaining ~200.

The merged file is seeded into the `museumscat-gemini-progress` dataset under the name the
notebook's Cell 4 looks for (`test_wholeimage_primary_kaggle.csv`), so day 2's Kaggle session
resumes past everything *both* keys did rather than redoing the first 497.

Code is already wired and tested ahead of the data:
`src/route_agreement.py` + `--wholeimage-primary-predictions-path` on
`src.generate_submission_gemini`, 329 tests passing. An end-to-end run against the real
mid-collection files is what caught the one real bug: uncovered specimens fall back to per-card
text, so scoring agreement between per-card text and *itself* returned a trivial 1.0 and gave
unverified rows maximum confidence. They now fall back to per-card self-consistency instead.

#### Progress (day 2, 2026-08-22): 3,064/3,300

With all four model-days available the rate roughly doubled as projected: Kaggle 963 new rows,
local 942, both stopping cleanly on `[stop] every model exhausted`. Merged total **3,064/3,300
(92.8%)**, still **0 field-level conflicts** — now across 2,800 overlapping keys spanning two
days, two API keys and both models. The two halves agree on the text of every row they both
transcribed, which is what greedy `temperature=0.0` decoding is supposed to buy and is worth
having confirmed at this scale rather than assumed.

The 236 remaining specimens are a single contiguous block (test.csv index 1460..1695) — exactly
the signature of a forward and a reverse walk converging, and a cheap invariant to check that
the two halves really are covering disjoint ranges.

Throughput is bounded by **request latency, not quota or `--rpm`**: ~19s per call for a 45 MP
upload, so ~190 rows/hour per key regardless of how the rate limiter is set. A day's ~1,000-call
budget therefore needs ~5.5h of wall clock to spend, which is why a Kaggle session's 12h cap has
never been the binding constraint here.

Prediction stats over the 3,064 collected: `verbatimDate` MISSING 31.0%, `verbatimLocality`
MISSING 15.3% (vs. the shipped per-card+fallback submission's 31.0% / 15.9%) — consistent with
D4's dev-set finding that the whole-image route does not lose coverage relative to per-card.

Day 3 needs one short run of 236 rows (~1.2h on a single key), after which the submission
command in this phase is unblocked.

Operational note for the rebuild: this session's scratchpad had been wiped, so the local runner
was reconstructed from the notebook's Cell 5 rather than from the lost script. That is the
reason the runner lives inlined in the notebook in the first place — it is the authoritative
copy, and the local half is defined as "the same thing, walking backwards". The rebuild also
fixed a latent waste: the local half now seeds its skip-set from the *merged canonical* file,
not just its own output, so when the two walks finally meet it skips the other key's rows
instead of re-transcribing them.

#### Progress (day 3, 2026-08-23): 3,300/3,300 — COMPLETE

One local run, 236 rows in 34 minutes, no retries. Final merge: **3,300/3,300 distinct
specimens, 0 field-level conflicts across 3,064 overlapping keys.** The two independent keys
never disagreed on a single field across the whole test set.

`data/processed/submission_gemini_wholeimage_primary.csv` generated via
`python -m src.generate_submission_gemini --wholeimage-primary-predictions-path ...
--local-vlm-predictions-path ...`. The whole-image pass covers **3,300/3,300, so 0 specimens
fall back to per-card text** — the uncovered-row branch (and the trivial-self-agreement bug it
was written for) is now dead code on this submission, though it stays for correctness.

Validation: `validate_submission` clean, exact column order, 0 duplicate rows, 0 empty values,
0 null confidences, confidence in [0,1] both fields. 329 tests pass. Versus the shipped 0.04423
submission: `verbatimDate` text changed 794/3300, `verbatimLocality` 1148/3300 — this is the
first candidate this round whose *text* differs, every prior one having been confidence-only.
MISSING rates barely move (date 31.0%→30.9%, locality 15.9%→15.4%).

**Label-free sanity check on the text change.** With no test ground truth, an eyeball of the
first 12 changed localities looked alarming (`Dyrehaven` → `Dyrepaven`/`Dyretiaven`) — but
sorting by `image_file` groups one collection, so that was a cluster, not a sample. Two checks
corrected it: a random sample is dominated by genuine repairs (`Lynghornbaevaulnmu` →
`Dyrehaven`, `Tiüs Sø` → `Tiis Sø`, `MISSING` → `Asserbo Overdrev`) against a minority of real
regressions; and in aggregate the **singleton rate falls 33.1% → 28.5%** (distinct values
1228 → 1122). Real Danish place names recur across specimens and character corruptions do not,
so a falling singleton rate is evidence of less noise, not more. Consistent with D4's dev-set
measurement, which is the actual evidence — this only checks the route did not behave
differently at test scale.

**Submitted 2026-08-23 (user-triggered): public AURC 0.04423 → 0.02530** (ref `55721990`),
**−42.8%**. Rank ~11th; 10th place is 0.02524, 0.00006 away.

This is the only step in the project whose public delta *exceeded* its dev projection
(−42.8% public vs −27.5% dev). Every prior round landed at or below its dev signal, and the
partial-MISSING spike landed on the wrong side of zero twice. The plausible reason is that this
was the first change to alter **text** rather than confidence rank: lower per-row NED improves
AURC through both the risk axis and the ranking problem at once, whereas a pure re-rank can only
improve one of them. It also means dev-set deltas are not a reliable *magnitude* estimator
across change types — only a direction-and-significance test, which is how the adoption bar
already treats them.

The oracle-rank floor for this text is **0.0077**, versus 0.02530 shipped. Roughly two thirds of
the remaining gap is therefore still ranking, not transcription — but the ranking machinery from
Track A/A3 was built against per-card self-consistency tiers that no longer exist. That
re-measurement is the next move, and it needs the dev whole-image pass re-collected first (this
plan's own dev arm was lost with a session scratchpad; see the day-3 note).

## Phase 6: re-measuring the rank layer on the new route (2026-08-23) — SHIPPED, 0.02530 → 0.02382

Track A/A3's rank rules were built against per-card self-consistency tiers that the whole-image
route does not have (`n_samples=1`), so the whole ranking layer needed re-deriving. D4's own dev
arm had been lost with a session scratchpad, so the 200-specimen whole-image dev pass was
re-collected first (200 requests, 21.6 min).

**Stage 1, the diagnostic that drove everything else.** Mean NED by agreement group, full dev set:

| group | n (date) | NED date | n (loc) | NED loc |
|---|---|---|---|---|
| exact text agreement | 83 | 0.0132 | 75 | 0.0464 |
| both-MISSING | 60 | 0.0167 | 33 | 0.1212 |
| one-MISSING | 32 | 0.0296 | 46 | 0.1750 |
| partial agreement | 25 | 0.1294 | 45 | 0.2195 |

Two orderings in the shipped design were wrong, each visible independently in both fields:

1. **both-MISSING sat at 1.0**, sharing the top tier with two routes reading identical text. True
   for dates (0.0167 vs 0.0132 — a specimen genuinely without a date is common and both routes
   see that), false for localities (0.1212 vs 0.0464, 2.6x riskier). The fix has to be
   field-specific, which is why `ROUTE_TIERS` is keyed by field.
2. **one-MISSING sat at 0.0**, the bottom tier, while measuring *better* than partial agreement
   in both fields. One route reading text the other missed is weak corroboration, not maximal
   disagreement. Partial agreement — two routes reading the same card differently — is the
   genuinely worst group of the four, and it used to float above one-MISSING.

**Stage 2, design grid** (full 200 rows, identical text throughout, so every delta is pure rank):

| design | overall | date | locality |
|---|---|---|---|
| A shipped (agreement + corpus/vlm) | 0.0381 | 0.0144 | 0.0618 |
| B risk-ordered tiers | 0.0349 | 0.0128 | 0.0570 |
| C both-MISSING demotion only | 0.0377 | 0.0152 | 0.0603 |
| D one-MISSING promotion only | 0.0354 | 0.0121 | 0.0587 |
| E risk-ordered + coverage penalty | 0.0375 | 0.0158 | 0.0592 |
| **F risk-ordered + per-card conf tiebreak** | **0.0345** | 0.0125 | 0.0565 |
| *oracle rank ceiling (same text)* | *0.0080* | | |

Attribution matters here: most of the gain is the **one-MISSING promotion** (D alone, 0.0354),
not the both-MISSING demotion (C alone, 0.0377, barely moves). The coverage penalty (E) actively
hurts — it was designed for a per-card text source that no longer exists.

**Stage 3, paired bootstrap** — every candidate improves, **none excludes 0**:

| design | delta | 90% CI |
|---|---|---|
| B | −0.00396 | [−0.00892, +0.00081] |
| C | −0.00132 | [−0.00645, +0.00330] |
| D | −0.00265 | [−0.00572, +0.00040] |
| E | −0.00114 | [−0.00579, +0.00353] |
| F | −0.00426 | [−0.00958, +0.00081] |

**Submitted anyway, deliberately, as an exploratory submission that did not clear the adoption
bar** — and it worked: **public AURC 0.02530 → 0.02382** (ref `55722819`), −5.8%, versus dev's
−9.4%. Rank 10th. The judgment call rested on three things the bar does not encode: the text is
byte-identical to the 0.02530 submission (0 rows changed in either field), so the downside was
one submission slot and the attribution is unambiguous; all five candidates moved the same
direction; and the mechanism is an *ordering* error confirmed independently in both fields, not
a fitted magnitude — which matters because AURC reads nothing but rank, so `ROUTE_TIERS`'
constants only encode an ordering.

Implementation: `src.route_agreement.ROUTE_TIERS` and `route_tier_score` (new),
`wholeimage_primary_confidence` gaining a `per_card_confidence` argument;
`route_agreement_score` is left untouched as the raw agreement primitive. 335 tests pass.

**Still open:** the oracle ceiling for this text is 0.0080 against 0.02382 shipped, so roughly
two thirds of the remaining gap is still ranking. `partial_scale` (where the partial-agreement
continuum interleaves with the MISSING groups) is the one genuinely fitted constant and was
never swept.

## Adoption bar

Unchanged from the rank-refinement round: a change ships only with a real mechanism plus a
paired-bootstrap delta over the 200 dev specimens whose 90% interval excludes 0. Dev-set gains
alone are not sufficient — the partial-MISSING spike improved on dev twice and lost on the real
leaderboard both times.
