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

### D1. `media_resolution` and `thinking_config` are never set — PRIME SUSPECT

`src.gemini_transcribe.call_gemini_once` passes only `system_instruction`, `temperature` and the
response schema to `GenerateContentConfig`. The installed SDK (google-genai 2.17.0) exposes both
`media_resolution` (`MEDIA_RESOLUTION_LOW/MEDIUM/HIGH`) and `thinking_config`
(`thinking_level`, `thinking_budget`), and neither has ever been used in this project.

Card crops have median resolution 1117×450 px. At default media resolution Gemini downsamples
the image before the model sees it, so real detail is being discarded ahead of inference — on a
task that is *entirely* fine-detail handwriting reading. Thinking is off by default on the
flash-lite tier.

Hypothesis, not yet a verified gain. It is the cheapest thing to falsify and it would change the
value of everything below it, so it goes first.

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

### D4. The per-card route discards context

Multi-card count/order errors: `Grenaa | Brastrup` vs `Grenaa`, `Rotholme | Rotholme Jyll` vs
`Rotholme Jyll. | Rotholme Jyll.`, dropped `Ti |` and `NS |` prefixes. Whole-image is currently
used *only* as a both-MISSING fallback, yet adding it moved dev date mean NED 0.192 → 0.037,
which suggests the crop route loses a great deal.

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
| flash-lite | `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, … | ~2,000 | **~36,000** |
| strong flash | `gemini-3.7-flash`, `gemini-3.6-flash` | 20 each = 40 | **~720** |
| Pro | `gemini-3.1-pro-preview` | 0 (429s on free tier) | — |

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

### Phase 2 — multi-model ensemble and input framing

Only after Phase 1 sets the base config.

- D2: 1 sample × 4 flash-lite-class models vs. today's 3 samples × 1 model, same request count.
- D4: tight crop (current) vs. padded crop vs. whole image, as the primary route.

### Phase 3 — corpus-augmented second pass (D3)

For low-confidence cards, re-query with the crop plus the top-k collection candidates by edit
proximity, instructed to keep its own reading unless a candidate matches what it sees.

### Phase 4 — targeted escalation with the strong-flash budget

First ~40 requests: run `gemini-3.7-flash`/`3.6-flash` against the 20 worst dev locality errors
to establish a ceiling. If the strong model reads them correctly, spend the remaining ~680 on
the lowest-confidence test rows. If it fails too, those rows are genuinely illegible and effort
returns to confidence ranking for them.

### Phase 5 — full test re-transcription at the winning config, regenerate, validate, submit

`kaggle competitions submit` stays user-triggered. 10/day, 2 final submissions, and neither the
classical nor any Gemini submission has been marked final yet.

## Adoption bar

Unchanged from the rank-refinement round: a change ships only with a real mechanism plus a
paired-bootstrap delta over the 200 dev specimens whose 90% interval excludes 0. Dev-set gains
alone are not sufficient — the partial-MISSING spike improved on dev twice and lost on the real
leaderboard both times.
