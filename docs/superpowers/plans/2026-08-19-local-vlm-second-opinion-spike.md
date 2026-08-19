# Local VLM Second-Opinion Confidence Spike

**Branch:** `spike/kaggle-vlm-second-opinion` (kept for reference only — not
merged, not going to be used further). This doc is the permanent record of
what was tried and found; the branch just holds the working notebook.

**Goal:** the Gemini pipeline's confidence is self-consistency across 3 samples
from the *same* model, so its errors are correlated with each other. Could a
genuinely different, independent model — run entirely on Kaggle's free GPU, no
Gemini quota spent — serve as a second "vote" that improves AURC? This is
exactly the kind of new-information signal `2026-08-07-confidence-calibration-
experiments.md` identified as the only thing that *could* still move AURC
(any monotonic reshaping of self-consistency alone provably can't, since AURC
only depends on confidence rank order).

**Answer, upfront: no — not with any simple combiner.** The correlation this
spike was looking for is real and reproducible at full dev-set scale, but every
way tried to actually use it (5 variants, see below) either didn't beat Gemini's
own confidence or only matched it. Full writeup below for whoever considers
this again.

## What was built

A Kaggle notebook (`notebooks/kaggle_local_vlm_spike.ipynb` on the spike
branch) running **Qwen2.5-VL-7B-Instruct** (4-bit quantized via `bitsandbytes`)
over the full 200-specimen dev set — same prompts, same `self_consistency_scores`
formula, same `PROMPT`/`WHOLE_IMAGE_PROMPT` confound rules as
`src.gemini_transcribe`, so its output is directly comparable to Gemini's. 10
samples per item (vs. Gemini's production default of 3), parallelized across
both of a Kaggle T4x2's GPUs.

Getting this to run reliably took **9 push/debug iterations** — worth recording
since the failure modes are non-obvious and would bite any future from-scratch
local-VLM-on-Kaggle attempt the same way:

| version | change | result |
|---|---|---|
| v1 | Qwen2.5-VL-3B, `torch_dtype=bfloat16` | OOM at model load — `torch_dtype` silently ignored by a `transformers` version pulled in by an unpinned `pip install -U`, model loaded in fp32 (~13.6GB) instead |
| v2 | switched to "2B" to save memory | 401 Unauthorized from HF — **`Qwen2.5-VL-2B-Instruct` doesn't exist** (only 3B/7B/32B/72B were ever released); HF returns 401, not 404, for a nonexistent repo when unauthenticated, so it looked identical to an auth problem |
| v3 | forced `token=False` (anonymous HF access) | same 401 — there was never a token problem, see v2 |
| v4 | back to 3B, 4-bit `bitsandbytes` quantization | worked — loaded and ran |
| v5 | scaled to full dev set (268 items), 2 GPUs via **threading** | both GPUs eventually died (cuda:1 genuine OOM, cuda:0 `device-side assert`) — unbounded image resolution (no `min_pixels`/`max_pixels`) fragmenting memory across hundreds of differently-shaped inputs |
| v6 | added `min_pixels`/`max_pixels`, switched **3B → 7B** per request | **both GPUs died on their first item** with `CUDA error: illegal memory access` — a **bitsandbytes thread-safety bug**: two threads each driving a 4-bit-quantized model on a different GPU, but sharing one OS process, corrupt shared CUDA/bitsandbytes state. More kernel launches per forward pass (7B vs 3B) meant a bigger window for the race, hence failing immediately instead of after a while |
| v7 | **real process isolation** — each GPU's work runs as its own `subprocess.Popen`, no threads | ran clean, no crashes, but **all 71 whole-image items + 45 card crops OOM'd deterministically** (100% failure, not intermittent) |
| v8 | added unconditional per-item `gc.collect()`/`empty_cache()` (even on error, not just every-20-on-success) | **zero improvement** — the exact same 116 items failed again, identical error signatures. This ruled out fragmentation/accumulation as the cause |
| v9 | **split each item's 10 samples into 2 sub-batches of 5** instead of one batch of 10 (same total samples, same resolution) | **268/268 items, 0 errors.** Root cause: those specific 116 images individually need more peak memory at batch=10 than a T4+7B has, deterministically, regardless of run order — sub-batching just halves the peak per call |

Also built: a resumability mechanism (previous run's output re-uploaded into
the Kaggle input dataset as `previous_card_results.csv`/
`previous_wholeimage_results.csv`; the notebook skips anything already done)
that made v7→v8→v9 practical instead of re-running the full set from scratch
each time.

## Findings

**A 30-crop pilot (Qwen2.5-VL-3B) suggested a real signal on `verbatimLocality`**
specifically — the project's dominant error source — where Gemini's own
confidence barely correlates with its own correctness (Spearman ≈ -0.2), but
the local VLM's confidence correlated with *Gemini* being right (≈ -0.38), and
was well self-calibrated to its own correctness (≈ -0.7) despite much lower raw
accuracy than Gemini.

**At full dev-set scale (Qwen2.5-VL-7B, n=124 single-card specimens + n=71
whole-image specimens) the signal is real but weaker than the pilot suggested**
— small-n noise inflated the pilot's numbers, consistent with this project's
established finding that its 200-row dev set has significant fold-to-fold
variance:

| pass / field | Gemini AURC (own conf) | local VLM AURC (own conf) | `local_conf` vs `gemini_error` (Spearman) | `gemini_conf` vs `gemini_error` (Spearman) |
|---|---|---|---|---|
| card / date | 0.0741 | 0.1491 | -0.237 | -0.292 |
| card / locality | 0.1982 | 0.2524 | -0.234 | -0.360 |
| whole-image / date | 0.0082 | 0.0507 | -0.241 | -0.497 |
| whole-image / locality | 0.0792 | 0.2931 | -0.167 | -0.446 |

The local VLM's confidence is consistently, moderately informative about
Gemini's correctness (all four Spearman values negative, -0.17 to -0.24) — the
correlation is real, not noise. But Gemini's *own* confidence is **also**
informative about its own correctness in every case (and more so than the
pilot's small sample suggested), leaving less room for a second signal to add
value than the pilot implied — especially in whole-image, where Gemini is
already well-calibrated on its own.

**Concrete wins do exist.** Mining actual disagreements (this project's usual
convention — real errors before designed fixes) found 19 cases across the full
dev set where the local VLM was right and Gemini was wrong, clustering in
patterns worth knowing about even though the aggregate signal didn't translate
into a usable combiner:

- **Handwritten digit misreads**: `26-2-1951` (Gemini) vs correct `26-7-1951`,
  `10-5 1935` vs correct `10-9.1935`, `25-5-1969` vs correct `25-9-1969`.
- **Exactly the æøå bug the prompt warns against**: `Köbenhavn` (ö) vs correct
  `København` (ø); `Froslev` (dropped) vs correct `Frøslev`; one case where
  Gemini emitted a raw unescaped HTML entity, `B&#248;rstinge Mose`, instead of
  `Børstinge Mose`.
- **Short 1-3 letter locality abbreviations** (`Al`, `Ti`) that Gemini
  discarded as `MISSING` — exactly the edge case the prompt explicitly calls
  out as tricky.
- **Gemini hallucinations**: inventing `Uds. Dreus.` where truth was
  `MISSING`; appending `Gønget` to `Jydelejet Møn` that wasn't there.

**But mining the reverse case (Gemini said `MISSING`, local VLM found *some*
text) shows why a naive "prefer the VLM when it disagrees" rule doesn't work:**
46 such cases exist, but only 3 were actually correct (the abbreviations
above). The other 43 are the local VLM hallucinating — and not randomly: it
repeatedly misreads a collector's name (`Schiødte`/`Schiøtte`) as a locality,
picks up habitat/substrate notes (`cow dung on grassland`), or reads catalog
numbers as dates (`608.`, `410.`, `2704`) — precisely the confound categories
the shared prompt explicitly tells both models to exclude. The local VLM is
*less* disciplined about ignoring these than Gemini is.

## Combiners tried (all against Gemini's own confidence as baseline)

| combiner | card/date | card/locality | whole-image/date | whole-image/locality |
|---|---|---|---|---|
| Gemini alone (baseline) | 0.0741 | 0.1982 | 0.0082 | 0.0792 |
| `(gemini_conf + local_conf) / 2`, unconditional | 0.0709 (better) | 0.1923 (better) | 0.0090 (worse) | 0.1460 (much worse) |
| `min(gemini_conf, local_conf)`, unconditional | 0.0731 (better) | 0.1994 (worse) | 0.0095 (worse) | 0.1637 (much worse) |
| only switch to local's text where it contains æøåÆØÅ, text-level not confidence | — (dates never trigger) | 0.4206 vs 0.3865 mean NED (worse) | — | 0.2161 vs 0.1470 mean NED (worse) |
| blend only where `gemini_conf < 1.0`, else keep Gemini's confidence | 0.0737 (tiny better) | 0.1976 (tiny better) | 0.0082 (unchanged) | 0.0794 (unchanged) |
| blend only where `gemini_conf == 1/3` (its minimum value) | 0.0741 (unchanged) | 0.1981 (unchanged) | 0.0082 (unchanged) | 0.0792 (unchanged) |

Why the unconditional blend/min hurt whole-image so badly: most of the damage
came from rows where Gemini was **already at maximum confidence and already
correct** — blending in the noisier local signal there only ever adds risk,
never removes it. Gating on `gemini_conf < 1.0` (row 5) eliminates that damage
almost entirely, but also eliminates almost all of the gain — the real
information the local VLM has turns out to concentrate in exactly the rows
Gemini is already sure about (rank-reordering among Gemini's minimum-confidence
bucket, row 6, moves the AURC needle by nothing, since that bucket is already
scored last in the risk-coverage sweep regardless of internal order). The
æøå-content text-level swap (row 4) failed for a related reason: "contains a
Danish letter" doesn't distinguish a genuine ø/ö correction from the local
VLM hallucinating some other Danish-sounding word (most often the collector
name `Schiødte`), so it fires about as often on the bad case as the good one.

## Outcome

**Not integrated into `src/generate_submission_gemini.py` or any production
path.** The signal is real (documented above) but every combiner tried is
either a wash or a net loss on full dev-set AURC — worse than this project's
2026-08-07 finding that multi-feature *fitted* calibrators barely justified
themselves against dev-set noise, since a hand-picked rule has even less
principled reason to generalize than a fitted one did.

If this is revisited: the resumable, subprocess-isolated, sub-batched Kaggle
pipeline in `notebooks/kaggle_local_vlm_spike.ipynb` (branch
`spike/kaggle-vlm-second-opinion`) still works end-to-end and could be rerun
cheaply (no Gemini quota) to extend to the full 3300-image test set, or a
smarter (fitted, not hand-picked) combiner could be tried against the raw
per-sample data already collected (`raw_samples_json` columns in
`local_vlm_spike_card_predictions.csv`/`local_vlm_spike_wholeimage_predictions.csv`,
same shape as Gemini's own — reusable the same way
`src.calibrate_gemini_confidence` already reuses Gemini's). Neither is planned;
this doc exists so that decision doesn't have to be re-derived from scratch.
