# Test: quantitative OCR engine/recognizer comparison (real NED/AURC, not visual)

## Why

Visual inspection (`ocr_benchmark_doctr/`, `ocr_benchmark_doctr-multilingual/`,
`ocr_benchmark_easyocr/`, `ocr_benchmark_pytesseract/`) on 5 hand-picked
images gave mixed, inconclusive impressions between doctr and easyocr, and
raised a concern (not a confirmed number) that the Danish-capable doctr
recognizer might be overconfident on wrong handwriting reads. User asked to
specifically keep pursuing doctr and search for a Danish-capable
model/plugin before giving up on it (see `ocr_benchmark_doctr-multilingual/`
and `ocr_benchmark_doctr-multilingual-fastdet/` for that search + the
`Felix92/doctr-torch-parseq-multilingual-v1` find). This test replaces
eyeballing with the actual competition metric on a bigger, unseen sample.

## What was tried

`src/score_ocr_dev.py <engine> 25` -- new script, not the final Task 7
pipeline. For 25 dev-set rows (real ground truth, sampled with a fixed
seed, none in the earlier 5-image visual sample): detect ROI (Task 5) ->
run the OCR engine -> `classify_field` (Task 3) each word/phrase snippet ->
join same-field snippets left-to-right/top-to-bottom with a plain space ->
`best_pipe_ned` per field against ground truth -> `aurc` using mean snippet
confidence as the field's confidence. This is an approximation (word-level
`classify_field` misclassifies bare surnames/etc as "locality" fairly
often, and space-joining isn't the real multi-card pipe-merge from Task 4)
-- not a substitute for the real Task 7 pipeline, but a fair apples-to-apples
way to A/B the 4 candidate engine configs against each other, since the
same approximation noise applies equally to all of them.

## Result

| config | date NED | date AURC | locality NED | locality AURC |
|---|---|---|---|---|
| doctr (default, French vocab) | **0.615** | **0.315** | 0.768 | 0.704 |
| doctr-multilingual (db_resnet50 det) | 0.910 | 0.718 | 0.827 | 0.719 |
| doctr-multilingual (fast_base det) | 0.870 | 0.628 | 0.829 | 0.727 |
| easyocr (da+en) | 0.868 | 0.710 | **0.789** | **0.683** |

(n=25 per field per config; lower is better for both columns)

Key findings:
- The Danish-capable doctr recognizer (`Felix92/doctr-torch-parseq-multilingual-v1`)
  is **worse than doctr's own default recognizer on every single metric**,
  not just less-calibrated as the visual inspection suggested -- its raw
  accuracy is worse too. Fixing the vocab problem didn't help in practice:
  this specific community checkpoint appears to be weaker overall (likely
  tuned more for printed/invoice-style multilingual text than handwriting).
  Swapping the detector (db_resnet50 -> fast_base) recovered some ground but
  never caught up to plain doctr, let alone easyocr.
- doctr's default (vocab-limited) recognizer is unexpectedly the **best on
  dates** by a wide margin -- consistent with dates rarely needing
  aeoeaa/AEOEAA, so the vocab gap doesn't bite there, and this recognizer is
  otherwise solid.
- On locality (where aeoeaa/AEOEAA actually matters), doctr-default and
  easyocr are close (0.768 vs 0.789 NED) in this noisy proxy metric --
  the vocab gap doesn't dominate as much as expected, likely because
  misclassified confound snippets (collector names etc., leaking into the
  locality bucket under this simplified per-word classifier) already add a
  lot of noise common to every engine, masking smaller vocab-driven errors.

## Fine-tune result (see docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md)

Fine-tuned doctr's `crnn_vgg16_bn` recognizer with `--vocab danish
--pretrained --freeze-backbone`, 20 epochs on 391 human-labeled word crops
(98 held out for validation), on Kaggle's free GPU tier. Scored with
`src/score_ocr_dev.py doctr-finetuned-danish 25` -- same 25-row sample
(seed=7) as the table above, so directly comparable.

| config | date NED | date AURC | locality NED | locality AURC |
|---|---|---|---|---|
| doctr (default, French vocab) | **0.615** | **0.315** | **0.768** | **0.704** |
| doctr-finetuned-danish | 0.984 | 0.938 | 0.882 | 0.843 |

The fine-tune made things dramatically worse, not better -- every single
date prediction came back "MISSING" (no date-shaped text survived
`classify_field` at all), and recognized text is visibly garbled noise
("Coll 15 Sdte", "Ms Coll 1 M Hansen") rather than near-miss transcriptions.
This isn't a small regression; the recognizer looks non-functional, not
just under-trained.

Likely cause: `--vocab danish` reinitializes the final classifier layer
(vocab length differs from the pretrained default's), and `--freeze-backbone`
only trains the RNN + that reinitialized head. 391 crops over 20 epochs is
240 gradient steps total for a head starting from scratch -- plausibly not
enough signal to escape a near-random initialization, rather than a subtler
calibration issue.

**Decision at this point: do not adopt v1.** Superseded below -- longer
training turned out to fix the collapse. Keep reading.

## Fine-tune retry v2: two-stage recipe, longer training

v1's 20-epoch, frozen-backbone-only run left the reinitialized vocab-danish
classifier head undertrained (240 gradient steps total). Retried as a
two-stage recipe on the same 391/98 train/val crop split:

- **Stage 1**: frozen backbone (only the reinitialized RNN + classifier
  head trains), 150 epoch ceiling, early-stop patience 15.
- **Stage 2**: resume from Stage 1, unfreeze the whole network
  (`--freeze-backbone` dropped), fine-tune everything at a 10x lower LR
  (0.0001 vs 0.001), 50 epoch ceiling, patience 10. The idea: let the conv
  backbone adapt slightly to this handwriting once the head has something
  real to build on, without wrecking its pretrained features by unfreezing
  from the start.

Both stages finished in **~1 minute total** on a Kaggle T4 -- suspicious
for "150/50 epoch ceilings," strongly suggesting early stopping tripped on
validation-loss noise (only ~4-7 batches/epoch with n=98 val crops) rather
than genuine convergence.

| config | date NED | date AURC | locality NED | locality AURC |
|---|---|---|---|---|
| doctr (default) | **0.615** | **0.315** | 0.768 | 0.704 |
| doctr-finetuned-danish (v1, 20ep) | 0.984 | 0.938 | 0.882 | 0.843 |
| doctr-finetuned-danish-stage1 (v2) | 0.761 | 0.532 | 0.740 | 0.684 |
| doctr-finetuned-danish-stage2 (v2) | 0.733 | 0.537 | **0.721** | **0.678** |

Huge fix vs v1 (recognizer is no longer broken -- real words come back,
not garbled noise), and locality now genuinely beats baseline. Date is
still worse than baseline, though: several dates that should transcribe
cleanly still come back "MISSING" entirely. Likely cause: `extract_reco_crops.py`
selects *images* by "does verbatimDate or verbatimLocality contain a
Danish letter," but dates essentially never contain aeoeaa/AEOEAA -- so
while date-snippet crops are still present in the training set (they're
extracted per-image, not per-field), the label set skews toward
locality-style vocabulary. The fine-tune may be trading a little digit/date
robustness for Danish-letter robustness.

## Fine-tune retry v3: pushed ceilings/patience much further

Suspecting v2 under-trained (1-minute total runtime despite 150/50 epoch
ceilings), retried with much higher ceilings and much more patient early
stopping, same architecture and two-stage structure, plus:
- Batch size 16 (was 32) -- more gradient updates/epoch on the same 391 crops.
- AdamW + weight decay (1e-4) in Stage 2's full-backbone unfreeze, for
  regularization (unfreezing the whole network on 391 crops is the riskiest
  step for overfitting).
- `--amp` (mixed precision) to keep wall-clock time reasonable at this scale.

| stage | epoch ceiling | patience |
|---|---|---|
| Stage 1 | 1500 | 120 |
| Stage 2 | 800 | 100 |

This run took **~5 minutes total** (vs v2's ~1 minute) -- meaningfully
longer, consistent with actually reaching a real early-stopping plateau
this time rather than stopping on noise.

| config | date NED | date AURC | locality NED | locality AURC |
|---|---|---|---|---|
| doctr (default) | 0.615 | 0.315 | 0.768 | 0.704 |
| doctr-finetuned-danish-stage1 (v3) | 0.727 | 0.600 | 0.741 | 0.700 |
| doctr-finetuned-danish-stage2 (v3) | **0.687** | **0.483** | 0.741 | 0.692 |

Trend across v1 -> v2 -> v3: date keeps improving steadily (0.984 -> 0.733
-> 0.687 NED), closing in on baseline but not surpassing it. Locality
plateaued and ticked slightly worse from v2 to v3 (0.721 -> 0.741) -- it
had already converged; more epochs on the same 391 crops isn't helping it
further, and may be trading a little of its capacity toward the date
improvement instead. **Conclusion: more training on this dataset size has
hit diminishing/mixed returns -- not worth pushing further without more
labeled data or fixing the date-crop-underrepresentation root cause.**

## Hybrid strategies tried (combining engines per field)

All combined-AURC figures below are the unweighted mean of the date and
locality AURC columns, since that's the shape of the competition's actual
scoring (AURC per field, averaged across fields).

**1. Blanket per-field swap** (not separately scripted -- computed directly
from the tables above): always doctr-default for `verbatimDate` (AURC
0.315, best in every comparison) + always doctr-finetuned-danish-stage2 for
`verbatimLocality` (AURC 0.692, beats baseline's 0.704). Combined mean AURC
~0.50 -- beats plain baseline (0.5095) and every single-engine option
tried, including easyocr.

**2. Danish-character fallback** (`src/score_hybrid_dev.py`, v1 of the
script): for `verbatimLocality`, use doctr-finetuned-danish-stage2's
prediction only if it contains a Danish letter (aeoeaa/AEOEAA); otherwise
fall back to doctr-default's prediction. Idea: only trust the fine-tuned
model on the specific cases it exists for.

Result: locality NED=0.744, AURC=0.690 (n=25) -- essentially identical to
just using stage2 outright (0.741/0.692). **Doesn't help.** Looking at the
per-row `source` labels, the rule doesn't actually separate good
predictions from bad ones: garbled noise from the fine-tuned model can
still contain a stray Danish letter by chance (e.g. `'rsbes ø 196187 cta
ant 1 J Suv 1 Drtus  e dal'` got selected purely because of the lone `ø`,
despite being pure noise, truth `'Jÿdske Aas Vendsÿssel'`, ned=0.778).
Character presence isn't a proxy for correctness.

**3. Confidence-based fallback** (`src/score_hybrid_dev.py`, v2 of the
script): for `verbatimLocality`, run both doctr-default and
doctr-finetuned-danish-stage2, keep whichever has the higher mean snippet
confidence.

Result: locality NED=0.768, AURC=0.704 (n=25) -- **exactly matches plain
baseline**, and every single row's source was "baseline." The rule
collapsed entirely: baseline doctr's confidences run consistently high
(0.75-0.97 in this sample) since it's a mature, well-calibrated general
recognizer, while the fine-tuned model's confidences run consistently
lower (0.3-0.7) even when its transcription is actually correct, since
it's a small model trained on 391 crops with a reshaped vocab. Raw
confidence isn't comparable across two differently-trained models without
calibrating each one onto the same scale first (exactly what Task 8 of the
main OCR pipeline plan, "confidence calibration," exists to eventually do
-- not yet built). **Don't compare raw cross-engine confidence without
calibrating first; this is not a quick tweak, it needs real calibration
work.**

## Decision

No currently-available Danish-capable doctr recognizer beats even doctr's
own vocab-limited default as a single engine for both fields, and neither
does easyocr. But the two-stage fine-tune (v3, `doctr-finetuned-danish-stage2`)
genuinely beats baseline on `verbatimLocality` specifically (AURC 0.692 vs
0.704). The best option found so far is the **blanket per-field hybrid**
(strategy 1 above): doctr-default for `verbatimDate`, doctr-finetuned-danish-stage2
for `verbatimLocality`, unconditionally -- no per-prediction fallback
logic, since both attempts at that (Danish-character presence, raw
confidence) either did nothing or actively regressed to one engine.

This is the recommended choice for the main OCR pipeline (Task 6/7 there)
unless/until: (a) more labeled crops are collected (391 is thin for a
from-scratch head + full-backbone unfreeze), (b) the date-crop
underrepresentation in the training set is fixed at the source (oversample
images whose *date* field specifically needs unusual characters/formats,
not just whichever field has a Danish letter), or (c) per-engine confidence
calibration (Task 8) is built, which could make a genuine confidence-based
fallback viable where the character/raw-confidence heuristics failed.
