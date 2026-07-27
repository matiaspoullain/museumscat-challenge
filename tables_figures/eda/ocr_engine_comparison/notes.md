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

**Decision: do not adopt.** Reverting to the pre-fine-tune choice below.

## Decision

No currently-available Danish-capable doctr recognizer beats even doctr's
own vocab-limited default, let alone easyocr -- and the fine-tune attempt
(above) made doctr worse, not better. Sticking with doctr as a single
engine for both fields isn't supported by the data. Two ways forward, not
yet decided:

1. **Switch to easyocr** for both fields (simplest, matches the
   recommendation from the visual-only comparison too).
2. **Hybrid**: doctr-default for `verbatimDate` (clear numeric win) +
   easyocr for `verbatimLocality` (only engine that's both usably accurate
   *and* structurally capable of producing aeoeaa/AEOEAA) -- adds a second
   engine to maintain for a modest gain, worth it only if Task 7's fuller
   pipeline confirms the gap holds up.
