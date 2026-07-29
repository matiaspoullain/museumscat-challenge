# Test: Segment Anything (SAM) for per-card detection

## Why

Three hand-crafted CV approaches (brightness+small-window texture, Canny+
rectangularity, blur-then-variance -- see
`../card_detection_variance_test/notes.md`) were all abandoned: the
card-vs-mount "smoothness" relationship flips sign depending on how much ink
a given card carries, so no fixed threshold at any scale separates card from
mount for both a nearly-blank card and a densely-handwritten one. User's
suggestion: try a pretrained zero-shot segmentation model instead of any
hand-tuned brightness/texture heuristic.

## What was tried

`src/eda_sam_card_detection.py` -- Meta's Segment Anything
(`segment-anything` PyPI package, `sam_vit_b_01ec64.pth` checkpoint, the
smallest official SAM checkpoint, ~375MB, chosen over the full ViT-H/ViT-L
checkpoints for CPU-feasible runtime since this machine has no GPU) runs its
`SamAutomaticMaskGenerator` over each sample's search-ROI crop (downsampled
to 1600px wide for speed), on the same 6 train images used throughout the
CV testing (2-3 real cards each, several with small confound tag cards).

Raw SAM output includes far more than individual cards: one huge mask for
the mount background, a long thin mask for the foam edge strip, and small
nested masks on individual words/letters within a card (SAM proposes masks
at every granularity, not just "whole object"). `select_card_masks` filters
these down to card-shaped candidates via, in order: area fraction of the
full ROI (drops the background, drops tiny letter fragments), aspect ratio
(drops the foam strip), solidity (`mask pixel area / bounding-box area`,
drops merged/spurious masks -- see below), then containment-based
non-max-suppression (drops masks mostly nested inside an already-kept
larger one, e.g. a letter mask inside its card's mask).

## Result: zero-shot segmentation works, first pass had 2 real bugs

Initial run (`points_per_side=16`, no solidity filter): 4/6 images got a
clean, single box per real card, including small confound tags ("NS", "Kb",
"Mus. Løv") that every CV approach failed to isolate. But 2 bugs surfaced:

**Bug 1 -- solidity.** In `1706347.jpeg`, SAM proposed a spurious mask
spanning both the "Ti" and "Rørvig" cards *and* the mount gap between them
(bbox area 70384px, but only 34594px actually in the mask -- **solidity
0.49**). This bbox is larger than the two real card masks individually
("Rørvig" alone: solidity 0.98; "Ti" alone: solidity 0.91), so
containment-based NMS kept the spurious merged box and dropped both correct
ones. Fixed with a `MIN_SOLIDITY = 0.7` filter -- real cards measured
0.83-0.98 solidity across all samples; the two spurious merged-region masks
found in testing measured 0.46 and 0.49. After the fix: `1706347.jpeg` and
`1743909.jpeg` (which had the same failure mode) both went from missing a
card to a clean box per card.

**Bug 2 -- recall, not filtering.** In `1706979.jpeg`, SAM's raw output
never proposed *any* mask over the "Mus. Løv" card at all (only 4 raw masks
total: background, 2 real cards, one tiny noise speck) -- not a filtering
problem, a generation problem: with the default `points_per_side=16` query
grid, no sample point landed on that small card. Retested with
`points_per_side=32` (4x denser grid): found it correctly, and re-verified
all 6 samples still segment cleanly at the higher setting (see
`filtered_pps32_1706979.jpeg`).

## Runtime tradeoff (open decision)

| points_per_side | runtime/image (CPU, 1600px-wide ROI) | recall on 6-sample set |
|---|---|---|
| 16 | ~43-72s | 5/6 images perfect, 1 missed card (small tag) |
| 32 | ~166s (~4x) | 6/6 images perfect |

Extrapolated to the full dataset (3500 images, serial CPU): ~16 (~1.5 days)
vs. 32 (~6.5 days). This is a one-time offline pass (not repeated per
training iteration), but the difference is large enough to be a real
decision, not a rounding error. Options not yet evaluated: run only on
images likely to have multiple/small cards (e.g. `train.csv`'s pipe-count
distribution suggests most images have exactly 1 card and may not need the
denser grid), parallelize across CPU cores, or move to a Kaggle Notebook
GPU run (same pattern already used for the doctr recognizer fine-tune, see
`docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md`).

## Final validation (points_per_side=32, all 6 samples)

Re-ran all 6 samples at `points_per_side=32` with both fixes in place
(solidity filter + the denser grid). **5/6 images: one clean box per real
card, zero false positives.** `1709431.jpeg` (3 real cards) got a 4th, small
spurious box in the bottom-left corner (area frac 0.0027, solidity 0.90 --
passes every current filter on its own geometry) that doesn't correspond to
any card; likely a rounded-corner/edge artifact of the ROI crop itself. Not
special-cased: it's small, sits where there's no text, and Task 6's OCR
will simply find nothing inside it (a harmless empty region downstream), so
adding a corner-specific rule on a sample of 1 would be overfitting to this
one image rather than a real fix -- revisit only if the full 200/3500-image
run shows this is systematic rather than a one-off.

Final per-image results, `points_per_side=32`:

| image | real cards | boxes found | false positives |
|---|---|---|---|
| 1709431.jpeg | 3 | 3 + 1 spurious corner sliver | 1 (harmless, see above) |
| 1705903.jpeg | 2 | 2 | 0 |
| 1706979.jpeg | 3 | 3 | 0 |
| 1705485.jpeg | 3 | 3 | 0 |
| 1706347.jpeg | 3 | 3 | 0 |
| 1743909.jpeg | 2 | 2 | 0 |

## Decision

Adopt SAM (ViT-B) + `select_card_masks` (area/aspect/solidity/containment
filter, `points_per_side=32`) as the per-card detection step, replacing the
abandoned hand-crafted CV approaches. This directly resolves the original
multi-line-locality-on-one-card grouping question: OCR runs per detected
card crop, and every line inside one card's box belongs to that card's
value by construction, no separate line-clustering heuristic needed.

Still open, pending user input: how to run this over all 3500 images
within a reasonable time budget (~6.5 days serial CPU at pps=32) -- options
include restricting the denser grid to images likely to need it, running on
Kaggle's free GPU notebook tier (same pattern as the doctr recognizer
fine-tune), or parallelizing across CPU cores.
