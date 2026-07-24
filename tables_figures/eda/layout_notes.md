# Label layout EDA (sample of 12 images + pixel-level probing on 3)

Sample reviewed: `layout_grid.png` (12 random images) plus pixel brightness
profiling on `1709496.jpeg`, `1743525.jpeg`, `1707688.jpeg` (full 8192x5464 res).

## Fixed vs. variable structure

Every image is an L-shaped white speckled foam mounting block against a uniform
grey-blue background, photographed with the same framing:

- **Vertical bar (fixed furniture, same across all images)**: pixel columns
  `x ≈ 0-2200` (measured 166-2172, +buffer), nearly full image height. Contains,
  top to bottom: NHM Denmark logo card, DaSSCo logo card, X-Rite color checker +
  1cm ruler, NHMD barcode/ID card. **Never a transcription target** — exclude
  this whole column outright.
- **Horizontal bar (variable content, same material as the vertical bar)**:
  extends from the vertical bar's right edge (`x ≈ 2200`) to near the right
  image edge (`x ≈ 7200-7300`), roughly the top quarter of the frame
  (`y ≈ 0-2100`, but this varies — one sample had a bright region extending to
  `y ≈ 2120` at `x=5000`, likely a card pinned unusually low). This is the
  surface the actual specimen collection-date/locality tag cards are pinned to.
  **This is the search region for Task 5's detector**, not a region to exclude.
- **Specimen area**: below/right of the horizontal bar, plain grey-blue
  background, holds a small white card with the pinned beetle. Not a text
  target (no transcription-relevant text on it in samples reviewed).

## Why a single brightness threshold (e.g. plain Otsu) is not enough

The speckled foam surface itself is *bright* (mean ~180-215 in a 0-255 grayscale
probe) — almost as bright as the plain white/cream tag cards pinned on it. A
brightness-only threshold (the naive approach sketched in the plan's Task 5
skeleton) would just recover the whole mounting block, not the individual tag
cards.

What actually distinguishes a tag card from the mount surface it sits on:
- **Local texture**: the mount surface has a fine black speckle pattern (high
  local variance in a small window); tag cards are smooth solid-color paper/
  plastic (low local variance) even though both are bright.
- **Color**: cards are white, cream, pale yellow, or (for "Dania coll."-style
  confound cards) orange/red — all distinguishable from the mount's neutral
  grey-white speckle by saturation, not just brightness.

**Revised detection strategy for Task 5**: within the search region (outside
the fixed furniture column), flag pixels that are both bright (>~160) *and*
low local variance (smooth, e.g. via a small-window std-dev or Laplacian
filter) as card candidates, then find connected components / contours on that
mask instead of on a plain brightness threshold. This replaces the plan's
original Otsu-only sketch.

## Card count and size

Matches `train.csv`'s pipe-count distribution: most images have 1 tag card,
some have 2-3 (occasionally more for multi-card specimens). Card sizes vary
widely — from small square tags (roughly the size of one color-checker swatch)
to larger rectangular note cards with multiple lines of handwriting. `MIN_CARD_AREA`/
`MAX_CARD_AREA` in Task 5 need a wide range to cover both.

## Other observations

- Lighting/exposure is not perfectly uniform across images — one sample
  (`1707688.jpeg`-style) was visibly washed out/overexposed. Brightness
  thresholds should be relative (e.g. adaptive/local, or normalized per-image)
  rather than hardcoded global pixel values, or Task 5 will need per-image
  fallback handling.
- Confound cards (collector, determination, "Dania coll.") are visually
  identical in material to date/locality cards — color/texture alone can't
  separate them from the transcription targets. That split is `classify_field`'s
  (Task 3) job, applied *after* OCR, not something the detector should try to
  solve.
