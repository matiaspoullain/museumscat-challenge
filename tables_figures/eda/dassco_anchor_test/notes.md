# Test: DaSSCo-card color anchor for the furniture-column cut

## Why

`find_search_roi` (Task 5) originally cut the furniture column at a hardcoded
`FURNITURE_X = 2200`. This is fragile: the vertical furniture bar and the
card-bearing horizontal bar are one physically connected bright blob, so if
the cut falls even ~30-100px short of the true corner, the connected
component reconnects to the vertical bar (which runs nearly the full image
height, down to the barcode near the bottom) and the detected ROI balloons to
cover almost the entire image.

Tried and rejected first: `cv2.matchTemplate` on the "STATENS NATURHISTORISKE
MUSEUM" logo card. Match confidence was inconsistent (0.76-1.00) and, because
it's a fixed-size template, it measures *position* only, not true size — no
way to tell a scale change from a position shift.

## What was tried (this test)

Detect the DaSSCo card instead, by color: it's a solid, highly saturated blue
against a much less saturated blue-grey background and white/speckled mount,
so a plain HSV threshold gives its true bounding box directly (real detected
size, not an assumed template size).

```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (95, 95, 40), (130, 255, 255))  # hue 95-130, sat > 95
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
# largest connected component's bbox = DaSSCo card
```

(First attempt used `sat > 60`, which let the background — sat ~59 — bleed
into the mask on 14/20 images. Raising to `sat > 95` fixed it: 20/20 clean,
tight detections, area varying only ~2.04M-2.16M px².)

Calibrated `furniture_x = dassco_x + 1.21 * dassco_width` from the ratio on
well-behaved images, then validated on all 20.

## Result

All 20/20 images: resulting horizontal-bar bottom edge lands at 38.5%-40.0%
of image height — tight and consistent, including the 2 images
(`1705468`, `1706175`) that previously broke the fixed-x2200 approach.

Bonus finding: those 2 images aren't a different camera scale (as an earlier,
now-abandoned logo-template-based theory suggested) — the DaSSCo card occupies
the *same* % of the image in all 20 (~18.5% width, ~27.4% height). They're
just shifted right in-frame by ~350-450px. The color-based, true-size
measurement is what revealed this; the template-matching experiment couldn't
distinguish "smaller" from "shifted" and misled toward a wrong theory.

Images here: green box = detected DaSSCo card, blue line = derived
`furniture_x` cut, red box = resulting search ROI. `1709018`/`1706758` are
normal cases; `1705468`/`1706175` are the two previously-broken ones, now
fixed.

## Decision

Adopt this as `detect_labels.py`'s furniture-column detection, replacing the
hardcoded `FURNITURE_X`.

## Follow-up: decoupling the safety cut from the reported ROI edge

User caught a real issue after the above was adopted: `furniture_x = dassco_x
+ 1.21 * dassco_width` sits ~300px to the *right* of the DaSSCo card's own
right edge (that's the point of the 1.21 ratio — reaching the true corner
safely) — but that means the reported ROI's left edge also started 300px
right of the card, clipping any tag card pinned close to the corner. Tested
`ratio=1.0` (cut exactly at the card's edge) to fix that: broke all 20/20
images (100% reconnected to the furniture bar's near-full-height extent) —
that 300px gap is a real, physical margin the connected-component search
needs to cross safely, not slack that can be removed.

Fix: use two different values from the same anchor. `safe_cut` (the 1.21
ratio) is used *only* to mask the image before running connected components,
so the search never re-touches the furniture bar. The reported ROI's `x1` is
set independently, directly to the DaSSCo card's own right edge (`dassco_x +
dassco_width`, minus padding) — the ~300px strip in between is blank mount
material either way, so including it in the reported ROI is free (Task 6's
OCR won't find text there) while excluding it from the connected-component
mask is what keeps detection robust. Re-validated on the same 20 images:
`y2` unchanged (2133-2217, same as before), `x1` now starts ~300px earlier
(~1800-1870 vs ~2150-2220), snug against the DaSSCo card on visual review.
