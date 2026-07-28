# Card/Label Text-Matching App Design

**Goal:** Manually match each detected card crop (from `data/processed/card_crops/`,
produced by the SAM detection pipeline in `src/detect_cards.py` /
`src/extract_card_crops.py`) against the specific piece(s) of ground-truth text
from `data/raw/train.csv` that are actually written on that card, in reading
order. This produces card-level, ordered text ground truth that a later step
(not part of this design) can use to build per-line training data for a
fine-tuned doctr recognizer -- the current gap is that `train.csv` gives
per-*specimen* text (potentially split across multiple physical cards via `|`),
not per-*card* text, and a specimen's cards need to be individually matched up
before doctr can be trained/evaluated per crop.

**Why this matters:** as established in `tables_figures/eda/sam_card_detection_test/notes.md`,
detecting the physical card first (rather than classifying loose OCR words)
was adopted specifically so that "every line inside one card's box belongs to
that card's value by construction." This app is the manual step that confirms
that assumption against real ground truth and builds the training pairs for it.

## Scope

In scope: a local web app for one person (the user) to work through every card
crop belonging to a `train.csv` row that has at least one non-`"MISSING"`
field, selecting which ground-truth text piece(s) belong to that card, in
order.

Out of scope (explicitly, to keep this from growing): drawing/adjusting
bounding boxes within a card (a future step re-runs doctr's own line detector
inside each card and aligns detected lines against this app's ordered text --
not needed here), OCR itself, and any automatic/ML-assisted matching (this is
a fully manual pass since the whole point is trustworthy ground truth for
later training).

## Data model

**Candidate pool (per specimen / `train.csv` row):**

```python
def build_candidates(row: dict) -> list[dict]:
    """row has verbatimDate/verbatimLocality (str). Returns an ordered list of
    {field: "date"|"locality", index: int, text: str} for every non-"MISSING"
    pipe-separated piece, dates first then localities, each piece stripped."""
```

Example, for `1742772.jpeg` (`verbatimDate="6/1862"`,
`verbatimLocality="Ti | Kulhuset Jægerspris"`):

```python
[
    {"field": "date", "index": 0, "text": "6/1862"},
    {"field": "locality", "index": 0, "text": "Ti"},
    {"field": "locality", "index": 1, "text": "Kulhuset Jægerspris"},
]
```

Rows where both fields are `"MISSING"` produce an empty candidate list and
their cards are excluded from the queue entirely (confirmed with the user --
there is nothing to match, and no value in a manual confirmation step for
them).

**Queue:** built from `data/processed/card_crops/manifest.csv`, filtered to
rows whose `source_image_file` is in `train.csv` and has a non-empty candidate
list, ordered by `source_image_file` then `card_index` (so a specimen's cards
are always shown consecutively, with all its candidates visible together for
context).

**Candidate reuse:** unrestricted (confirmed with the user) -- the same
candidate can be selected on more than one card of the same specimen, since
ground truth genuinely repeats sometimes (e.g. two pinned specimens sharing a
label: `"Rotholme Jyll. | Rotholme Jyll."`).

## Interaction

One card shown at a time:

- The card's cropped image (served from `data/processed/card_crops/images/`).
- Progress: `card {n} of {total in specimen}` for the current specimen, plus
  overall queue position.
- The specimen's full candidate list rendered as clickable chips.
- Clicking a candidate appends it to an ordered "selected" list rendered
  separately (numbered 1, 2, 3, ...); clicking an item in the selected list
  removes it (re-numbering the rest). No drag-to-reorder needed -- click order
  *is* the order; to fix a mistake, remove and re-click in the right sequence.
- **Guardar y siguiente**: commits the current selection and advances.
  Disabled (or alerts) if zero candidates are selected -- forces an explicit
  choice between selecting something and the button below.
- **Sin texto relevante**: commits an explicit empty match (confirmed with the
  user -- distinct from "not yet reviewed") for cards that are pure confound
  content (collector name, det./coll. boilerplate, museum stamps) within an
  otherwise-relevant specimen.
- **Saltar (mala detección)**: for crops that aren't a real card at all (SAM
  artifacts, e.g. the known corner-sliver false positive documented in
  `tables_figures/eda/sam_card_detection_test/notes.md`).
- On finishing the last card of a specimen: a non-blocking banner lists any of
  that specimen's candidates that were never selected on any of its cards
  (`"'Dyrehaven' no fue usado en ninguna tarjeta"`). This is a hint, not a
  hard stop -- it surfaces likely missed detections (a card SAM didn't find)
  or an accidental omission, without blocking progress if the user confirms
  it's expected (e.g. a candidate that legitimately isn't visible because its
  card wasn't captured well).

## Output format

`data/processed/card_crops/card_text_matches.json`:

```json
{
  "1742772_card00.jpeg": {
    "status": "matched",
    "selections": [{"field": "locality", "index": 0, "text": "Ti"}]
  },
  "1742772_card01.jpeg": {
    "status": "matched",
    "selections": [
      {"field": "date", "index": 0, "text": "6/1862"},
      {"field": "locality", "index": 1, "text": "Kulhuset Jægerspris"}
    ]
  },
  "1709018_card00.jpeg": {
    "status": "no_relevant_text",
    "selections": []
  },
  "1743909_card01.jpeg": {
    "status": "skipped",
    "selections": []
  }
}
```

Written atomically (write to `.tmp`, `os.replace`) after every save, same
pattern as `src/label_crops_app.py`. `status` is one of `matched` /
`no_relevant_text` / `skipped`.

## Resumability

On startup, load `card_text_matches.json` (if present) and skip any
`crop_filename` already present as a key -- identical resumability pattern to
every other long-running/manual script in this repo (`src/extract_card_crops.py`'s
`processed.log`, `src/label_crops_app.py`'s `labels.json`).

## Error handling

- Missing crop image file referenced by the manifest: the `/crop/<filename>`
  route 404s the same way `label_crops_app.py`'s does; shouldn't happen since
  the manifest was verified against disk after the Kaggle extraction, but
  fails loud rather than silently if it ever does.
- Directory-traversal on the crop filename: same containment check already
  used in `label_crops_app.py` (`os.path.dirname(os.path.abspath(...))` must
  equal the crops directory).
- A `train.csv` row whose `image_file` has zero rows in `manifest.csv` (no
  cards detected at all for that specimen): logged once at startup as a
  warning list, not fatal -- it just means that specimen contributes zero
  cards to the queue, which is itself useful information (a full detection
  miss worth revisiting later).

## Testing

- `tests/test_match_card_labels_app.py`: unit tests for the pure
  `build_candidates(row)` function (empty-fields case, single-value case,
  multi-pipe case, whitespace-stripping case) -- mirrors the existing
  `tests/test_label_crops_app.py` pattern.
- The app itself (HTTP server, browser interaction): manual verification by
  running it and matching a handful of real specimens, per this repo's
  established convention of pytest for pure logic + manual runs for
  UI/network-touching code (see `src/label_crops_app.py`,
  `src/extract_card_crops.py`).

## Self-review

**Placeholder scan:** no TBD/TODO left.

**Scope check:** single cohesive tool, one implementation plan.

**Consistency:** output path/format referenced consistently across Data
model, Output format, and Resumability sections; matches manifest.csv's own
`crop_filename` as the join key throughout.
