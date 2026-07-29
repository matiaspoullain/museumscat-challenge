# Card Word-Box Labeling, Danish Recognizer Fine-Tune, and Card Relevance Classifier — Design

**Goal:** Use the now-complete card/label matching (`data/processed/card_crops/card_text_matches.json`,
produced by `src/match_card_labels_app.py`) to (1) build a real, manually-boxed
word-level training set for a Danish-vocab doctr recognizer fine-tune, (2) run
that recognizer (or whichever engine wins) over every detected card crop
(~6854, not just the 200-specimen dev set), and (3) build a keyword-based
classifier that tells apart cards actually carrying `verbatimDate`/
`verbatimLocality` text from confound cards (collector, determination,
museum-boilerplate), using the ground truth the matching step already
produced. Training and full-dataset inference run on Kaggle's free GPU tier;
everything else (labeling app, dataset generation, evaluation, classifier) is
local CPU work with short smoke tests before any Kaggle run.

**Why not reuse the earlier word-crop fine-tune:** that work (391 crops,
hand-transcribed from images pre-filtered to ones whose ground truth contains
a Danish letter) lives unmerged in `worktree-danish-recognizer-finetune` and is
being superseded, not extended — see `tables_figures/eda/ocr_engine_comparison/notes.md`
in that worktree for its full history. Its dataset was biased toward
Danish-letter text, which measurably hurt `verbatimDate` accuracy (dates
rarely need æøå) while only marginally helping `verbatimLocality`. The
card-matching ground truth used here isn't biased that way (it's every
matched card, date and locality alike) and is more plentiful (424 words across
162 cards vs. 391 words total before). The old worktree is left untouched; it
is not deleted or merged.

## Architecture

```
card_text_matches.json (162 matched cards, ordered text per card)
        |
        v  (Phase 1) box_card_words_app.py -- human draws one box per word
word_boxes.json
        |
        v  build_word_crop_dataset.py -- crops each word (union of boxes),
        |  writes images/ + labels.json in doctr's training format
        v
third_party/doctr_recognition/train.py (vendored, two-stage recipe)
        |
        v  (Kaggle, GPU)
danish_crnn_vgg16_bn.pt  ---------------------------+
        |                                            |
        v  (local) benchmark_ocr.py engine +         |
        |  evaluate vs baseline on val split          |
        v                                            v
   winning engine  --------------------->  (Phase 2, same Kaggle notebook)
                                           run winning engine over all
                                           ~6854 card_crops -> predictions.csv
                                                     |
                                                     v
                                    (Phase 3, local) classify_card_relevance.py
                                    reuses classify_field's keyword rules,
                                    evaluated against the 323 labeled cards
                                    (162 matched + 161 no_relevant_text),
                                    then applied to all ~6854 predictions
```

## Phase 1: Word-box labeling app + fine-tune

### Labeling app: `src/box_card_words_app.py`

Same single-file `stdlib http.server` pattern as `match_card_labels_app.py`
and the (unmerged, not reused) `label_crops_app.py` — a new port (8767).

**Queue:** built from `card_text_matches.json`, filtered to `status ==
"matched"`. For each matched card, each `selections` entry (already in the
app's saved reading order) is split on whitespace into words. The queue is a
flat, ordered list of `{crop_filename, field, index, word_index, text}` — one
entry per word, 424 total across 162 cards. Cards are still shown grouped
(one image loaded per card, all its words asked in sequence) so the user
isn't re-loading the same image per word.

**Interaction:** the card image is shown at a size that lets the user draw a
rectangle by click-drag (canvas overlay on an `<img>`, converting displayed
coordinates back to the crop's own full-resolution pixel coordinates). The
current word to box is displayed (`"palabra 2/3 de esta tarjeta: 'Jægerspris'"`).

- **Draw box, then "Siguiente palabra"**: commits the box for the current
  word and advances.
- **"Agregar otro cuadro para esta palabra"**: for a word visibly split
  across two lines — lets the user draw a second (or third) box for the same
  word before advancing. Multi-box words are stored as-is; Phase 1's dataset
  builder decides how to use them (see below), not the app.
- **"No la encuentro / ilegible"**: marks the word as skipped (excluded from
  training).
- Back/forward navigation and resumability, same pattern as the other two
  apps (skip words already present in the output file).

**Output:** `data/processed/card_crops/word_boxes.json`:

```json
{
  "1742772_card01.jpeg": [
    {"field": "date", "index": 0, "word_index": 0, "text": "6/1862",
     "boxes": [[120, 40, 210, 78]]},
    {"field": "locality", "index": 1, "word_index": 0, "text": "Kulhuset",
     "boxes": [[30, 90, 140, 125]]},
    {"field": "locality", "index": 1, "word_index": 1, "text": "Jægerspris",
     "boxes": [[30, 130, 175, 168]]}
  ]
}
```
Skipped words are recorded with `"boxes": []` so resumability can tell
"skipped" from "not yet reached" apart.

Written atomically (`.tmp` + `os.replace`), identical technique to the other
two apps.

### Dataset generation: `src/build_word_crop_dataset.py`

**Consumes:** `word_boxes.json`, `data/processed/card_crops/images/`.
**Produces:** `data/processed/reco_finetune_v2/dataset/{images/,labels.json}`
(new directory, `_v2` suffix distinguishes it from the abandoned worktree's
`data/processed/reco_finetune/` if that ever gets rehydrated locally).

For each word entry with exactly one box: crop it directly (plus a small
fixed pixel margin, matching how doctr's own detector crops in
`benchmark_ocr.py`) and pair it with `text` in `labels.json`. For a word with
**more than one box**: take the union (min/max) of all its boxes as the crop
region — simplest option, no special-cased multi-region stitching. Report a
count of multi-box words at the end of the run (`print(...)`, not a file) so
it's visible whether this is a handful of cards or enough to warrant a
smarter approach later; the union-crop approach is not revisited pre-emptively
since the user said this case is rare.

Words marked skipped (`boxes: []`) are excluded. Function surface for
testing: `union_box(boxes: list[tuple[int,int,int,int]]) -> tuple[int,int,int,int]`
(pure, tested) and `build_dataset() -> int` (crop count, manual verification
like the other extraction scripts in this repo).

### Vendored training script

`third_party/doctr_recognition/{train.py,utils.py,README.md}` — re-vendored
at `v1.0.1` (matches installed `python-doctr==1.0.1`), same as the abandoned
plan's Task 4, since it doesn't exist on `main`. Byte-identical to upstream,
not covered by this repo's style conventions (see the old plan's README text
for the exact provenance note to reuse).

### Split + package + Kaggle

`src/split_reco_dataset_v2.py` (train/val split, same deterministic-seed
pattern as the abandoned plan) and `src/package_reco_dataset_v2.py` (zip +
`dataset-metadata.json` for a new private Kaggle Dataset,
`museumscat-danish-recognizer-word-boxes` or similar — ask the user to
confirm the exact slug at upload time, not decided here). Upload is
user-triggered, same as every other Kaggle-account-touching step in this
repo.

### Kaggle notebook: `notebooks/kaggle_finetune_and_predict_danish.py`

Cell-marked script (paste into a Kaggle Notebook), combining fine-tune +
Phase 2's full-dataset prediction in one GPU session:

1. Install `python-doctr[torch]==1.0.1`, fetch vendored `train.py`/`utils.py`.
2. Attach two inputs: the word-box dataset (this phase) and the **already-uploaded**
   card-crops + manifest dataset (no re-upload needed — confirmed already on
   Kaggle).
3. Fine-tune: reuse the two-stage recipe that converged in the abandoned
   work's v3 (frozen-head warm-up, then unfrozen low-LR continuation with
   AdamW + weight decay, high epoch ceilings + patience, `--amp`, batch 16) —
   proven to converge in ~5 minutes on this recipe shape; exact epoch/patience
   numbers re-tuned once the real word-count (bigger than 391, smaller than
   thousands) is known.
4. Save checkpoint to `/kaggle/working/`, plus an auto-save to a permanent
   Kaggle Dataset at run end (the abandoned work's last commit added exactly
   this safety net — reuse the idea: fine-tunes on Kaggle's ephemeral free
   tier are worth checkpointing defensively).
5. Run the winning engine (decided locally in the next step, or defaulted to
   "the fine-tune, always" pending that comparison) over every row in the
   card-crops manifest, writing `predictions.csv`
   (`crop_filename,predicted_text,confidence`) to `/kaggle/working/`.
6. User downloads both the checkpoint and `predictions.csv`.

### Local evaluation: modify `src/benchmark_ocr.py`, new `src/evaluate_card_reco.py`

Add `ENGINES["doctr-finetuned-danish"]` (same shape as the abandoned plan's
Task 7 addition — load `crnn_vgg16_bn` with `VOCABS["danish"]`, load the
downloaded state dict). `evaluate_card_reco.py` scores the fine-tuned engine
against baseline `doctr` **directly on the val split's real word-level ground
truth** (no ROI/classify_field approximation needed this time, since the
labels are exact words) — reports per-engine word accuracy / mean NED. This
result decides which engine (or whether a hybrid) runs in the notebook's step
5; if the Kaggle run already happened with a default choice, this can also be
a post-hoc sanity check rather than a gate.

## Phase 2: Full-dataset prediction

Covered by notebook step 5 above — the only "new" local piece is inspecting
the downloaded `predictions.csv` (row count should match `manifest.csv`'s
~6854 rows) and moving it to `data/processed/card_crops/predictions.csv`.

## Phase 3: Card relevance classifier — `src/classify_card_relevance.py`

**Consumes:** `src.classify_field.classify_field` (existing, unmodified
unless evaluation surfaces gaps in `CONFOUND_KEYWORDS`), `predictions.csv`
(Phase 2), `card_text_matches.json` (ground truth: `matched` → relevant,
`no_relevant_text` → not relevant, `skipped` excluded — 323 labeled cards
total).

**Produces:** `is_relevant_card(words: list[str]) -> bool` — `True` if any
word's `classify_field(word) != "confound"`. Pure, tested with hand-written
word lists covering: a card with a date and no confound words (relevant), a
card with only `"Coll. J. P. Johansen"`-style confound words (not relevant),
an empty word list (not relevant — a card with zero recognized text can't be
carrying a real value).

**Evaluation:** run `is_relevant_card` over each of the 323 labeled cards'
`predictions.csv` words, compare to ground truth, print precision/recall/
accuracy. If misclassifications cluster around a specific confound pattern
not yet in `CONFOUND_KEYWORDS` (e.g. catalog numbers, a museum stamp phrase
not seen in Task 3's original test cases), add it and re-run — same
iterate-on-real-errors loop the original classifier was built with.

**Application:** once precision/recall look reasonable, apply
`is_relevant_card` to the full `predictions.csv` (~6854 rows), writing a
`relevant` boolean column alongside it. Wiring this into the actual
submission merge step (Task 4/7 of the main OCR pipeline plan) is a natural
next step but explicitly **out of scope** here — this phase's deliverable is
the classifier and its evaluated accuracy, not a new submission.

## Testing

- `tests/test_box_card_words_app.py`: pure queue-building and resumability
  logic (build the flat word queue from `card_text_matches.json`, skip words
  already in `word_boxes.json`), same split as the other two apps (pytest for
  pure logic, manual run for the HTTP/canvas interaction).
- `tests/test_build_word_crop_dataset.py`: `union_box` (single box passthrough,
  two-box union, three-box union).
- `tests/test_classify_card_relevance.py`: `is_relevant_card` cases above.
- `tests/test_split_reco_dataset_v2.py`: deterministic split, same shape as
  the abandoned plan's test.
- Kaggle notebook, the labeling app's browser interaction, and the SAM/doctr
  CV code: manual verification by running and inspecting output, per this
  repo's established split (see `src/extract_card_crops.py`,
  `src/match_card_labels_app.py`).

## Error handling

- Directory-traversal / 404 handling for served crop images: identical
  containment-check pattern already used in `match_card_labels_app.py`.
- A word with zero boxes drawn (neither boxed nor explicitly skipped) if the
  user closes the browser mid-word: treated as "not yet reached" on restart,
  not silently dropped — matches the resumability contract of the other apps.
- Kaggle notebook cells fail loud (assert checkpoint/predictions file exists
  before declaring success) rather than silently producing a truncated
  `predictions.csv`.

## Global constraints

- No new dependencies — `python-doctr[torch]`, `easyocr`, `segment-anything`,
  `tqdm` are already in `environment.yml` from earlier work; the labeling app
  is stdlib-only (`http.server`, `json`, `os`), matching the other two apps.
- `data/processed/`, `models/` stay gitignored; nothing under them is
  committed.
- CC BY-NC 4.0 — the new Kaggle Dataset (word-box crops) must be **private**,
  same requirement as the existing card-crops dataset.
- Ask before `git commit` (standing preference) and before any
  `kaggle datasets create` / notebook run (touches the user's Kaggle account
  and GPU quota) — every such step ends with a proposed command, not an
  automatic run.
- The abandoned `worktree-danish-recognizer-finetune` worktree/branch is left
  as-is — not merged, not deleted, not read from by any script in this plan.

## Self-review

**Placeholder scan:** no TBD/TODO. The Kaggle dataset slug and exact
epoch/patience numbers are explicitly flagged as decided at implementation
time (from real dataset size), not left vague — everything else is concrete.

**Internal consistency:** `word_boxes.json`'s `{field, index, word_index,
text, boxes}` shape matches what `build_word_crop_dataset.py` consumes;
`labels.json`'s output format matches what the vendored `train.py` expects
(no spaces per label, since every label is a single word by construction).
`predictions.csv`'s columns are referenced identically in Phase 2 and Phase
3. The relevance classifier's ground truth (`matched`/`no_relevant_text`)
matches `card_text_matches.json`'s actual `status` values confirmed earlier
against the real file (162/161/17 split).

**Scope check:** three phases, each with one clear deliverable (labeled
dataset + fine-tuned checkpoint; full-dataset predictions; evaluated
relevance classifier). Wiring the relevance classifier into the actual
submission pipeline is explicitly deferred, keeping this spec's scope to
what was asked.
