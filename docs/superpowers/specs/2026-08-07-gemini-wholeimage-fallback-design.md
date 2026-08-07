# Gemini Whole-Image Fallback for Both-Missing Specimens — Design

**Goal:** the per-card Gemini pipeline (`docs/superpowers/plans/...` Gemini VLM
work, dev-set AURC 0.124) leaves 71/198 dev specimens (36%) with
`verbatimDate == verbatimLocality == "MISSING"` after the per-card merge. Add
a fallback pass that sends the **original, uncropped** specimen photo to
Gemini for exactly those specimens, since the per-card path never got to look
at large parts of the label (failed card detection, an over-eager confound
filter, or a crop the model itself couldn't read). Text extracted this way
will likely surface recurring museum/collection boilerplate
("DaSSCo", "Statens Naturhistoriske Museum") that the per-card prompt already
excludes but a whole-image view may re-trigger differently; cleaning that up
is a second, evidence-driven step after the first real run, not designed
blind here.

**Why "both fields MISSING" and not "crop detection failed":** investigated
the real breakdown before designing anything (`git` history / ad hoc query
against `data/processed/gemini/dev_card_predictions.csv` and
`data/processed/card_crops/manifest.csv`):

| cause | # specimens |
|---|---|
| zero card crops in the manifest at all | 5 |
| crops exist, all flagged confound by the visual relevance classifier (skipped, no Gemini call) | 46 |
| a relevant crop was sent to Gemini, which returned MISSING for both fields anyway | 25 (many of these are genuinely MISSING in `train.csv` ground truth) |

"Both fields MISSING after merge" is a single, cheap-to-compute criterion
that covers all three causes without having to branch on which one applies
per specimen — and for the 25-specimen group where the ground truth already
says MISSING, the whole-image call is expected to correctly say MISSING
again, at no cost beyond one extra API call.

## Architecture

```
run_gemini_dev.py
  Phase 1 (existing, unchanged): run_batch() over per-card crops
        |
        v
  merge_predictions_by_specimen() over dev_card_predictions.csv
        |
        v  select specimens where both fields == "MISSING"
        v
  Phase 2 (new): run_wholeimage_batch() over data/raw/images/<source_image_file>
        |
        v
  data/processed/gemini/dev_wholeimage_predictions.csv
```

`evaluate_gemini_dev.py` and `generate_submission_gemini.py` gain a fallback
merge step that overlays whole-image predictions onto the per-card merge,
field by field, only where the per-card side is MISSING.

## New/changed pieces

### `src/gemini_transcribe.py`

- `call_gemini_once` gains a `prompt=PROMPT` parameter (default unchanged) so
  the whole-image path can supply a different prompt through the same
  function instead of duplicating the API-call plumbing.
- New `WHOLE_IMAGE_PROMPT`: the existing `PROMPT`'s confound-exclusion rules
  verbatim (collector names, det./Tilg., "Dania", substrate notes, museum
  boilerplate, catalog numbers/initials — already correct, confirmed against
  `kaggle_meta/pages/data-description.md` again this session), plus:
  - explicit framing that this is the **full photo**, which may show more
    than one pinned specimen card, stray labels, and repeated
    institution/collection stamps;
  - instruction that if multiple cards each carry a date/locality, join them
    with `" | "` directly in the returned string (matching the competition's
    own pipe-join scoring rule, `kaggle_meta/pages/data-description.md`)
    rather than returning only one card's value;
  - same `"MISSING"`-per-field default and same strict-JSON response format,
    reusing `RESPONSE_SCHEMA` unchanged.
- `transcribe_card_with_confidence` is reused unchanged (its `call_fn`
  injection point already supports swapping in a whole-image call); no new
  self-consistency logic needed.

### `src/gemini_batch_runner.py`

- New `wholeimage_pending(image_files, merged_predictions)`: pure function,
  returns `[image_file, ...]` for specimens in `image_files` whose
  `merged_predictions[image_file]` has both fields `== "MISSING"`. Takes
  `merged_predictions` (the dict `merge_predictions_by_specimen` already
  produces) rather than recomputing anything, so it has no direct dependency
  on `calibrate_gemini_confidence` (avoids a circular import — that module
  already imports from `pipeline`, not from `gemini_batch_runner`).
- New `run_wholeimage_batch(image_files, merged_predictions, output_path,
  n_samples, rpm, limit, api_key, models, images_dir=WHOLEIMAGE_IMAGES_DIR)`:
  same resumable shape as `run_batch` (reuses `already_done`,
  `ModelRotation`, `_call_with_fallback`, `[setup]`/`[i/N]`/`[done]` progress
  prints, `f.flush()` after every row) but:
  - iterates whole `image_file`s instead of `(crop_filename, source_image_file)`
    pairs, reading from `data/raw/images/` instead of the card-crops dir;
  - calls `call_gemini_once(..., prompt=WHOLE_IMAGE_PROMPT)` (threaded through
    `_call_with_fallback`, which gains the same `prompt` passthrough
    parameter);
  - writes `OUTPUT_COLUMNS_WHOLEIMAGE = ["source_image_file",
    "verbatimDate", "verbatimDate_confidence", "verbatimLocality",
    "verbatimLocality_confidence", "models_used", "raw_samples_json"]` (no
    `crop_filename` column — there is no crop).
  - **Resumability** (explicitly requested): identical mechanism to the
    existing per-card batch — `already_done(output_path)` reads
    `source_image_file` values already written and skips them, each row is
    flushed to disk immediately after the API call returns, and the model
    rotation state naturally starts over from `models[0]` on a fresh process
    (matching existing behavior — quota resets are per-day, not tracked
    across restarts). Re-running the same command after an interruption
    (crash, quota exhaustion, Ctrl-C) always continues from the last
    unwritten row, never repeats a specimen already in the CSV, and never
    requires deleting/editing any file by hand.
- `WHOLEIMAGE_IMAGES_DIR = os.path.join("data", "raw", "images")`.

### `src/run_gemini_dev.py`

After Phase 1's existing `run_batch(...)` call:
1. Read back `data/processed/gemini/dev_card_predictions.csv` via
   `load_predictions_by_crop` + `merge_predictions_by_specimen` (both already
   in `calibrate_gemini_confidence.py`, imported here).
2. `wholeimage_pending(dev_image_files, merged)` to get the both-missing
   subset; print `[setup] N specimens still both-MISSING after per-card
   pass, running whole-image fallback...`.
3. `run_wholeimage_batch(...)` with the same `--n-samples`/`--rpm`/`--limit`/
   `--models` CLI flags already defined (no new flags needed — `--limit`
   naturally applies to whichever phase is currently running, since each
   phase checks its own `already_done` independently).

`OUTPUT_PATH_WHOLEIMAGE = os.path.join("data", "processed", "gemini",
"dev_wholeimage_predictions.csv")`.

`src/run_gemini_test.py` is **not** touched in this pass — mirrors this
repo's established gate ("only run the 3300-image test set once dev-set
numbers justify it"); the whole-image fallback gets the same treatment once
its dev-set impact is measured.

### `src/calibrate_gemini_confidence.py`

New `load_wholeimage_predictions(path)` (mirrors `load_predictions_by_crop`
but keyed by `source_image_file`, no crop-level dict nesting needed since
it's one row per specimen already) and `apply_wholeimage_fallback(merged,
wholeimage_predictions)`:

```python
def apply_wholeimage_fallback(merged, wholeimage_predictions):
    result = {}
    for image_file, fields in merged.items():
        fallback = wholeimage_predictions.get(image_file)
        updated = dict(fields)
        if fallback is not None:
            for field in FIELDS:
                if updated[field][0] == "MISSING" and fallback[field][0] != "MISSING":
                    updated[field] = fallback[field]
        result[image_file] = updated
    return result
```

Field-by-field, not specimen-by-specimen: even though a specimen only gets a
whole-image call when *both* fields were MISSING, the whole-image response
itself may find only one of the two (e.g. a legible date but no locality
anywhere on the photo) — no reason to discard the field it did find.
Never overwrites a per-card value that was already present.

### `src/evaluate_gemini_dev.py`

New fourth report block, after the existing three: "raw confidence + cleanup
+ whole-image fallback" — applies `apply_wholeimage_fallback` on top of the
existing `clean_merged_predictions` output (fallback runs after cleanup, so a
whole-image locality also gets `ö→ø`/trailing-punctuation cleanup by reusing
`clean_merged_predictions` a second time on the merged-with-fallback dict,
rather than duplicating the cleanup call). This is the number that decides
whether the fallback is worth keeping.

### `src/generate_submission_gemini.py`

`build_submission_df` gains the same fallback step, gated behind an optional
`wholeimage_predictions_by_specimen=None` parameter (default `None` = current
behavior unchanged, for the existing tests) so the function stays testable
without requiring a whole-image fixture in every test. Only wired to a real
value from `generate_submission_gemini()`'s `main()` path once
`evaluate_gemini_dev` confirms the fallback helps.

## What happens after the first real dev run

`src/analyze_gemini_errors.py` (existing diff-mining tool) gets pointed at
the whole-image predictions the same way it was used for the per-card merge
bug and the `ö→ø` pattern — run it, inspect the top recurring
prediction-vs-truth diffs, and only then decide whether a cleanup rule
belongs in `clean_gemini_predictions.py` (e.g. a boilerplate-phrase
blocklist for `verbatimLocality`, mirroring `classify_field.py`'s existing
`CONFOUND_KEYWORDS` list) or whether tightening `WHOLE_IMAGE_PROMPT` itself
is enough. Not designed speculatively here — same reasoning as this repo's
existing decision not to build gazetteer-matching without real error
evidence to bound its risk (`src/clean_gemini_predictions.py`'s docstring).

## Testing

- `tests/test_gemini_transcribe.py`: `call_gemini_once` accepts and forwards
  a custom `prompt` (fake client capturing the call args, same style as
  existing tests); `WHOLE_IMAGE_PROMPT` contains the same confound-keyword
  substrings as `PROMPT` (regression guard against silently dropping an
  exclusion rule when the two prompts diverge).
- `tests/test_gemini_batch_runner.py`: `wholeimage_pending` (both-missing
  included, one-field-present excluded, specimen absent from
  `merged_predictions` excluded); `run_wholeimage_batch` resumability
  (pre-seed the output CSV with one `source_image_file`, assert a second run
  skips it — same pattern the existing per-card resumability isn't
  explicitly tested for via `run_batch` directly, but `already_done` already
  has coverage; add the equivalent for the wholeimage output shape).
- `tests/test_calibrate_gemini_confidence.py`: `apply_wholeimage_fallback`
  (fills a MISSING field from fallback, leaves a present field untouched,
  leaves a specimen with no fallback entry untouched, partial fallback where
  only one of the two fallback fields is non-MISSING).
- `tests/test_generate_submission_gemini.py`: one new test with
  `wholeimage_predictions_by_specimen` supplied, confirming a MISSING field
  gets filled and cleaned.

## Error handling

- A missing/unreadable original image file (shouldn't happen — every
  specimen in `train.csv`/`test.csv` has a corresponding file in
  `data/raw/images/`, per this repo's existing data-download step) is not
  specially handled — same as the per-card path, which also doesn't guard
  against a missing crop file today; consistent with the repo's existing
  error-handling posture (surface loudly via the normal `FileNotFoundError`
  rather than adding a new try/except).
- Quota exhaustion / API errors during the whole-image phase: identical
  `ModelRotation` fallback-forward-through-models behavior as the per-card
  phase; once the last model is also exhausted, the exception propagates and
  the process exits — the next invocation resumes via `already_done`, per
  the resumability requirement above.

## Global constraints

- No new dependencies.
- `data/processed/gemini/dev_wholeimage_predictions.csv` is gitignored, same
  as the existing predictions CSVs.
- Ask before `git commit` (standing preference).
- `src/run_gemini_test.py` / the 3300-image test set are not touched until
  this is validated on dev, per the repo's existing gate.

## Self-review

**Placeholder scan:** no TBD/TODO. `WHOLEIMAGE_IMAGES_DIR` and all new
column/function names are concrete.

**Internal consistency:** `run_wholeimage_batch`'s output columns match what
`load_wholeimage_predictions` consumes; `apply_wholeimage_fallback`'s
per-field (not per-specimen) overlay is consistent with the possibility that
a whole-image call finds only one of the two fields; the fourth
`evaluate_gemini_dev` report block's order (cleanup, then fallback, then
cleanup again on the filled-in value) matches how `build_submission_df` would
need to apply the same steps for a real submission.

**Scope check:** one clear deliverable — a whole-image fallback pass wired
into `run_gemini_dev.py`, evaluated against the existing 0.124 baseline via a
new `evaluate_gemini_dev.py` report block. Cleanup-rule design and the
test-set run are explicitly deferred to after real dev-set results exist, not
built speculatively here.
