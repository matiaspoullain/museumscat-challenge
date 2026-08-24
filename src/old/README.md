# `src/old` — code off the active path

Moved here on 2026-08-24. **Nothing in this folder is imported by the pipeline that
produces the best submission** (`src.generate_submission_gemini`, whole-image-primary
route, public AURC 0.02382). It is kept rather than deleted because:

- it documents how artifacts that *are* still in use were produced
  (`data/processed/card_crops/`, the Danish recognizer dataset), and
- several modules are the measurement of a closed dead end — and the measurement, not
  the implementation, is what stops it being re-proposed.

Tests stay in `tests/` and still run: they import `src.old.<module>`.

**Caveat for Kaggle runs:** the `museumscat-gemini-code` dataset is built from `src/*.py`
flattened, so `src/old/` is *not* shipped. A notebook needing one of these modules has to
copy it in explicitly.

## What is here, by group

### 1. Classical OCR pipeline (superseded, AURC 0.576)

Replaced by Gemini VLM transcription — see CLAUDE.md, Status 2026-08-07 onward.

| module | what it did |
|---|---|
| `benchmark_ocr.py` | compared OCR engines on real detected ROI crops |
| `score_ocr_dev.py` | scored an engine/recognizer choice on dev (NED + AURC) |
| `score_hybrid_dev.py` | same, for the hybrid doctr+easyocr strategy |
| `evaluate_card_reco.py` | evaluated the fine-tuned Danish recognizer |
| `evaluate_local.py` | end-to-end dev evaluation of the classical pipeline |
| `predict_cards.py` | flattened doctr output into word-level rows |

Note that `pipeline.py`, `classify_field.py`, `merge_records.py`,
`reconstruct_lines.py` and `generate_submission.py` are *also* classical-era code but
stayed in `src/` — the Gemini route still imports them. Age is not the criterion here;
reachability from the active entry points is.

### 2. Danish recognizer fine-tune tooling (no longer used)

| module | what it did |
|---|---|
| `extract_reco_crops.py` | extracted word crops from dev images |
| `build_word_crop_dataset.py` | assembled the word-level image+label training set |
| `split_reco_dataset.py`, `split_reco_dataset_v2.py` | train/val split |
| `package_reco_dataset.py`, `package_reco_dataset_v2.py` | packaged it for upload as a private Kaggle dataset |

### 3. EDA and visual inspection (one-off scripts)

`eda_card_blur_variance_test.py`, `eda_card_variance_test.py`, `eda_label_layout.py`,
`eda_sam_card_detection.py`, `inspect_detection.py`, `review_detections.py`,
`review_detections_check.py`.

These calibrated the constants in `src.detect_labels` / `src.detect_cards`, which are
still live.

### 4. Local manual-annotation web apps (their output is already captured)

`box_card_words_app.py`, `label_crops_app.py`, `match_card_labels_app.py`.

### 5. Closed confidence analyses and experiments

| module | outcome |
|---|---|
| `analyze_gemini_errors.py`, `analyze_confidence_errors.py` | error-pattern mining; fed `src.clean_gemini_predictions` and `src.confidence_penalty_rules` |
| `analyze_rank_headroom.py` | measured the oracle ranking floor (dev 0.0077) |
| `analyze_local_vlm_sample_count.py` | showed the local-VLM agreement signal is already present at `n_samples=1` |
| `experiment_confidence_grid.py`, `gemini_calibration_features.py` | grid over 1-D and multi-feature calibrators: **none beat raw confidence** |
| `consensus_text.py` | Track B (MBR selection, corpus snapping): **measured as harmful / not adopted** |
