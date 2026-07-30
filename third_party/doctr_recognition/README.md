# Vendored doctr recognition training script

Source: https://github.com/mindee/doctr/tree/v1.0.1/references/recognition
(`train.py`, `utils.py`), pinned to `v1.0.1` to match the installed
`python-doctr==1.0.1` in `environment.yml`. Apache License 2.0
(https://github.com/mindee/doctr/blob/main/LICENSE).

Unmodified from upstream — kept byte-identical so it can be diffed against
a future re-vendor. Not covered by this repo's Python style conventions.

Fine-tunes doctr's own pretrained `crnn_vgg16_bn` recognizer with a wider
vocab (`--vocab danish`, already `english + "æøåÆØÅ"` in doctr's own
`VOCABS`), freezing the backbone (`--freeze-backbone`) so only the reshaped
classification head is retrained in Stage 1, then unfreezing for a full
low-LR Stage 2. Dataset format: a folder with `images/` (crop files) +
`labels.json` (`{filename: text}`, no spaces in any label -- every label
here is a single word by construction).

Used by two fine-tune iterations, each building its own dataset and
invoking this script the same way: the current one,
docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md, and
the earlier, superseded one,
docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md (see
tables_figures/eda/ocr_engine_comparison/notes.md for why v1 was
superseded).
