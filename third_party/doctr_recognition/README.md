# Vendored doctr recognition training script

Source: https://github.com/mindee/doctr/tree/v1.0.1/references/recognition
(`train.py`, `utils.py`), pinned to `v1.0.1` to match the installed
`python-doctr==1.0.1` in `environment.yml`. Apache License 2.0
(https://github.com/mindee/doctr/blob/main/LICENSE).

Unmodified from upstream — kept byte-identical so it can be diffed against
a future re-vendor. Not covered by this repo's Python style conventions.

Fine-tunes doctr's own pretrained `crnn_vgg16_bn` recognizer with a wider
vocab (`--vocab danish`, already `english + "æøåÆØÅ"` in
doctr's own `VOCABS`), freezing the backbone (`--freeze-backbone`) so only
the reshaped classification head is retrained. Dataset format: a folder
with `images/` (crop files) + `labels.json` (`{filename: text}`, no spaces
in any label). See
docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md for how
this repo builds that dataset and uses this script.
