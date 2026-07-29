# Danish-Vocab doctr Recognizer Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune doctr's default `crnn_vgg16_bn` recognizer so its output vocabulary includes æøåÆØÅ, since `tables_figures/eda/ocr_engine_comparison/notes.md` showed the default recognizer beats every other tested engine/checkpoint except for literally being unable to emit those characters — first validated locally as a CPU smoke test, then run for real on Kaggle Notebooks' free GPU.

**Architecture:** doctr's own `references/recognition/train.py` (vendored verbatim, pinned to the installed `python-doctr==1.0.1` tag) already supports exactly this recipe: build the model with `vocab="danish"` (built into doctr's `VOCABS`, already `english + "æøåÆØÅ"`) while loading the `--pretrained` backbone weights, and `--freeze-backbone` so only the reshaped final classification head trains. The new work is (1) building a small labeled word-crop dataset since we only have field-level ground truth, via a from-scratch human-labeling web app, and (2) wiring that dataset + the vendored script into a local CPU smoke test and a Kaggle GPU run.

**Tech Stack:** Python 3.11, `python-doctr[torch]` (already installed), `torch`/`torchvision` (already installed as doctr deps), stdlib `http.server` for the labeling app (no new dependency), Kaggle Notebooks free GPU tier (T4x2 or P100, ~30h/week, requires phone-verified Kaggle account).

## Global Constraints

- All Python: snake_case names, `os`/`os.path` for paths, spaces around `=` in assignments, every function gets a one/few-line docstring — **except** the two vendored files in `third_party/doctr_recognition/`, which stay byte-identical to upstream (their own style, not ours) so they can be diffed/re-vendored cleanly later.
- Run everything with `conda run -n museumscat-challenge python ...`, never assume an activated shell.
- `data/processed/`, `models/` are gitignored — every artifact this plan produces under those paths (crops, labels, checkpoints) never gets committed.
- No new pip/conda packages: the labeling app uses only stdlib (`http.server`, `json`, `os`), and Kaggle installs `python-doctr[torch]` inside its own managed notebook environment, not locally.
- CC BY-NC 4.0 data license (non-commercial) applies to the museum images and anything derived from them (the word crops) — the Kaggle Dataset created in Task 6 **must be private**, never public.
- Kaggle's 30h/week GPU quota and the Kaggle account itself are the user's own — per this repo's existing policy for `kaggle competitions submit`, no script in this plan calls `kaggle datasets create`/`kaggle kernels push` or otherwise touches the user's Kaggle account automatically. Task 6 produces ready-to-run artifacts; the user (or Claude, asked explicitly in the moment) triggers the actual upload/run.
- Ask before `git commit` (standing user preference) — each task ends with staging + a proposed commit message, not an automatic commit.
- This codebase's existing convention (see `src/benchmark_ocr.py`, `src/detect_labels.py`, `src/score_ocr_dev.py`) is: pure logic gets `tests/` pytest coverage, image/CV/network-touching code gets a manual "run it, inspect the output" verification step instead of mocked unit tests. This plan follows that split rather than force-fitting pytest around cv2/doctr calls.

---

## File Structure

- `src/extract_reco_crops.py` — finds the dev-set rows whose ground-truth text contains æøåÆØÅ, runs the existing word detector on those images, saves every detected word crop as its own file.
- `tests/test_extract_reco_crops.py` — tests the pure row-filtering logic.
- `src/label_crops_app.py` — a single-file local web app (stdlib `http.server` + inline HTML/JS) for transcribing the crops from Task 1, with buttons for æøåÆØÅ.
- `tests/test_label_crops_app.py` — tests the pure "what's the next unlabeled crop" logic.
- `third_party/doctr_recognition/train.py`, `utils.py`, `README.md` — doctr's own recognition fine-tuning script, vendored verbatim at the `v1.0.1` tag (matches the installed package).
- `src/split_reco_dataset.py` — splits the human-labeled dataset into train/val folders in the layout `third_party/doctr_recognition/train.py` expects.
- `tests/test_split_reco_dataset.py` — tests the pure split logic.
- `src/package_reco_dataset.py` — zips the train/val folders into a Kaggle-Dataset-ready directory with `dataset-metadata.json`.
- `notebooks/kaggle_finetune_danish_recognizer.py` — the actual fine-tune run, written as cell-marked (`# %%`) code to paste into a Kaggle Notebook.
- `src/benchmark_ocr.py` (modify) — add a `doctr-finetuned-danish` engine that loads the checkpoint produced by the Kaggle run.

---

### Task 1: Extract Danish-candidate word crops

**Files:**
- Create: `src/extract_reco_crops.py`
- Test: `tests/test_extract_reco_crops.py`

**Interfaces:**
- Consumes: `src.benchmark_ocr.ENGINES["doctr"]` (`crop_bgr -> list[(text, confidence, (x1,y1,x2,y2))]`), `src.detect_labels.get_search_crop(image_path) -> crop_bgr` (both already exist).
- Produces: `rows_with_danish_letters(df: pandas.DataFrame) -> pandas.DataFrame` and `extract_crops() -> int` (crop count), consumed by Task 2's labeling app via the files it writes: `data/processed/reco_finetune/dataset/images/*.png` and `data/processed/reco_finetune/dataset/candidates.json` (`{crop_filename: source_image_file}`).

- [ ] **Step 1: Write the failing test for the pure filter**

```python
# tests/test_extract_reco_crops.py
import pandas as pd

from src.extract_reco_crops import rows_with_danish_letters


def test_keeps_rows_with_danish_letters_in_either_field():
    df = pd.DataFrame({
        "image_file": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "verbatimDate": ["12.06.1932", "MISSING", "3 år siden", "1901"],
        "verbatimLocality": ["København", "Århus", "Roskilde", "Odense"],
    })
    kept = rows_with_danish_letters(df)
    assert set(kept["image_file"]) == {"a.jpg", "b.jpg", "c.jpg"}


def test_drops_rows_with_no_danish_letters():
    df = pd.DataFrame({
        "image_file": ["e.jpg"],
        "verbatimDate": ["1945"],
        "verbatimLocality": ["Roskilde"],
    })
    kept = rows_with_danish_letters(df)
    assert kept.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n museumscat-challenge pytest tests/test_extract_reco_crops.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError: cannot import name 'rows_with_danish_letters'`

- [ ] **Step 3: Write the implementation**

```python
# src/extract_reco_crops.py
"""Extract word-level crops from dev images whose ground-truth text contains
a Danish letter, for manual transcription and doctr recognizer fine-tuning.
See docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md.

Usage:
    conda run -n museumscat-challenge python -m src.extract_reco_crops
"""
import json
import os
import re

import cv2
import pandas as pd

from src.benchmark_ocr import ENGINES
from src.detect_labels import get_search_crop

TRAIN_CSV = os.path.join("data", "raw", "train.csv")
IMAGES_DIR = os.path.join("data", "raw", "images")
DATASET_DIR = os.path.join("data", "processed", "reco_finetune", "dataset")
IMAGES_OUT_DIR = os.path.join(DATASET_DIR, "images")
CANDIDATES_PATH = os.path.join(DATASET_DIR, "candidates.json")

DANISH_LETTERS = re.compile(r"[æøåÆØÅ]")


def rows_with_danish_letters(df):
    """Rows whose verbatimDate or verbatimLocality contains æ/ø/å (either case)."""
    has_danish = (
        df["verbatimDate"].str.contains(DANISH_LETTERS, na=False)
        | df["verbatimLocality"].str.contains(DANISH_LETTERS, na=False)
    )
    return df[has_danish]


def extract_crops():
    """Run word detection on every Danish-candidate image, save each detected
    word crop as its own PNG, and write a {crop_filename: source_image_file}
    manifest for the labeling app. Returns the number of crops written."""
    os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    rows = rows_with_danish_letters(df)
    run_engine = ENGINES["doctr"]

    candidates = {}
    for _, row in rows.iterrows():
        image_file = row["image_file"]
        crop = get_search_crop(os.path.join(IMAGES_DIR, image_file))
        words = run_engine(crop)

        stem = os.path.splitext(image_file)[0]
        for i, (_, _, (x1, y1, x2, y2)) in enumerate(words):
            if x2 <= x1 or y2 <= y1:
                continue
            word_crop = crop[y1:y2, x1:x2]
            crop_filename = f"{stem}_{i:03d}.png"
            cv2.imwrite(os.path.join(IMAGES_OUT_DIR, crop_filename), word_crop)
            candidates[crop_filename] = image_file

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    return len(candidates)


if __name__ == "__main__":
    n = extract_crops()
    print(f"Wrote {n} candidate crops to {IMAGES_OUT_DIR}")
    print(f"Manifest: {CANDIDATES_PATH}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n museumscat-challenge pytest tests/test_extract_reco_crops.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the real extraction and manually verify**

Run: `conda run -n museumscat-challenge python -m src.extract_reco_crops`
Expected: prints a crop count (dozens to low hundreds, depending on how many of the 200 GT rows contain æøåÆØÅ) and the manifest path. Manually open a handful of files under `data/processed/reco_finetune/dataset/images/` to sanity-check the crops are legible single-word images, not empty/garbage boxes.

- [ ] **Step 6: Stage and propose a commit**

```bash
git add src/extract_reco_crops.py tests/test_extract_reco_crops.py
```
Propose commit message: `feat: extract Danish-candidate word crops for recognizer fine-tuning` — ask before committing.

---

### Task 2: Local word-crop labeling app

**Files:**
- Create: `src/label_crops_app.py`
- Test: `tests/test_label_crops_app.py`

**Interfaces:**
- Consumes: `data/processed/reco_finetune/dataset/candidates.json` (from Task 1), `data/processed/reco_finetune/dataset/images/*.png`.
- Produces: `next_unlabeled(candidates: dict, labels: dict, skipped: list) -> str | None`; on disk, `data/processed/reco_finetune/dataset/labels.json` (doctr's exact `{filename: text}` format — no spaces allowed in any value), `skipped.json` (list of filenames), `needs_help.json` (list of filenames flagged for Claude to help transcribe), all consumed by Task 3 (human labeling) and Task 5 (train/val split).

- [ ] **Step 1: Write the failing test for the pure "next crop" logic**

```python
# tests/test_label_crops_app.py
from src.label_crops_app import next_unlabeled


def test_returns_first_candidate_not_yet_labeled_or_skipped():
    candidates = {"a.png": "img1.jpg", "b.png": "img1.jpg", "c.png": "img2.jpg"}
    assert next_unlabeled(candidates, labels={"a.png": "Aabenraa"}, skipped=["b.png"]) == "c.png"


def test_returns_none_when_all_candidates_handled():
    candidates = {"a.png": "img1.jpg"}
    assert next_unlabeled(candidates, labels={"a.png": "Aabenraa"}, skipped=[]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n museumscat-challenge pytest tests/test_label_crops_app.py -v`
Expected: FAIL with `ImportError: cannot import name 'next_unlabeled'`

- [ ] **Step 3: Write the implementation**

```python
# src/label_crops_app.py
"""Local web app for manually transcribing the word crops extracted by
extract_reco_crops.py. Writes labels.json incrementally in doctr's
RecognitionDataset format (https://github.com/mindee/doctr/blob/main/references/recognition/README.md)
so progress is never lost.

Usage:
    conda run -n museumscat-challenge python -m src.label_crops_app
Then open http://localhost:8765 in a browser (WSL2 forwards localhost to Windows automatically).
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

DATASET_DIR = os.path.join("data", "processed", "reco_finetune", "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CANDIDATES_PATH = os.path.join(DATASET_DIR, "candidates.json")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.json")
SKIPPED_PATH = os.path.join(DATASET_DIR, "skipped.json")
NEEDS_HELP_PATH = os.path.join(DATASET_DIR, "needs_help.json")
PORT = 8765

DANISH_CHARS = ["æ", "ø", "å", "Æ", "Ø", "Å"]

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Danish crop labeling</title>
<style>
  body { font-family: sans-serif; max-width: 700px; margin: 40px auto; }
  #crop { max-width: 100%; border: 1px solid #ccc; image-rendering: pixelated; }
  #text { font-size: 20px; width: 100%; box-sizing: border-box; margin: 10px 0; }
  button { font-size: 16px; margin: 4px; padding: 6px 12px; }
  #progress { color: #666; }
  #done { display: none; font-size: 24px; }
</style>
</head>
<body>
  <div id="progress"></div>
  <img id="crop">
  <div>
    <input id="text" autocomplete="off" placeholder="transcribe this word">
  </div>
  <div id="danish-buttons"></div>
  <button onclick="save()">Save &amp; next (Enter)</button>
  <button onclick="skip()">Skip (illegible)</button>
  <button onclick="needsHelp()">Ask Claude for help</button>
  <div id="done">All candidates handled.</div>

<script>
let state = null;

function render() {
  if (!state.filename) {
    document.getElementById('crop').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    return;
  }
  document.getElementById('progress').textContent =
    `${state.labeled} labeled, ${state.skipped} skipped, ${state.needs_help} need help / ${state.total} total`;
  document.getElementById('crop').src = '/crop/' + state.filename;
  document.getElementById('text').value = '';
  document.getElementById('text').focus();
}

async function refresh() {
  const res = await fetch('/api/state');
  state = await res.json();
  render();
}

async function post(path, body) {
  const res = await fetch(path, { method: 'POST', body: JSON.stringify(body) });
  return res.json();
}

async function save() {
  const text = document.getElementById('text').value.trim();
  if (!text || text.includes(' ')) {
    alert('Text must be non-empty and contain no spaces (one word per crop).');
    return;
  }
  const result = await post('/api/label', { filename: state.filename, text: text });
  if (result.error) { alert(result.error); return; }
  await refresh();
}

async function skip() {
  await post('/api/skip', { filename: state.filename });
  await refresh();
}

async function needsHelp() {
  await post('/api/needs_help', { filename: state.filename });
  await refresh();
}

document.getElementById('text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') save();
});

const buttonsDiv = document.getElementById('danish-buttons');
for (const ch of DANISH_CHARS_PLACEHOLDER) {
  const btn = document.createElement('button');
  btn.textContent = ch;
  btn.onclick = () => {
    const input = document.getElementById('text');
    const pos = input.selectionStart;
    input.value = input.value.slice(0, pos) + ch + input.value.slice(pos);
    input.selectionStart = input.selectionEnd = pos + 1;
    input.focus();
  };
  buttonsDiv.appendChild(btn);
}

refresh();
</script>
</body>
</html>"""


def _load_json(path, default):
    """Read JSON from path, or return default if the file doesn't exist yet."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path, data):
    """Write JSON to path atomically (write to a temp file, then rename)."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def next_unlabeled(candidates, labels, skipped):
    """First candidate filename not yet in labels or skipped, or None if done."""
    for filename in candidates:
        if filename not in labels and filename not in skipped:
            return filename
    return None


class Handler(BaseHTTPRequestHandler):
    """Serves the single-page labeling UI and its small JSON/image API."""

    def _send_json(self, payload, status=200):
        """Write a JSON response with the given status code."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Route GET / (the page), GET /api/state, and GET /crop/<filename>."""
        if self.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/state":
            candidates = _load_json(CANDIDATES_PATH, {})
            labels = _load_json(LABELS_PATH, {})
            skipped = _load_json(SKIPPED_PATH, [])
            needs_help = _load_json(NEEDS_HELP_PATH, [])
            filename = next_unlabeled(candidates, labels, skipped)
            self._send_json({
                "filename": filename,
                "labeled": len(labels),
                "skipped": len(skipped),
                "needs_help": len(needs_help),
                "total": len(candidates),
            })
            return

        if self.path.startswith("/crop/"):
            filename = self.path[len("/crop/"):]
            crop_path = os.path.join(IMAGES_DIR, filename)
            same_dir = os.path.dirname(os.path.abspath(crop_path)) == os.path.abspath(IMAGES_DIR)
            if not same_dir or not os.path.isfile(crop_path):
                self.send_response(404)
                self.end_headers()
                return
            with open(crop_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        """Route POST /api/label, /api/skip, /api/needs_help."""
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        filename = payload["filename"]

        if self.path == "/api/label":
            text = payload["text"].strip()
            if not text or " " in text:
                self._send_json({"error": "text must be non-empty and contain no spaces"}, status=400)
                return
            labels = _load_json(LABELS_PATH, {})
            labels[filename] = text
            _save_json(LABELS_PATH, labels)
            self._send_json({"ok": True})
            return

        if self.path == "/api/skip":
            skipped = _load_json(SKIPPED_PATH, [])
            if filename not in skipped:
                skipped.append(filename)
            _save_json(SKIPPED_PATH, skipped)
            self._send_json({"ok": True})
            return

        if self.path == "/api/needs_help":
            needs_help = _load_json(NEEDS_HELP_PATH, [])
            if filename not in needs_help:
                needs_help.append(filename)
            _save_json(NEEDS_HELP_PATH, needs_help)
            skipped = _load_json(SKIPPED_PATH, [])
            if filename not in skipped:
                skipped.append(filename)
            _save_json(SKIPPED_PATH, skipped)
            self._send_json({"ok": True})
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        """Silence the default request logging (one line per click is noise)."""


def main():
    """Start the labeling server and block until interrupted."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Labeling app running at http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

Note: replace the `DANISH_CHARS_PLACEHOLDER` token in `INDEX_HTML` with the actual JS array literal built from `DANISH_CHARS` before writing the file, e.g. by formatting the template string with `json.dumps(DANISH_CHARS)` substituted in for that token at module load time, since the six characters can't be typed directly into this plan's code block reliably — do this substitution (`INDEX_HTML = INDEX_HTML.replace("DANISH_CHARS_PLACEHOLDER", json.dumps(DANISH_CHARS))`) right after the `INDEX_HTML` assignment in the real file.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n museumscat-challenge pytest tests/test_label_crops_app.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Manually verify the app runs**

Run: `conda run -n museumscat-challenge python -m src.label_crops_app`, open `http://localhost:8765`. Expected: a crop image loads, typing text + clicking one of the æøåÆØÅ buttons inserts the character at the cursor, "Save & next" advances and updates the progress counter, `data/processed/reco_finetune/dataset/labels.json` gets created/updated on disk. Stop the server with Ctrl-C.

- [ ] **Step 6: Stage and propose a commit**

```bash
git add src/label_crops_app.py tests/test_label_crops_app.py
```
Propose commit message: `feat: add local web app for transcribing Danish word crops` — ask before committing.

---

### Task 3: Manual labeling session (human checkpoint, no automated tests)

This task has no code — it's the actual transcription work, done through the app from Task 2.

- [ ] **Step 1: Run the app and label crops**

`conda run -n museumscat-challenge python -m src.label_crops_app`, open `http://localhost:8765`, transcribe crops. Aim for at least ~150-200 labeled crops (adjust based on how many candidates Task 1 actually produced) — enough for the frozen-backbone head fine-tune in Task 4/6 to have real signal, without turning this into a huge manual effort. Use "Skip" for genuinely illegible crops (bad detection box, blur) and "Ask Claude for help" for crops that are just hard to read.

- [ ] **Step 2: Claude reviews needs_help.json**

Once some crops are flagged, tell Claude to proceed. Claude reads `data/processed/reco_finetune/dataset/needs_help.json`, opens each listed crop under `data/processed/reco_finetune/dataset/images/` with the Read tool (image viewing), proposes a transcription per crop in chat (or "genuinely illegible, suggest skip"), and the user types the agreed transcription into the app themselves (Claude does not write directly into `labels.json` — keeps a human decision in the loop for training-label quality).

- [ ] **Step 3: Confirm the dataset is ready**

Run: `conda run -n museumscat-challenge python -c "import json; print(len(json.load(open('data/processed/reco_finetune/dataset/labels.json'))))"`
Expected: a count the user is satisfied with before moving to Task 5's split. No commit — `data/processed/` is gitignored.

---

### Task 4: Vendor doctr's training script + local CPU smoke test

**Files:**
- Create: `third_party/doctr_recognition/train.py` (vendored, unmodified)
- Create: `third_party/doctr_recognition/utils.py` (vendored, unmodified)
- Create: `third_party/doctr_recognition/README.md`

**Interfaces:**
- Produces: a working `python third_party/doctr_recognition/train.py crnn_vgg16_bn --vocab danish --pretrained --freeze-backbone ...` CLI, reused unmodified by Task 6's Kaggle run. No Python interfaces (external script, not imported).

- [ ] **Step 1: Vendor the two files at the pinned tag**

```bash
mkdir -p third_party/doctr_recognition
curl -s -o third_party/doctr_recognition/train.py \
  "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/train.py"
curl -s -o third_party/doctr_recognition/utils.py \
  "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/utils.py"
```

- [ ] **Step 2: Write the provenance README**

```markdown
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
```

- [ ] **Step 3: Run the local CPU smoke test with synthetic data**

The script falls back to its own synthetic `WordGenerator` when `--train_path`/`--val_path` are omitted, so this validates the whole model-build + vocab-reshape + freeze-backbone + train/val loop without needing the real labeled dataset yet:

```bash
conda run -n museumscat-challenge python third_party/doctr_recognition/train.py crnn_vgg16_bn \
  --vocab danish --pretrained --freeze-backbone \
  --epochs 1 --train-samples 1 --val-samples 1 --batch_size 8 --workers 0 \
  --output_dir /tmp/doctr_smoke_test --name smoke_test
```

Expected: no CUDA on this machine, so it logs `No accessible GPU, target device set to CPU.` and continues (not an error); it prints dataset sizes, then one epoch of training/validation loss, then `Validation loss decreased ... saving state...` and exits 0. Confirm `/tmp/doctr_smoke_test/smoke_test.pt` was written.

- [ ] **Step 4: Stage and propose a commit**

```bash
git add third_party/doctr_recognition/
```
Propose commit message: `feat: vendor doctr's recognition training script (pinned v1.0.1)` — ask before committing.

---

### Task 5: Train/val split of the labeled dataset

**Files:**
- Create: `src/split_reco_dataset.py`
- Test: `tests/test_split_reco_dataset.py`

**Interfaces:**
- Consumes: `data/processed/reco_finetune/dataset/labels.json` (from Task 3), `data/processed/reco_finetune/dataset/images/`.
- Produces: `split_labels(labels: dict, val_fraction: float, seed: int) -> tuple[dict, dict]` (train_labels, val_labels); on disk, `data/processed/reco_finetune/dataset/train/{images/,labels.json}` and `.../val/{images/,labels.json}`, consumed directly by Task 6's `--train_path`/`--val_path`.

- [ ] **Step 1: Write the failing test for the pure split logic**

```python
# tests/test_split_reco_dataset.py
from src.split_reco_dataset import split_labels


def test_split_is_deterministic_and_covers_all_labels():
    labels = {f"crop_{i}.png": f"word{i}" for i in range(20)}
    train_a, val_a = split_labels(labels, val_fraction=0.2, seed=7)
    train_b, val_b = split_labels(labels, val_fraction=0.2, seed=7)
    assert train_a == train_b and val_a == val_b
    assert set(train_a) | set(val_a) == set(labels)
    assert set(train_a) & set(val_a) == set()


def test_val_fraction_is_approximately_respected():
    labels = {f"crop_{i}.png": f"word{i}" for i in range(20)}
    _, val = split_labels(labels, val_fraction=0.2, seed=7)
    assert len(val) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n museumscat-challenge pytest tests/test_split_reco_dataset.py -v`
Expected: FAIL with `ImportError: cannot import name 'split_labels'`

- [ ] **Step 3: Write the implementation**

```python
# src/split_reco_dataset.py
"""Split the human-labeled Danish word-crop dataset into train/val folders
in the layout third_party/doctr_recognition/train.py expects
(images/ + labels.json in each).

Usage:
    conda run -n museumscat-challenge python -m src.split_reco_dataset
"""
import json
import os
import random
import shutil

DATASET_DIR = os.path.join("data", "processed", "reco_finetune", "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.json")
VAL_FRACTION = 0.2
SEED = 7


def split_labels(labels, val_fraction, seed):
    """Deterministically split a {filename: text} dict into (train, val) dicts."""
    filenames = sorted(labels)
    rng = random.Random(seed)
    rng.shuffle(filenames)
    n_val = round(len(filenames) * val_fraction)
    val_filenames = set(filenames[:n_val])
    train = {f: labels[f] for f in filenames if f not in val_filenames}
    val = {f: labels[f] for f in filenames if f in val_filenames}
    return train, val


def _write_split(name, split_labels_dict):
    """Copy each labeled crop into DATASET_DIR/<name>/images/ and write labels.json there."""
    split_dir = os.path.join(DATASET_DIR, name)
    split_images_dir = os.path.join(split_dir, "images")
    os.makedirs(split_images_dir, exist_ok=True)
    for filename in split_labels_dict:
        shutil.copy(os.path.join(IMAGES_DIR, filename), os.path.join(split_images_dir, filename))
    with open(os.path.join(split_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(split_labels_dict, f, ensure_ascii=False, indent=2)


def main():
    """Read labels.json, split it, and write train/ and val/ subfolders."""
    with open(LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    train, val = split_labels(labels, VAL_FRACTION, SEED)
    _write_split("train", train)
    _write_split("val", val)
    print(f"train: {len(train)} crops, val: {len(val)} crops")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n museumscat-challenge pytest tests/test_split_reco_dataset.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the real split and manually verify**

Run: `conda run -n museumscat-challenge python -m src.split_reco_dataset`
Expected: prints train/val counts; `data/processed/reco_finetune/dataset/train/images/` and `val/images/` are populated, each with a matching `labels.json`.

- [ ] **Step 6: Stage and propose a commit**

```bash
git add src/split_reco_dataset.py tests/test_split_reco_dataset.py
```
Propose commit message: `feat: split labeled Danish crops into train/val for fine-tuning` — ask before committing.

---

### Task 6: Package the dataset and run the real fine-tune on Kaggle

**Files:**
- Create: `src/package_reco_dataset.py`
- Create: `notebooks/kaggle_finetune_danish_recognizer.py`

**Interfaces:**
- Consumes: `data/processed/reco_finetune/dataset/train/`, `.../val/` (from Task 5).
- Produces: `data/processed/reco_finetune/kaggle_dataset/` (zipped, `dataset-metadata.json` included, ready for `kaggle datasets create`) and a checkpoint file the user downloads back to `models/reco_finetune/<exp_name>.pt`, consumed by Task 7.

- [ ] **Step 1: Write the packaging script**

```python
# src/package_reco_dataset.py
"""Package the train/val split into a folder ready to become a private
Kaggle Dataset (kaggle datasets create -p <this folder>) for the real
GPU fine-tune run. Does not call the Kaggle API itself -- see
docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md Task 6
for why that step stays manual/explicit.

Usage:
    conda run -n museumscat-challenge python -m src.package_reco_dataset
"""
import json
import os
import shutil

DATASET_DIR = os.path.join("data", "processed", "reco_finetune", "dataset")
OUT_DIR = os.path.join("data", "processed", "reco_finetune", "kaggle_dataset")
KAGGLE_DATASET_SLUG = "museumscat-danish-recognizer-crops"


def package():
    """Copy train/ and val/ into OUT_DIR and write dataset-metadata.json for kaggle CLI."""
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    shutil.copytree(os.path.join(DATASET_DIR, "train"), os.path.join(OUT_DIR, "train"))
    shutil.copytree(os.path.join(DATASET_DIR, "val"), os.path.join(OUT_DIR, "val"))

    metadata = {
        "title": "MuseumSCAT Danish recognizer crops",
        "id": f"CHANGE_ME/{KAGGLE_DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    with open(os.path.join(OUT_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Packaged at {OUT_DIR}")
    print("Edit dataset-metadata.json 'id' to '<your-kaggle-username>/" + KAGGLE_DATASET_SLUG + "' before upload.")


if __name__ == "__main__":
    package()
```

- [ ] **Step 2: Run it**

Run: `conda run -n museumscat-challenge python -m src.package_reco_dataset`
Expected: `data/processed/reco_finetune/kaggle_dataset/` created with `train/`, `val/`, `dataset-metadata.json`. Edit the `id` field to the user's actual Kaggle username.

- [ ] **Step 3: Upload the dataset (manual, user-triggered)**

Ask the user before running this — it touches their Kaggle account:
```bash
conda run -n museumscat-challenge kaggle datasets create -p data/processed/reco_finetune/kaggle_dataset --dir-mode zip
```
Then in the Kaggle web UI, open the new dataset's settings and confirm visibility is **Private** (per this plan's Global Constraints — the CC BY-NC 4.0 source images must not end up public).

- [ ] **Step 4: Write the Kaggle notebook script**

```python
# notebooks/kaggle_finetune_danish_recognizer.py
"""Paste each `# %%` cell below into a new Kaggle Notebook, in order.
Before running: create the notebook, attach the
'museumscat-danish-recognizer-crops' private dataset (from
src/package_reco_dataset.py) as input, and under Settings enable
Accelerator = GPU T4 x2 (or P100) and Internet = On.
"""

# %% Cell 1: install doctr and fetch the pinned vendored training script
# !pip install -q "python-doctr[torch]==1.0.1"
# !mkdir -p doctr_recognition
# !curl -s -o doctr_recognition/train.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/train.py"
# !curl -s -o doctr_recognition/utils.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/utils.py"

# %% Cell 2: locate the attached dataset's train/val folders
# import os
# INPUT_DIR = "/kaggle/input/museumscat-danish-recognizer-crops"
# TRAIN_PATH = os.path.join(INPUT_DIR, "train")
# VAL_PATH = os.path.join(INPUT_DIR, "val")
# assert os.path.isfile(os.path.join(TRAIN_PATH, "labels.json"))
# assert os.path.isfile(os.path.join(VAL_PATH, "labels.json"))

# %% Cell 3: run the fine-tune (freeze backbone, extend vocab to Danish)
# !cd doctr_recognition && python train.py crnn_vgg16_bn \
#     --vocab danish --pretrained --freeze-backbone \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 20 --batch_size 32 --lr 0.001 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn

# %% Cell 4: confirm the checkpoint was written -- download this file
# (Kaggle Notebook output pane -> /kaggle/working/danish_crnn_vgg16_bn.pt)
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn.pt")
# print("checkpoint ready to download: /kaggle/working/danish_crnn_vgg16_bn.pt")
```

- [ ] **Step 5: Run the notebook and download the checkpoint (manual, user-triggered)**

Ask the user before/while doing this — it consumes their GPU quota. Once the run finishes, download `danish_crnn_vgg16_bn.pt` from the notebook's Output pane into `models/reco_finetune/danish_crnn_vgg16_bn.pt` locally (`models/` is gitignored, this is expected to stay local).

- [ ] **Step 6: Stage and propose a commit**

```bash
git add src/package_reco_dataset.py notebooks/kaggle_finetune_danish_recognizer.py
```
Propose commit message: `feat: package Danish crop dataset and add Kaggle fine-tune notebook script` — ask before committing.

---

### Task 7: Evaluate the fine-tuned checkpoint against the baseline

**Files:**
- Modify: `src/benchmark_ocr.py`

**Interfaces:**
- Consumes: `models/reco_finetune/danish_crnn_vgg16_bn.pt` (from Task 6), `doctr.models.recognition.__dict__["crnn_vgg16_bn"]`, `doctr.datasets.VOCABS["danish"]` (both already used transitively by the vendored script).
- Produces: `ENGINES["doctr-finetuned-danish"]` added to the existing `src.benchmark_ocr.ENGINES` dict, so `src/score_ocr_dev.py doctr-finetuned-danish <n>` (already-existing script, unmodified) works immediately and its output is directly comparable to the `doctr (default, French vocab)` row already recorded in `tables_figures/eda/ocr_engine_comparison/notes.md`.

- [ ] **Step 1: Add the fine-tuned engine to `src/benchmark_ocr.py`**

Insert after `_make_doctr_multilingual_words` and before the `_doctr_multilingual_words = ...` line:

```python
FINETUNED_DANISH_CHECKPOINT = os.path.join("models", "reco_finetune", "danish_crnn_vgg16_bn.pt")


def _doctr_finetuned_danish_words(crop_bgr):
    """Run doctr on a single crop using the Danish-vocab fine-tuned
    crnn_vgg16_bn recognizer (see
    docs/superpowers/plans/2026-07-24-danish-recognizer-finetune.md), return
    list of (text, confidence, (x1,y1,x2,y2)) in crop pixel coordinates."""
    import torch
    from doctr.datasets import VOCABS
    from doctr.models import ocr_predictor, recognition

    if not hasattr(_doctr_finetuned_danish_words, "_predictor"):
        reco_model = recognition.__dict__["crnn_vgg16_bn"](pretrained=False, vocab=VOCABS["danish"])
        reco_model.load_state_dict(torch.load(FINETUNED_DANISH_CHECKPOINT, map_location="cpu", weights_only=True))
        _doctr_finetuned_danish_words._predictor = ocr_predictor(
            det_arch="db_resnet50", reco_arch=reco_model, pretrained=True, assume_straight_pages=True
        )

    h, w = crop_bgr.shape[:2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    result = _doctr_finetuned_danish_words._predictor([crop_rgb])

    words = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (rx1, ry1), (rx2, ry2) = word.geometry
                    words.append((word.value, word.confidence, (int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h))))
    return words
```

Then add `"doctr-finetuned-danish": _doctr_finetuned_danish_words,` to the `ENGINES` dict.

- [ ] **Step 2: Manually verify it loads and runs**

Run: `conda run -n museumscat-challenge python -m src.benchmark_ocr doctr-finetuned-danish 5`
Expected: no errors loading the checkpoint (shape mismatch would raise from `load_state_dict`), 5 composite images written to `tables_figures/eda/ocr_benchmark_doctr-finetuned-danish/` — inspect a couple to confirm æøåÆØÅ show up in recognized text where expected.

- [ ] **Step 3: Run the real comparison against the recorded baseline**

Run: `conda run -n museumscat-challenge python -m src.score_ocr_dev doctr-finetuned-danish 25`
Expected: prints per-field `mean_NED`/`AURC`. Compare directly against the `doctr (default, French vocab)` row in `tables_figures/eda/ocr_engine_comparison/notes.md` (`date NED 0.615 / AURC 0.315`, `locality NED 0.768 / AURC 0.704`) — same 25-row sample (seed=7), so this is apples-to-apples. Append a short "Fine-tune result" section to that notes.md file with the new numbers and a one-line decision (adopt / not worth it / needs more labeled data), matching that file's existing format.

- [ ] **Step 4: Stage and propose a commit**

```bash
git add src/benchmark_ocr.py tables_figures/eda/ocr_engine_comparison/notes.md
```
Propose commit message: `feat: add fine-tuned Danish-vocab recognizer engine, record eval vs baseline` — ask before committing.

---

## Self-Review

**Spec coverage:** "test de funcionamiento" → Task 4 Step 3 (local CPU smoke test with synthetic data, no GPU/dataset needed). "corrida real" → Task 6 (real Kaggle GPU run on the human-labeled dataset) + Task 7 (does it actually help, measured the same way as the existing engine comparison). "gratis" constraint on Kaggle → confirmed in Global Constraints and the plan header (free tier, phone-verification one-time cost only, no billing). Labeling app with Danish-character buttons + Claude-assist fallback → Task 2 + Task 3 Step 2.

**Placeholder scan:** no TBD/"add error handling"/"similar to Task N" phrasing; every code step has real, complete code (the one exception — the `DANISH_CHARS_PLACEHOLDER` token in Task 2 — is called out explicitly with the exact substitution to perform, not left vague).

**Type consistency:** `rows_with_danish_letters` (Task 1) takes a DataFrame, matching its test. `next_unlabeled` (Task 2) signature matches its test and its use inside `Handler.do_GET`. `split_labels` (Task 5) returns `(dict, dict)`, matching its test and `main()`'s unpacking. The Task 6 packaging script's `dataset-metadata.json` `id` field and Task 6's notebook script's `INPUT_DIR` are both keyed to the same slug (`museumscat-danish-recognizer-crops`). Task 7's checkpoint path (`models/reco_finetune/danish_crnn_vgg16_bn.pt`) matches Task 6 Step 5's download instruction and the `--name`/`--output_dir` used in the notebook script's Cell 3.
