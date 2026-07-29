# Card Word-Box Labeling, Danish Recognizer Fine-Tune, and Card Relevance Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a word-level, manually-boxed training set from the already-matched
cards (`data/processed/card_crops/card_text_matches.json`), fine-tune doctr's
`crnn_vgg16_bn` recognizer with Danish vocab on Kaggle's free GPU, run that
recognizer over every one of the ~6854 detected card crops (not just the
200-specimen dev set), and build a keyword-based classifier that separates
cards actually carrying `verbatimDate`/`verbatimLocality` text from confound
cards (collector, determination, museum boilerplate) — evaluated against the
323 cards already labeled by the matching app.

**Architecture:** A new local labeling app lets the user draw one bounding box
per word on every `matched` card (424 words across 162 cards); a dataset
builder crops those boxes into doctr's `{filename: text}` training format; the
vendored `doctr` training script fine-tunes `crnn_vgg16_bn` in two stages
(frozen-backbone warm-up, then full unfreeze at low LR — the recipe that
converged in an earlier, now-superseded attempt); the same Kaggle notebook
then runs the fine-tuned recognizer over every detected card crop, producing
one row per recognized word; and a keyword classifier (reusing the existing
`classify_field`) decides per-card relevance from those words, evaluated
against ground truth the matching app already produced.

**Tech Stack:** Python 3.11, `python-doctr[torch]`, `opencv-python-headless`,
stdlib `http.server` (labeling app, no new dependency), Kaggle Notebooks free
GPU tier (T4x2 or P100).

## Global Constraints

- All Python: snake_case names, `os`/`os.path` for paths, spaces around `=`
  in assignments, every function gets a one/few-line docstring — **except**
  the vendored files in `third_party/doctr_recognition/`, which stay
  byte-identical to upstream.
- Run everything with `conda run -n museumscat-challenge python ...`, never
  assume an activated shell.
- No new dependencies — `python-doctr[torch]`, `easyocr`, `segment-anything`,
  `tqdm` are already in `environment.yml`; the labeling app is stdlib-only.
- `data/processed/`, `models/` are gitignored — nothing under them is
  committed.
- The new Kaggle Dataset (word-box crops) must be created **private** — CC
  BY-NC 4.0 applies to the source images and anything derived from them.
- No script in this plan calls `kaggle datasets create`, `kaggle datasets
  version`, or otherwise touches the user's Kaggle account automatically —
  those steps are proposed as a command and run by the user (or Claude,
  asked explicitly in the moment).
- Don't ask for confirmation before `git commit` in this repo — commit
  directly at each task's commit step.
- This repo's existing convention (see `src/extract_card_crops.py`,
  `src/match_card_labels_app.py`): pure logic gets `tests/` pytest coverage;
  image/CV/HTTP-server code gets a manual "run it, inspect the output"
  verification step instead of mocked unit tests.
- The unmerged `worktree-danish-recognizer-finetune` worktree/branch is left
  as-is — not merged, not deleted, not read from by any script in this plan.

## File Structure

- `src/box_card_words_app.py` — new local web app: one bounding box per word,
  drawn by the user, on every `matched` card.
- `src/build_word_crop_dataset.py` — crops each boxed word out of its card
  image, writes doctr's `{filename: text}` training format.
- `third_party/doctr_recognition/{train.py,utils.py,README.md}` — vendored
  doctr recognition training script, pinned to `v1.0.1`.
- `src/split_reco_dataset_v2.py` — deterministic train/val split.
- `src/package_reco_dataset_v2.py` — zips train/val into a Kaggle-Dataset-ready
  folder.
- `notebooks/kaggle_finetune_and_predict_danish.py` — cell-marked script:
  fine-tune (two stages) + full-dataset prediction, pasted into one Kaggle
  Notebook.
- `src/benchmark_ocr.py` (modify) — add `ENGINES["doctr-finetuned-danish"]`.
- `src/evaluate_card_reco.py` — compares the fine-tuned recognizer against
  baseline directly on the val split's word crops.
- `src/classify_card_relevance.py` — keyword-based card relevance classifier,
  reusing `src.classify_field.classify_field`.

---

### Task 1: Word queue + resumability logic for the labeling app

**Files:**
- Create: `src/box_card_words_app.py` (pure functions only this task; the
  HTTP server is added in Task 2)
- Test: `tests/test_box_card_words_app.py`

**Interfaces:**
- Consumes: `data/processed/card_crops/card_text_matches.json` (dict,
  `crop_filename -> {status, selections}`, `selections` a list of `{field,
  index, text}` in the card's saved reading order — already on disk from
  `src.match_card_labels_app`).
- Produces: `build_word_queue(matches: dict) -> list[dict]` and
  `next_unboxed(queue: list[dict], word_boxes: dict) -> dict | None`, both
  consumed by Task 2's HTTP handlers.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_box_card_words_app.py
from src.box_card_words_app import build_word_queue, next_unboxed


def test_build_word_queue_splits_selections_into_words_in_order():
    matches = {
        "a_card00.jpeg": {
            "status": "matched",
            "selections": [
                {"field": "date", "index": 0, "text": "6/1862"},
                {"field": "locality", "index": 1, "text": "Kulhuset Jaegerspris"},
            ],
        },
        "a_card01.jpeg": {"status": "no_relevant_text", "selections": []},
    }
    queue = build_word_queue(matches)
    assert queue == [
        {"crop_filename": "a_card00.jpeg", "field": "date", "index": 0, "word_index": 0, "text": "6/1862"},
        {"crop_filename": "a_card00.jpeg", "field": "locality", "index": 1, "word_index": 0, "text": "Kulhuset"},
        {"crop_filename": "a_card00.jpeg", "field": "locality", "index": 1, "word_index": 1, "text": "Jaegerspris"},
    ]


def test_build_word_queue_excludes_skipped_and_no_relevant_text():
    matches = {
        "b_card00.jpeg": {"status": "skipped", "selections": []},
        "c_card00.jpeg": {"status": "no_relevant_text", "selections": []},
    }
    assert build_word_queue(matches) == []


def test_next_unboxed_skips_already_handled_words():
    queue = [
        {"crop_filename": "a.jpeg", "field": "date", "index": 0, "word_index": 0, "text": "6/1862"},
        {"crop_filename": "a.jpeg", "field": "locality", "index": 0, "word_index": 0, "text": "Kb"},
    ]
    word_boxes = {
        "a.jpeg": [{"field": "date", "index": 0, "word_index": 0, "text": "6/1862", "boxes": [[1, 2, 3, 4]]}],
    }
    assert next_unboxed(queue, word_boxes) == queue[1]


def test_next_unboxed_treats_explicit_skip_as_handled():
    queue = [{"crop_filename": "a.jpeg", "field": "date", "index": 0, "word_index": 0, "text": "6/1862"}]
    word_boxes = {"a.jpeg": [{"field": "date", "index": 0, "word_index": 0, "text": "6/1862", "boxes": []}]}
    assert next_unboxed(queue, word_boxes) is None


def test_next_unboxed_none_when_all_handled():
    assert next_unboxed([], {}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge pytest tests/test_box_card_words_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.box_card_words_app'`

- [ ] **Step 3: Write the implementation**

```python
# src/box_card_words_app.py
"""Local web app for manually drawing one bounding box per word on every
'matched' card in data/processed/card_crops/card_text_matches.json, to
build a word-level training set for a Danish-vocab doctr recognizer
fine-tune. See
docs/superpowers/specs/2026-07-29-card-word-boxes-danish-finetune-design.md.

Usage:
    conda run -n museumscat-challenge python -m src.box_card_words_app
Then open http://localhost:8767 in a browser (WSL2 forwards localhost to
Windows automatically).
"""
import json
import os

MATCHES_PATH = os.path.join("data", "processed", "card_crops", "card_text_matches.json")
IMAGES_DIR = os.path.join("data", "processed", "card_crops", "images")
WORD_BOXES_PATH = os.path.join("data", "processed", "card_crops", "word_boxes.json")
PORT = 8767


def build_word_queue(matches):
    """matches: the card_text_matches.json dict (crop_filename -> {status,
    selections}). Returns an ordered list of {crop_filename, field, index,
    word_index, text} -- one entry per whitespace-split word of every
    selection on every 'matched' card, in matches' own iteration order
    (already grouped by specimen/card from the matching app), each card's
    selections in their saved reading order, each selection's words
    left-to-right."""
    queue = []
    for crop_filename, entry in matches.items():
        if entry["status"] != "matched":
            continue
        for selection in entry["selections"]:
            words = selection["text"].split()
            for word_index, word in enumerate(words):
                queue.append({
                    "crop_filename": crop_filename,
                    "field": selection["field"],
                    "index": selection["index"],
                    "word_index": word_index,
                    "text": word,
                })
    return queue


def _word_key(entry):
    """The (crop_filename, field, index, word_index) tuple identifying a
    queue entry, matched against word_boxes.json's own per-word records."""
    return (entry["crop_filename"], entry["field"], entry["index"], entry["word_index"])


def next_unboxed(queue, word_boxes):
    """First queue entry not yet present (boxed or explicitly skipped) in
    word_boxes -- word_boxes maps crop_filename -> list of {field, index,
    word_index, text, boxes} dicts (boxes == [] means explicitly skipped).
    Returns None if every entry has been handled."""
    done = set()
    for crop_filename, words in word_boxes.items():
        for w in words:
            done.add((crop_filename, w["field"], w["index"], w["word_index"]))
    for entry in queue:
        if _word_key(entry) not in done:
            return entry
    return None


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


if __name__ == "__main__":
    pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge pytest tests/test_box_card_words_app.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/box_card_words_app.py tests/test_box_card_words_app.py
git commit -m "feat: add word queue + resumability logic for the word-box labeling app"
```

---

### Task 2: HTTP server + canvas box-drawing UI

**Files:**
- Modify: `src/box_card_words_app.py` (append server code to Task 1's file)

**Interfaces:**
- Consumes: `build_word_queue`, `next_unboxed`, `_load_json`, `_save_json`
  (Task 1, same file).
- Produces: a running app at `http://localhost:8767` writing
  `data/processed/card_crops/word_boxes.json`, consumed by Task 4's dataset
  builder.

- [ ] **Step 1: Append the HTML/JS page and the HTTP handler**

First, add `from http.server import BaseHTTPRequestHandler, HTTPServer` to
`src/box_card_words_app.py`'s existing top-of-file imports (alongside
`import json` / `import os`).

Then add to the end of the file, replacing the placeholder
`if __name__ == "__main__": pass` at the bottom:

```python
INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Card word boxes</title>
<style>
  body { font-family: sans-serif; max-width: 900px; margin: 40px auto; }
  #imgwrap { position: relative; display: inline-block; }
  #crop { max-width: 100%; display: block; }
  #overlay { position: absolute; top: 0; left: 0; cursor: crosshair; }
  #progress { color: #666; margin-bottom: 10px; }
  #word { font-size: 22px; font-weight: bold; margin: 10px 0; }
  button { font-size: 16px; margin: 4px 8px 4px 0; padding: 8px 14px; }
  #boxlist { color: #333; margin: 8px 0; }
  #done { display: none; font-size: 24px; }
</style>
</head>
<body>
  <div>
    <button onclick="goBack()" id="back-btn">Atras</button>
  </div>
  <div id="progress"></div>
  <div id="word"></div>
  <div id="imgwrap">
    <img id="crop">
    <canvas id="overlay"></canvas>
  </div>
  <div id="boxlist"></div>
  <div id="actions">
    <button onclick="undoBox()">Deshacer ultimo cuadro</button>
    <button onclick="commit('boxed')">Siguiente palabra (Enter)</button>
    <button onclick="commit('skipped')">No la encuentro / ilegible</button>
  </div>
  <div id="done">Cola completa.</div>

<script>
let state = null;
let boxes = [];
let drawing = false;
let startX = 0, startY = 0;

const img = document.getElementById('crop');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

function sizeCanvasToImage() {
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + 'px';
  canvas.style.height = img.clientHeight + 'px';
  redraw();
}

function toNatural(x, y) {
  const scaleX = img.naturalWidth / img.clientWidth;
  const scaleY = img.naturalHeight / img.clientHeight;
  return [Math.round(x * scaleX), Math.round(y * scaleY)];
}

function toDisplay(x, y) {
  const scaleX = img.clientWidth / img.naturalWidth;
  const scaleY = img.clientHeight / img.naturalHeight;
  return [x * scaleX, y * scaleY];
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#ff3333';
  ctx.lineWidth = 2;
  for (const [x1, y1, x2, y2] of boxes) {
    const [dx1, dy1] = toDisplay(x1, y1);
    const [dx2, dy2] = toDisplay(x2, y2);
    ctx.strokeRect(dx1, dy1, dx2 - dx1, dy2 - dy1);
  }
}

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
});

canvas.addEventListener('mouseup', (e) => {
  if (!drawing) return;
  drawing = false;
  const rect = canvas.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;
  const [x1, y1] = toNatural(Math.min(startX, endX), Math.min(startY, endY));
  const [x2, y2] = toNatural(Math.max(startX, endX), Math.max(startY, endY));
  if (x2 - x1 > 2 && y2 - y1 > 2) {
    boxes.push([x1, y1, x2, y2]);
    renderBoxList();
    redraw();
  }
});

function renderBoxList() {
  document.getElementById('boxlist').textContent = boxes.length + ' cuadro(s) para esta palabra';
}

function undoBox() {
  boxes.pop();
  renderBoxList();
  redraw();
}

function render() {
  document.getElementById('back-btn').disabled = !state.can_go_back;
  if (state.done) {
    document.getElementById('word').style.display = 'none';
    document.getElementById('imgwrap').style.display = 'none';
    document.getElementById('boxlist').style.display = 'none';
    document.getElementById('actions').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    document.getElementById('progress').textContent = state.overall_progress;
    return;
  }
  document.getElementById('progress').textContent = state.overall_progress;
  document.getElementById('word').textContent = "Palabra: '" + state.text + "' (" + state.field + ")";
  boxes = [];
  renderBoxList();
  img.src = '/crop/' + state.crop_filename;
}

img.onload = sizeCanvasToImage;
window.addEventListener('resize', sizeCanvasToImage);

async function refresh() {
  const res = await fetch('/api/state');
  state = await res.json();
  render();
}

async function goBack() {
  await fetch('/api/back', { method: 'POST' });
  await refresh();
}

async function commit(status) {
  if (status === 'boxed' && boxes.length === 0) {
    alert('Dibuja al menos un cuadro, o usa "No la encuentro".');
    return;
  }
  await fetch('/api/save', {
    method: 'POST',
    body: JSON.stringify({
      crop_filename: state.crop_filename,
      field: state.field,
      index: state.index,
      word_index: state.word_index,
      text: state.text,
      status: status,
      boxes: status === 'boxed' ? boxes : [],
    }),
  });
  await refresh();
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && state && !state.done) commit('boxed');
});

refresh();
</script>
</body>
</html>"""


QUEUE = []
CURRENT_INDEX = 0


class Handler(BaseHTTPRequestHandler):
    """Serves the word-box labeling UI and its small JSON/image API."""

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
            word_boxes = _load_json(WORD_BOXES_PATH, {})
            handled = sum(len(v) for v in word_boxes.values())
            if CURRENT_INDEX >= len(QUEUE):
                self._send_json({
                    "done": True,
                    "can_go_back": CURRENT_INDEX > 0,
                    "overall_progress": f"{handled}/{len(QUEUE)} palabras",
                })
                return
            entry = QUEUE[CURRENT_INDEX]
            self._send_json({
                "done": False,
                "can_go_back": CURRENT_INDEX > 0,
                "crop_filename": entry["crop_filename"],
                "field": entry["field"],
                "index": entry["index"],
                "word_index": entry["word_index"],
                "text": entry["text"],
                "overall_progress": f"{handled}/{len(QUEUE)} palabras",
            })
            return

        if self.path.startswith("/crop/"):
            filename = self.path[len("/crop/"):]
            image_path = os.path.join(IMAGES_DIR, filename)
            same_dir = os.path.dirname(os.path.abspath(image_path)) == os.path.abspath(IMAGES_DIR)
            if not same_dir or not os.path.isfile(image_path):
                self.send_response(404)
                self.end_headers()
                return
            with open(image_path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        """Route POST /api/save and POST /api/back."""
        global CURRENT_INDEX

        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/api/back":
            CURRENT_INDEX = max(0, CURRENT_INDEX - 1)
            self._send_json({"ok": True})
            return

        if self.path == "/api/save":
            crop_filename = payload["crop_filename"]
            status = payload["status"]
            boxes = payload.get("boxes", [])

            if status not in ("boxed", "skipped"):
                self._send_json({"error": f"unknown status {status!r}"}, status=400)
                return
            if status == "boxed" and not boxes:
                self._send_json({"error": "boxed status requires at least one box"}, status=400)
                return

            word_boxes = _load_json(WORD_BOXES_PATH, {})
            existing = word_boxes.setdefault(crop_filename, [])
            existing[:] = [
                w for w in existing
                if not (w["field"] == payload["field"] and w["index"] == payload["index"]
                        and w["word_index"] == payload["word_index"])
            ]
            existing.append({
                "field": payload["field"],
                "index": payload["index"],
                "word_index": payload["word_index"],
                "text": payload["text"],
                "boxes": boxes if status == "boxed" else [],
            })
            _save_json(WORD_BOXES_PATH, word_boxes)

            CURRENT_INDEX = min(len(QUEUE), CURRENT_INDEX + 1)
            self._send_json({"ok": True})
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        """Silence the default request logging (one line per click is noise)."""


def main():
    """Build the word queue, resume at the first not-yet-boxed word, then
    start the labeling server and block until interrupted."""
    global QUEUE, CURRENT_INDEX

    matches = _load_json(MATCHES_PATH, {})
    QUEUE = build_word_queue(matches)

    word_boxes = _load_json(WORD_BOXES_PATH, {})
    resume_entry = next_unboxed(QUEUE, word_boxes)
    CURRENT_INDEX = QUEUE.index(resume_entry) if resume_entry is not None else len(QUEUE)

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Word-box app running at http://localhost:{PORT}  ({len(QUEUE)} words in queue)")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Re-run Task 1's tests to confirm nothing broke**

Run: `conda run -n museumscat-challenge pytest tests/test_box_card_words_app.py -v`
Expected: PASS (5 tests, unchanged)

- [ ] **Step 3: Manually verify the app runs**

Run: `conda run -n museumscat-challenge python -m src.box_card_words_app`, open
`http://localhost:8767`. Expected: a card image loads with its current word
shown above it; dragging on the image draws a red rectangle; "Deshacer ultimo
cuadro" removes the last one; "Siguiente palabra" requires at least one box
(alerts otherwise) and advances; "No la encuentro" advances without requiring
a box; `data/processed/card_crops/word_boxes.json` is created/updated on
disk after each action; reloading the page resumes at the same word (not
from the start). Stop the server with Ctrl-C.

- [ ] **Step 4: Commit**

```bash
git add src/box_card_words_app.py
git commit -m "feat: add HTTP server and canvas box-drawing UI for word-box labeling app"
```

---

### Task 3: Manual labeling session (human checkpoint, no automated tests)

- [ ] **Step 1: Run the app and box words**

`conda run -n museumscat-challenge python -m src.box_card_words_app`, open
`http://localhost:8767`. Work through as many of the 424 words as feasible
(all 162 matched cards, ~2-3 words each on average). Use "No la encuentro"
for genuinely illegible/undetectable words rather than drawing a bad box.

- [ ] **Step 2: Confirm the annotations are ready**

Run:
```bash
conda run -n museumscat-challenge python -c "
import json
d = json.load(open('data/processed/card_crops/word_boxes.json'))
total = sum(len(v) for v in d.values())
boxed = sum(1 for v in d.values() for w in v if w['boxes'])
skipped = total - boxed
multi = sum(1 for v in d.values() for w in v if len(w['boxes']) > 1)
print(f'{boxed} boxed, {skipped} skipped, {multi} boxed with >1 box, {total} total handled')
"
```
Expected: a count the user is satisfied with before moving to Task 4 (aim for
most of the 424 words boxed — a handful of skips is fine). No commit —
`data/processed/` is gitignored.

---

### Task 4: Word-crop dataset builder

**Files:**
- Create: `src/build_word_crop_dataset.py`
- Test: `tests/test_build_word_crop_dataset.py`

**Interfaces:**
- Consumes: `data/processed/card_crops/word_boxes.json` (Task 2/3),
  `data/processed/card_crops/images/` (existing).
- Produces: `union_box(boxes: list[tuple]) -> tuple` and `build_dataset() ->
  tuple[int, int]` (crop count, multi-box-word count); on disk,
  `data/processed/reco_finetune_v2/dataset/{images/,labels.json}`, consumed
  by Task 6's split.

- [ ] **Step 1: Write the failing tests for `union_box`**

```python
# tests/test_build_word_crop_dataset.py
from src.build_word_crop_dataset import union_box


def test_union_box_single_box_passthrough():
    assert union_box([(10, 20, 30, 40)]) == (10, 20, 30, 40)


def test_union_box_two_boxes():
    assert union_box([(10, 20, 30, 40), (5, 50, 60, 70)]) == (5, 20, 60, 70)


def test_union_box_three_boxes():
    assert union_box([(0, 0, 10, 10), (20, 20, 30, 30), (5, 5, 8, 8)]) == (0, 0, 30, 30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n museumscat-challenge pytest tests/test_build_word_crop_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.build_word_crop_dataset'`

- [ ] **Step 3: Write the implementation**

```python
# src/build_word_crop_dataset.py
"""Build a word-level image+label training set for the doctr recognizer
fine-tune from data/processed/card_crops/word_boxes.json (produced by
src.box_card_words_app). See
docs/superpowers/specs/2026-07-29-card-word-boxes-danish-finetune-design.md.

Usage:
    conda run -n museumscat-challenge python -m src.build_word_crop_dataset
"""
import json
import os

import cv2

WORD_BOXES_PATH = os.path.join("data", "processed", "card_crops", "word_boxes.json")
CARD_IMAGES_DIR = os.path.join("data", "processed", "card_crops", "images")
DATASET_DIR = os.path.join("data", "processed", "reco_finetune_v2", "dataset")
IMAGES_OUT_DIR = os.path.join(DATASET_DIR, "images")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.json")


def union_box(boxes):
    """boxes: list of (x1, y1, x2, y2). Returns the minimal box enclosing
    all of them -- used when a word was drawn with more than one box
    (e.g. split across two lines)."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def build_dataset():
    """Crop every boxed (non-skipped) word from word_boxes.json out of its
    source card image, save it under IMAGES_OUT_DIR, and write labels.json
    in doctr's {filename: text} training format. Returns (crop_count,
    multi_box_count)."""
    os.makedirs(IMAGES_OUT_DIR, exist_ok=True)
    with open(WORD_BOXES_PATH, encoding="utf-8") as f:
        word_boxes = json.load(f)

    labels = {}
    multi_box_count = 0
    for crop_filename, words in word_boxes.items():
        card_img = None
        for w in words:
            if not w["boxes"]:
                continue
            if len(w["boxes"]) > 1:
                multi_box_count += 1
            x1, y1, x2, y2 = union_box(w["boxes"])
            if card_img is None:
                card_img = cv2.imread(os.path.join(CARD_IMAGES_DIR, crop_filename))
            word_crop = card_img[y1:y2, x1:x2]
            out_filename = f"{os.path.splitext(crop_filename)[0]}_{w['field']}{w['index']}_{w['word_index']}.png"
            cv2.imwrite(os.path.join(IMAGES_OUT_DIR, out_filename), word_crop)
            labels[out_filename] = w["text"]

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    return len(labels), multi_box_count


if __name__ == "__main__":
    n, multi = build_dataset()
    print(f"Wrote {n} word crops to {IMAGES_OUT_DIR} ({multi} words had more than one box)")
    print(f"Labels: {LABELS_PATH}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n museumscat-challenge pytest tests/test_build_word_crop_dataset.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the real build and manually verify**

Run: `conda run -n museumscat-challenge python -m src.build_word_crop_dataset`
Expected: prints a crop count and multi-box count. Open a handful of files
under `data/processed/reco_finetune_v2/dataset/images/` to confirm they're
tight single-word crops, and spot-check a few `labels.json` entries against
the source card in `data/processed/card_crops/images/` to confirm the text
matches.

- [ ] **Step 6: Commit**

```bash
git add src/build_word_crop_dataset.py tests/test_build_word_crop_dataset.py
git commit -m "feat: build word-crop training dataset from manually boxed cards"
```

---

### Task 5: Vendor doctr's training script + local CPU smoke test

**Files:**
- Create: `third_party/doctr_recognition/train.py` (vendored, unmodified)
- Create: `third_party/doctr_recognition/utils.py` (vendored, unmodified)
- Create: `third_party/doctr_recognition/README.md`

**Interfaces:**
- Produces: a working `python third_party/doctr_recognition/train.py
  crnn_vgg16_bn --vocab danish --pretrained --freeze-backbone ...` CLI,
  reused unmodified by Task 7's Kaggle notebook.

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

Unmodified from upstream -- kept byte-identical so it can be diffed against
a future re-vendor. Not covered by this repo's Python style conventions.

Fine-tunes doctr's own pretrained `crnn_vgg16_bn` recognizer with a wider
vocab (`--vocab danish`, already `english + "aeoeaa/AEOEAA"` in doctr's own
`VOCABS`), freezing the backbone (`--freeze-backbone`) so only the reshaped
classification head is retrained in Stage 1, then unfreezing for a full
low-LR Stage 2. Dataset format: a folder with `images/` (crop files) +
`labels.json` (`{filename: text}`, no spaces in any label -- every label
here is a single word by construction). See
docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md for how
this repo builds that dataset and uses this script.
```

- [ ] **Step 3: Run the local CPU smoke test with synthetic data**

The script falls back to its own synthetic `WordGenerator` when
`--train_path`/`--val_path` are omitted, so this validates the whole
model-build + vocab-reshape + freeze-backbone + train/val loop without
needing the real labeled dataset yet:

```bash
conda run -n museumscat-challenge python third_party/doctr_recognition/train.py crnn_vgg16_bn \
  --vocab danish --pretrained --freeze-backbone \
  --epochs 1 --train-samples 1 --val-samples 1 --batch_size 8 --workers 0 \
  --output_dir /tmp/doctr_smoke_test --name smoke_test
```

Expected: no CUDA on this machine, so it logs `No accessible GPU, target
device set to CPU.` and continues (not an error); it prints dataset sizes,
then one epoch of training/validation loss, then `Validation loss decreased
... saving state...` and exits 0. Confirm `/tmp/doctr_smoke_test/smoke_test.pt`
was written.

- [ ] **Step 4: Commit**

```bash
git add third_party/doctr_recognition/
git commit -m "feat: vendor doctr's recognition training script (pinned v1.0.1)"
```

---

### Task 6: Train/val split

**Files:**
- Create: `src/split_reco_dataset_v2.py`
- Test: `tests/test_split_reco_dataset_v2.py`

**Interfaces:**
- Consumes: `data/processed/reco_finetune_v2/dataset/labels.json` (Task 4),
  `.../images/`.
- Produces: `split_labels(labels: dict, val_fraction: float, seed: int) ->
  tuple[dict, dict]`; on disk, `.../dataset/train/{images/,labels.json}` and
  `.../val/{images/,labels.json}`, consumed by Task 7's packaging and Task
  9's evaluation.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split_reco_dataset_v2.py
from src.split_reco_dataset_v2 import split_labels


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

Run: `conda run -n museumscat-challenge pytest tests/test_split_reco_dataset_v2.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.split_reco_dataset_v2'`

- [ ] **Step 3: Write the implementation**

```python
# src/split_reco_dataset_v2.py
"""Split the word-level Danish recognizer training set (built by
src.build_word_crop_dataset) into train/val folders in the layout
third_party/doctr_recognition/train.py expects (images/ + labels.json in
each).

Usage:
    conda run -n museumscat-challenge python -m src.split_reco_dataset_v2
"""
import json
import os
import random
import shutil

DATASET_DIR = os.path.join("data", "processed", "reco_finetune_v2", "dataset")
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

Run: `conda run -n museumscat-challenge pytest tests/test_split_reco_dataset_v2.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the real split and manually verify**

Run: `conda run -n museumscat-challenge python -m src.split_reco_dataset_v2`
Expected: prints train/val counts; `train/images/` and `val/images/` are
populated, each with a matching `labels.json`.

- [ ] **Step 6: Commit**

```bash
git add src/split_reco_dataset_v2.py tests/test_split_reco_dataset_v2.py
git commit -m "feat: split word-crop dataset into train/val for fine-tuning"
```

---

### Task 7: Package the dataset and write the Kaggle notebook script

**Files:**
- Create: `src/package_reco_dataset_v2.py`
- Create: `notebooks/kaggle_finetune_and_predict_danish.py`

**Interfaces:**
- Consumes: `.../dataset/train/`, `.../dataset/val/` (Task 6).
- Produces: `data/processed/reco_finetune_v2/kaggle_dataset/` (zipped,
  ready for `kaggle datasets create`); a notebook script that, once run,
  produces `danish_crnn_vgg16_bn_stage2.pt` and `predictions.csv`, both
  downloaded by the user for Task 9/10.

- [ ] **Step 1: Write the packaging script**

```python
# src/package_reco_dataset_v2.py
"""Package the train/val split into a folder ready to become a private
Kaggle Dataset (kaggle datasets create -p <this folder>) for the Danish
recognizer fine-tune run. Does not call the Kaggle API itself -- see
docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md for
why that step stays manual/explicit.

Usage:
    conda run -n museumscat-challenge python -m src.package_reco_dataset_v2
"""
import json
import os
import shutil

DATASET_DIR = os.path.join("data", "processed", "reco_finetune_v2", "dataset")
OUT_DIR = os.path.join("data", "processed", "reco_finetune_v2", "kaggle_dataset")
KAGGLE_DATASET_SLUG = "museumscat-danish-recognizer-word-boxes"


def package():
    """Copy train/ and val/ into OUT_DIR and write dataset-metadata.json for kaggle CLI."""
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    shutil.copytree(os.path.join(DATASET_DIR, "train"), os.path.join(OUT_DIR, "train"))
    shutil.copytree(os.path.join(DATASET_DIR, "val"), os.path.join(OUT_DIR, "val"))

    metadata = {
        "title": "MuseumSCAT Danish recognizer word boxes",
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

Run: `conda run -n museumscat-challenge python -m src.package_reco_dataset_v2`
Expected: `data/processed/reco_finetune_v2/kaggle_dataset/` created with
`train/`, `val/`, `dataset-metadata.json`. Edit the `id` field to the user's
actual Kaggle username.

- [ ] **Step 3: Upload the dataset (manual, user-triggered)**

Ask the user before running this — it touches their Kaggle account:
```bash
conda run -n museumscat-challenge kaggle datasets create -p data/processed/reco_finetune_v2/kaggle_dataset --dir-mode zip
```
Then in the Kaggle web UI, confirm the new dataset's visibility is
**Private**.

- [ ] **Step 4: Write the Kaggle notebook script**

```python
# notebooks/kaggle_finetune_and_predict_danish.py
"""Paste each `# %%` cell below into a new Kaggle Notebook, in order.
Before running: create the notebook, attach two input datasets --
'museumscat-danish-recognizer-word-boxes' (from
src/package_reco_dataset_v2.py, Task 7) and whichever slug the existing
card-crops + manifest.csv dataset was uploaded under (already done, before
this plan) -- and under Settings enable Accelerator = GPU T4 x2 (or P100)
and Internet = On.

Two-stage recipe (frozen-backbone warm-up, then full unfreeze at low LR)
reused from the earlier, now-superseded word-crop fine-tune attempt, which
converged with this exact shape -- see
docs/superpowers/specs/2026-07-29-card-word-boxes-danish-finetune-design.md
for why that attempt's *dataset* is being replaced, not this recipe.
"""

# %% Cell 1: install doctr and fetch the pinned vendored training script
# !pip install -q "python-doctr[torch]==1.0.1"
# !mkdir -p doctr_recognition
# !curl -s -o doctr_recognition/train.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/train.py"
# !curl -s -o doctr_recognition/utils.py "https://raw.githubusercontent.com/mindee/doctr/v1.0.1/references/recognition/utils.py"

# %% Cell 2: locate the attached word-box dataset's train/val folders
# import os
# INPUT_DIR = "/kaggle/input/museumscat-danish-recognizer-word-boxes"
# TRAIN_PATH = os.path.join(INPUT_DIR, "train")
# VAL_PATH = os.path.join(INPUT_DIR, "val")
# assert os.path.isfile(os.path.join(TRAIN_PATH, "labels.json"))
# assert os.path.isfile(os.path.join(VAL_PATH, "labels.json"))

# %% Cell 3: Stage 1 -- frozen backbone, train the reinitialized head
# !cd doctr_recognition && python train.py crnn_vgg16_bn \
#     --vocab danish --pretrained --freeze-backbone --amp \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 1500 --batch_size 16 --lr 0.001 \
#     --early-stop --early-stop-epochs 120 --early-stop-delta 0.0005 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn_stage1

# %% Cell 4: confirm Stage 1's checkpoint was written
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn_stage1.pt")
# print("stage 1 checkpoint ready")

# %% Cell 5: Stage 2 -- unfreeze the whole network, fine-tune at a low LR from Stage 1's weights
# !cd doctr_recognition && python train.py crnn_vgg16_bn \
#     --vocab danish --resume /kaggle/working/danish_crnn_vgg16_bn_stage1.pt --amp \
#     --train_path {TRAIN_PATH} --val_path {VAL_PATH} \
#     --epochs 800 --batch_size 16 --lr 0.0001 --optim adamw --wd 0.0001 \
#     --early-stop --early-stop-epochs 100 --early-stop-delta 0.0005 \
#     --output_dir /kaggle/working --name danish_crnn_vgg16_bn_stage2

# %% Cell 6: confirm the final checkpoint was written
# import os
# assert os.path.isfile("/kaggle/working/danish_crnn_vgg16_bn_stage2.pt")
# print("final checkpoint ready")

# %% Cell 7: locate the already-uploaded card-crops dataset (images + manifest)
# import os
# CARDS_INPUT_DIR = "/kaggle/input/CHANGE_ME_TO_ACTUAL_SLUG"  # the card crops + manifest.csv dataset uploaded before this plan
# CARDS_IMAGES_DIR = os.path.join(CARDS_INPUT_DIR, "images")
# MANIFEST_PATH = os.path.join(CARDS_INPUT_DIR, "manifest.csv")
# assert os.path.isfile(MANIFEST_PATH)

# %% Cell 8: run the fine-tuned recognizer over every card crop, one row per recognized word
# import cv2
# import pandas as pd
# import torch
# from doctr.datasets import VOCABS
# from doctr.models import ocr_predictor, recognition
#
# reco_model = recognition.__dict__["crnn_vgg16_bn"](pretrained=False, vocab=VOCABS["danish"])
# reco_model.load_state_dict(
#     torch.load("/kaggle/working/danish_crnn_vgg16_bn_stage2.pt", map_location="cpu", weights_only=True)
# )
# predictor = ocr_predictor(det_arch="db_resnet50", reco_arch=reco_model, pretrained=True, assume_straight_pages=True)
#
# manifest = pd.read_csv(MANIFEST_PATH)
# rows = []
# for crop_filename in manifest["crop_filename"]:
#     img_bgr = cv2.imread(os.path.join(CARDS_IMAGES_DIR, crop_filename))
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     result = predictor([img_rgb])
#     for page in result.pages:
#         for block in page.blocks:
#             for line in block.lines:
#                 for word in line.words:
#                     rows.append({
#                         "crop_filename": crop_filename,
#                         "predicted_text": word.value,
#                         "confidence": word.confidence,
#                     })
# pd.DataFrame(rows, columns=["crop_filename", "predicted_text", "confidence"]).to_csv(
#     "/kaggle/working/predictions.csv", index=False
# )
# print(f"wrote {len(rows)} word rows for {manifest['crop_filename'].nunique()} cards")

# %% Cell 9: confirm predictions.csv was written
# import os
# assert os.path.isfile("/kaggle/working/predictions.csv")
# print("predictions ready: /kaggle/working/predictions.csv")
```

- [ ] **Step 5: Commit**

```bash
git add src/package_reco_dataset_v2.py notebooks/kaggle_finetune_and_predict_danish.py
git commit -m "feat: package word-crop dataset and add combined Kaggle fine-tune+predict notebook script"
```

---

### Task 8: Run the Kaggle notebook (manual, user-triggered, no automated tests)

- [ ] **Step 1: Create the notebook and run it**

Ask the user before/while doing this — it consumes their GPU quota. Create a
new Kaggle Notebook, paste in Task 7's cells in order, attach both input
datasets, enable GPU + Internet, and run all cells.

- [ ] **Step 2: Download the outputs**

Once the run finishes, download from the notebook's Output pane:
- `danish_crnn_vgg16_bn_stage2.pt` into
  `models/reco_finetune_v2/danish_crnn_vgg16_bn_stage2.pt` locally
  (`models/` is gitignored).
- `predictions.csv` into `data/processed/card_crops/predictions.csv`
  (`data/processed/` is gitignored).

---

### Task 9: Evaluate the fine-tuned checkpoint against baseline

**Files:**
- Modify: `src/benchmark_ocr.py`
- Create: `src/evaluate_card_reco.py`

**Interfaces:**
- Consumes: `models/reco_finetune_v2/danish_crnn_vgg16_bn_stage2.pt` (Task
  8), `data/processed/reco_finetune_v2/dataset/val/` (Task 6, already local
  — no download needed for this task), `src.metrics.ned` (existing).
- Produces: `ENGINES["doctr-finetuned-danish"]` added to
  `src.benchmark_ocr.ENGINES`; `evaluate(engine_name: str) -> float` (mean
  word NED) in `src.evaluate_card_reco`.

- [ ] **Step 1: Add the fine-tuned engine to `src/benchmark_ocr.py`**

Insert near the other `_make_doctr_*`/`_doctr_*` engine functions, before the
`ENGINES = {...}` dict:

```python
FINETUNED_DANISH_CHECKPOINT = os.path.join("models", "reco_finetune_v2", "danish_crnn_vgg16_bn_stage2.pt")


def _doctr_finetuned_danish_words(crop_bgr):
    """Run doctr on a single crop using the Danish-vocab fine-tuned
    crnn_vgg16_bn recognizer (see
    docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md),
    return list of (text, confidence, (x1,y1,x2,y2)) in crop pixel
    coordinates."""
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

Then add `"doctr-finetuned-danish": _doctr_finetuned_danish_words,` to the
`ENGINES` dict.

- [ ] **Step 2: Write `src/evaluate_card_reco.py`**

```python
# src/evaluate_card_reco.py
"""Evaluate the fine-tuned Danish-vocab doctr recognizer against the
baseline default recognizer, directly on the held-out val split's word
crops (data/processed/reco_finetune_v2/dataset/val/), which have exact
per-word ground truth from the manual box-labeling app -- no
classify_field/ROI approximation needed here. See
docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md.

Usage:
    conda run -n museumscat-challenge python -m src.evaluate_card_reco
"""
import json
import os

import cv2

from src.benchmark_ocr import ENGINES
from src.metrics import ned

VAL_DIR = os.path.join("data", "processed", "reco_finetune_v2", "dataset", "val")
VAL_IMAGES_DIR = os.path.join(VAL_DIR, "images")
VAL_LABELS_PATH = os.path.join(VAL_DIR, "labels.json")


def _recognize(engine_fn, image_path):
    """Run an ENGINES[...] callable on a single tight word crop and return
    its single best-guess text (empty string if it found nothing)."""
    crop = cv2.imread(image_path)
    words = engine_fn(crop)
    if not words:
        return ""
    return max(words, key=lambda w: w[1])[0]


def evaluate(engine_name):
    """Run ENGINES[engine_name] over every val word crop, return mean NED
    against the true word (case-insensitive, per src.metrics.ned)."""
    with open(VAL_LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    engine_fn = ENGINES[engine_name]
    errors = [ned(_recognize(engine_fn, os.path.join(VAL_IMAGES_DIR, filename)), truth)
              for filename, truth in labels.items()]
    return sum(errors) / len(errors)


if __name__ == "__main__":
    for name in ("doctr", "doctr-finetuned-danish"):
        mean_ned = evaluate(name)
        print(f"{name}: mean word NED = {mean_ned:.3f}")
```

- [ ] **Step 3: Manually verify it loads and runs**

Run: `conda run -n museumscat-challenge python -m src.evaluate_card_reco`
Expected: no errors loading the checkpoint (a shape mismatch would raise
from `load_state_dict`); prints mean word NED for both `doctr` and
`doctr-finetuned-danish`. This is the number that decides whether the
fine-tune is worth keeping — note it, no code changes required either way
(Task 12 doesn't gate on this, but it's worth recording the result in this
task's commit message or a short note if the numbers are notable).

- [ ] **Step 4: Commit**

```bash
git add src/benchmark_ocr.py src/evaluate_card_reco.py
git commit -m "feat: add fine-tuned Danish-vocab recognizer engine, evaluate on real word-level val split"
```

---

### Task 10: Ingest and sanity-check the full-dataset predictions (no automated tests)

- [ ] **Step 1: Confirm predictions.csv covers every detected card**

Run:
```bash
conda run -n museumscat-challenge python -c "
import pandas as pd
predictions = pd.read_csv('data/processed/card_crops/predictions.csv')
manifest = pd.read_csv('data/processed/card_crops/manifest.csv')
print('cards with at least one predicted word:', predictions['crop_filename'].nunique())
print('cards in manifest:', len(manifest))
print('cards with zero predicted words:', len(set(manifest['crop_filename']) - set(predictions['crop_filename'])))
"
```
Expected: `cards with at least one predicted word` close to (not necessarily
exactly) `cards in manifest` — some cards may legitimately have zero
detected words (blank/damaged crop). A large gap would indicate the Kaggle
run didn't cover the full manifest and should be re-checked before Task 11.

---

### Task 11: Card relevance classifier

**Files:**
- Create: `src/classify_card_relevance.py`
- Test: `tests/test_classify_card_relevance.py`

**Interfaces:**
- Consumes: `src.classify_field.classify_field` (existing, unmodified unless
  Step 3 surfaces gaps), `data/processed/card_crops/predictions.csv` (Task
  10), `data/processed/card_crops/card_text_matches.json` (existing).
- Produces: `is_relevant_card(words: list[str]) -> bool`,
  `evaluate(predictions_by_card: dict, matches: dict) -> tuple[float, float,
  float]` (precision, recall, accuracy), `apply_to_all() -> int` (row
  count), consumed by Task 12/13.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_classify_card_relevance.py
from src.classify_card_relevance import evaluate, is_relevant_card


def test_card_with_a_date_word_is_relevant():
    assert is_relevant_card(["27.5.2022", "Coll.", "Johansen"]) is True


def test_card_with_only_confound_words_is_not_relevant():
    assert is_relevant_card(["Coll.", "J.", "P.", "Johansen"]) is False


def test_empty_word_list_is_not_relevant():
    assert is_relevant_card([]) is False


def test_evaluate_computes_precision_recall_accuracy():
    matches = {
        "a.jpeg": {"status": "matched", "selections": []},
        "b.jpeg": {"status": "no_relevant_text", "selections": []},
        "c.jpeg": {"status": "skipped", "selections": []},
    }
    predictions_by_card = {
        "a.jpeg": ["27.5.2022"],   # correctly predicted relevant
        "b.jpeg": ["Coll.", "X"],  # correctly predicted not relevant
    }
    precision, recall, accuracy = evaluate(predictions_by_card, matches)
    assert precision == 1.0
    assert recall == 1.0
    assert accuracy == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge pytest tests/test_classify_card_relevance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.classify_card_relevance'`

- [ ] **Step 3: Write the implementation**

```python
# src/classify_card_relevance.py
"""Classify whether a card crop's recognized text actually carries a real
verbatimDate/verbatimLocality value, or is confound content (collector,
determination, museum boilerplate) -- reuses src.classify_field's
keyword-based date/locality/confound classifier per word, applied across
all the words doctr recognized on one card. See
docs/superpowers/plans/2026-07-29-card-word-boxes-danish-finetune.md.

Usage:
    conda run -n museumscat-challenge python -m src.classify_card_relevance
"""
import csv
import json
import os

from src.classify_field import classify_field

MATCHES_PATH = os.path.join("data", "processed", "card_crops", "card_text_matches.json")
PREDICTIONS_PATH = os.path.join("data", "processed", "card_crops", "predictions.csv")
OUTPUT_PATH = os.path.join("data", "processed", "card_crops", "predictions_with_relevance.csv")


def is_relevant_card(words):
    """words: list of recognized text snippets on one card. True if any of
    them classifies as date or locality (i.e. not confound) -- an empty
    list is never relevant."""
    return any(classify_field(word) != "confound" for word in words)


def evaluate(predictions_by_card, matches):
    """predictions_by_card: crop_filename -> list of recognized words.
    matches: card_text_matches.json dict. Returns (precision, recall,
    accuracy) of is_relevant_card against ground truth over every card
    whose status is 'matched' or 'no_relevant_text' (labeled ground truth;
    'skipped' cards are excluded since they're bad detections, not a
    relevance judgment)."""
    tp = fp = tn = fn = 0
    for crop_filename, entry in matches.items():
        if entry["status"] not in ("matched", "no_relevant_text"):
            continue
        truth = entry["status"] == "matched"
        predicted = is_relevant_card(predictions_by_card.get(crop_filename, []))
        if predicted and truth:
            tp += 1
        elif predicted and not truth:
            fp += 1
        elif not predicted and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    return precision, recall, accuracy


def _load_predictions_by_card():
    """Group predictions.csv's per-word rows into crop_filename -> list of
    recognized word strings."""
    predictions_by_card = {}
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            predictions_by_card.setdefault(row["crop_filename"], []).append(row["predicted_text"])
    return predictions_by_card


def apply_to_all():
    """Add a 'relevant' column to every row of predictions.csv, writing
    OUTPUT_PATH. Returns the number of rows written."""
    predictions_by_card = _load_predictions_by_card()
    with open(PREDICTIONS_PATH, encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = reader.fieldnames + ["relevant"]

    for row in rows:
        row["relevant"] = is_relevant_card(predictions_by_card.get(row["crop_filename"], []))

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    with open(MATCHES_PATH, encoding="utf-8") as f:
        matches = json.load(f)
    predictions_by_card = _load_predictions_by_card()
    precision, recall, accuracy = evaluate(predictions_by_card, matches)
    print(f"precision={precision:.3f} recall={recall:.3f} accuracy={accuracy:.3f}")

    n = apply_to_all()
    print(f"wrote {n} rows with relevance to {OUTPUT_PATH}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge pytest tests/test_classify_card_relevance.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/classify_card_relevance.py tests/test_classify_card_relevance.py
git commit -m "feat: add keyword-based card relevance classifier reusing classify_field"
```

---

### Task 12: Run the evaluation against labeled cards and iterate keywords (human checkpoint)

- [ ] **Step 1: Run the classifier's evaluation**

Run: `conda run -n museumscat-challenge python -m src.classify_card_relevance`
Expected: prints `precision=... recall=... accuracy=...` over the 323
labeled cards (162 `matched` + 161 `no_relevant_text`).

- [ ] **Step 2: Inspect misclassifications, extend keywords if needed**

If precision/recall look weak, find which cards are misclassified:
```bash
conda run -n museumscat-challenge python -c "
import json
from src.classify_card_relevance import is_relevant_card, _load_predictions_by_card

matches = json.load(open('data/processed/card_crops/card_text_matches.json'))
predictions_by_card = _load_predictions_by_card()
for crop_filename, entry in matches.items():
    if entry['status'] not in ('matched', 'no_relevant_text'):
        continue
    truth = entry['status'] == 'matched'
    predicted = is_relevant_card(predictions_by_card.get(crop_filename, []))
    if predicted != truth:
        print(crop_filename, 'truth=', truth, 'predicted=', predicted, predictions_by_card.get(crop_filename))
"
```
For any recurring confound pattern not already in
`src.classify_field.CONFOUND_KEYWORDS` (e.g. a museum stamp phrase, a
catalog-number format), add it there and re-run Step 1. This is the same
iterate-on-real-errors loop `classify_field` was originally built with — no
new file, just extending the existing keyword list.

- [ ] **Step 3: Commit if `classify_field.py` was changed**

```bash
git add src/classify_field.py
git commit -m "fix: extend CONFOUND_KEYWORDS based on card relevance evaluation errors"
```

---

### Task 13: Apply the relevance classifier to all card predictions

- [ ] **Step 1: Run the full application**

`apply_to_all()` already runs as part of `python -m
src.classify_card_relevance`'s `__main__` block (Task 11 Step 3), so this
step is a verification, not new code:

Run:
```bash
conda run -n museumscat-challenge python -c "
import pandas as pd
df = pd.read_csv('data/processed/card_crops/predictions_with_relevance.csv')
print(df['relevant'].value_counts())
print('rows:', len(df))
"
```
Expected: a `relevant` column present with a `True`/`False` split that's
directionally sensible (most specimens have 1-2 relevant cards and 0-3
confound cards, per the earlier EDA in
`tables_figures/eda/sam_card_detection_test/notes.md`).

- [ ] **Step 2: No commit**

`data/processed/` is gitignored — this file is a local artifact, not
committed.

---

## Verification (end-to-end)

1. `conda run -n museumscat-challenge pytest tests/ -v` — all unit tests
   green (existing tests plus the ~14 new ones from this plan).
2. `data/processed/reco_finetune_v2/dataset/{train,val}/` populated with
   real word crops + labels (Task 6).
3. `models/reco_finetune_v2/danish_crnn_vgg16_bn_stage2.pt` downloaded from
   Kaggle and loadable by `src.benchmark_ocr.ENGINES["doctr-finetuned-danish"]`
   (Task 9).
4. `data/processed/card_crops/predictions.csv` covers essentially all of
   `manifest.csv`'s ~6854 rows (Task 10).
5. `data/processed/card_crops/predictions_with_relevance.csv` has a
   `relevant` column, and `src.classify_card_relevance`'s printed
   precision/recall/accuracy on the 323 labeled cards is reviewed with the
   user (Task 12/13).
