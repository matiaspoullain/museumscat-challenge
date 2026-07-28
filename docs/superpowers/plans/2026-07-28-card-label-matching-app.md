# Card/Label Text-Matching App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local single-file web app where the user manually matches each
detected card crop under `data/processed/card_crops/` against the piece(s) of
`data/raw/train.csv` ground-truth text actually written on that card, in
reading order, producing training pairs for a future doctr fine-tune.

**Architecture:** One new stdlib-only file, `src/match_card_labels_app.py`.
Four pure, independently-testable functions (`build_candidates`,
`build_queue`, `next_unmatched`, `unused_candidates`) do all the real logic;
a thin `http.server`-based HTTP layer (one HTML page + a tiny JSON API) wraps
them for the browser UI. Match results persist to
`data/processed/card_crops/card_text_matches.json`, written atomically after
every save so progress is never lost.

**Tech Stack:** Python 3.11 stdlib only (`http.server`, `csv`, `json`, `os`)
-- no new dependencies. Full design rationale:
`docs/superpowers/specs/2026-07-28-card-label-matching-design.md`.

## Global Constraints

- All Python: snake_case names, `os`/`os.path` for paths, spaces around `=`
  in assignments, every function gets a one/few-line docstring.
- Run everything with `conda run -n museumscat-challenge python ...`, never
  assume an activated shell.
- `data/processed/` is gitignored -- `card_text_matches.json` never gets
  committed.
- No new dependencies (stdlib only, confirmed in the spec).
- This repo's existing convention (see `src/extract_card_crops.py`): pure
  logic gets `tests/` pytest coverage; HTTP/network/UI code gets a manual
  "run it, inspect the output" verification step instead of mocked unit
  tests. This plan follows that split.
- Ask before `git commit` is normally this user's global preference, but
  this repo has standing feedback on file to skip that confirmation here --
  each task still ends with a proposed commit message for the record.

---

## File Structure

- `src/match_card_labels_app.py` -- the whole app: four pure functions, the
  `http.server` `Handler`, the inline HTML/JS page, and `main()`.
- `tests/test_match_card_labels_app.py` -- unit tests for the four pure
  functions.

---

### Task 1: `build_candidates` -- flatten one specimen's ground truth into an ordered candidate list

**Files:**
- Create: `src/match_card_labels_app.py`
- Test: `tests/test_match_card_labels_app.py`

**Interfaces:**
- Consumes: nothing (pure function over a plain dict).
- Produces: `build_candidates(row: dict) -> list[dict]`, consumed by Task 2's
  `build_queue` and by the app's startup warning check in Task 4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_match_card_labels_app.py
from src.match_card_labels_app import build_candidates


def test_single_value_each_field():
    row = {"verbatimDate": "22.5.1977", "verbatimLocality": "Svinø strand"}
    assert build_candidates(row) == [
        {"field": "date", "index": 0, "text": "22.5.1977"},
        {"field": "locality", "index": 0, "text": "Svinø strand"},
    ]


def test_both_fields_missing_returns_empty():
    row = {"verbatimDate": "MISSING", "verbatimLocality": "MISSING"}
    assert build_candidates(row) == []


def test_one_field_missing_only_other_included():
    row = {"verbatimDate": "MISSING", "verbatimLocality": "Ti"}
    assert build_candidates(row) == [{"field": "locality", "index": 0, "text": "Ti"}]


def test_multi_pipe_field_indexed_in_order():
    row = {"verbatimDate": "6/1862", "verbatimLocality": "Ti | Kulhuset Jægerspris"}
    assert build_candidates(row) == [
        {"field": "date", "index": 0, "text": "6/1862"},
        {"field": "locality", "index": 0, "text": "Ti"},
        {"field": "locality", "index": 1, "text": "Kulhuset Jægerspris"},
    ]


def test_pieces_are_stripped_of_whitespace():
    row = {"verbatimDate": "MISSING", "verbatimLocality": " Al | Nord for Storskoven Vendsyssel"}
    assert build_candidates(row) == [
        {"field": "locality", "index": 0, "text": "Al"},
        {"field": "locality", "index": 1, "text": "Nord for Storskoven Vendsyssel"},
    ]


def test_missing_piece_within_a_multi_pipe_field_is_skipped():
    row = {"verbatimDate": "MISSING", "verbatimLocality": "Ti | MISSING | Rørvig"}
    assert build_candidates(row) == [
        {"field": "locality", "index": 0, "text": "Ti"},
        {"field": "locality", "index": 2, "text": "Rørvig"},
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge pytest tests/test_match_card_labels_app.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError: cannot import name 'build_candidates'`

- [ ] **Step 3: Write the implementation**

```python
# src/match_card_labels_app.py
"""Local web app for manually matching each detected card crop under
data/processed/card_crops/ against the piece(s) of data/raw/train.csv
ground-truth text actually written on that card, in reading order. Produces
training pairs for a future doctr fine-tune. See
docs/superpowers/specs/2026-07-28-card-label-matching-design.md.

Usage:
    conda run -n museumscat-challenge python -m src.match_card_labels_app
Then open http://localhost:8766 in a browser (WSL2 forwards localhost to
Windows automatically).
"""
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

MANIFEST_PATH = os.path.join("data", "processed", "card_crops", "manifest.csv")
TRAIN_CSV = os.path.join("data", "raw", "train.csv")
IMAGES_DIR = os.path.join("data", "processed", "card_crops", "images")
MATCHES_PATH = os.path.join("data", "processed", "card_crops", "card_text_matches.json")
PORT = 8766

QUEUE = []


def build_candidates(row):
    """row has verbatimDate/verbatimLocality string values (as read from
    train.csv). Returns an ordered list of {field, index, text} dicts, one
    per non-"MISSING" pipe-separated piece -- all date pieces first, then
    all locality pieces, each piece stripped of surrounding whitespace."""
    candidates = []
    for field, column in (("date", "verbatimDate"), ("locality", "verbatimLocality")):
        pieces = [p.strip() for p in row[column].split("|")]
        for index, text in enumerate(pieces):
            if text == "MISSING":
                continue
            candidates.append({"field": field, "index": index, "text": text})
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge pytest tests/test_match_card_labels_app.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/match_card_labels_app.py tests/test_match_card_labels_app.py
```
Propose commit message: `feat: add build_candidates for card/label matching app` -- ask before committing.

---

### Task 2: `build_queue` -- join manifest crops against train.csv candidates

**Files:**
- Modify: `src/match_card_labels_app.py`
- Test: `tests/test_match_card_labels_app.py`

**Interfaces:**
- Consumes: `build_candidates(row: dict) -> list[dict]` (Task 1).
- Produces: `build_queue(manifest_rows: list[dict], train_rows: list[dict]) -> list[dict]`,
  where each entry is `{crop_filename: str, source_image_file: str,
  card_index: int, candidates: list[dict]}`. Consumed by Task 3's
  `next_unmatched` and by the HTTP layer in Task 4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_match_card_labels_app.py (append)
from src.match_card_labels_app import build_queue


def test_groups_cards_by_specimen_in_card_index_order():
    manifest_rows = [
        {"crop_filename": "a_card01.jpeg", "source_image_file": "a.jpeg", "card_index": "1"},
        {"crop_filename": "a_card00.jpeg", "source_image_file": "a.jpeg", "card_index": "0"},
    ]
    train_rows = [{"image_file": "a.jpeg", "verbatimDate": "1901", "verbatimLocality": "Odense"}]
    queue = build_queue(manifest_rows, train_rows)
    assert [q["crop_filename"] for q in queue] == ["a_card00.jpeg", "a_card01.jpeg"]
    assert queue[0]["candidates"] == [
        {"field": "date", "index": 0, "text": "1901"},
        {"field": "locality", "index": 0, "text": "Odense"},
    ]


def test_excludes_specimens_with_no_candidates():
    manifest_rows = [{"crop_filename": "b_card00.jpeg", "source_image_file": "b.jpeg", "card_index": "0"}]
    train_rows = [{"image_file": "b.jpeg", "verbatimDate": "MISSING", "verbatimLocality": "MISSING"}]
    assert build_queue(manifest_rows, train_rows) == []


def test_specimen_with_candidates_but_no_manifest_rows_contributes_nothing():
    train_rows = [{"image_file": "c.jpeg", "verbatimDate": "1901", "verbatimLocality": "Odense"}]
    assert build_queue([], train_rows) == []


def test_preserves_train_csv_specimen_order():
    manifest_rows = [
        {"crop_filename": "y_card00.jpeg", "source_image_file": "y.jpeg", "card_index": "0"},
        {"crop_filename": "x_card00.jpeg", "source_image_file": "x.jpeg", "card_index": "0"},
    ]
    train_rows = [
        {"image_file": "x.jpeg", "verbatimDate": "1901", "verbatimLocality": "Odense"},
        {"image_file": "y.jpeg", "verbatimDate": "1902", "verbatimLocality": "Aarhus"},
    ]
    queue = build_queue(manifest_rows, train_rows)
    assert [q["source_image_file"] for q in queue] == ["x.jpeg", "y.jpeg"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge pytest tests/test_match_card_labels_app.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_queue'`

- [ ] **Step 3: Write the implementation**

Append to `src/match_card_labels_app.py`, after `build_candidates`:

```python
def build_queue(manifest_rows, train_rows):
    """manifest_rows: dicts with crop_filename/source_image_file/card_index
    (as read from manifest.csv -- card_index is a string). train_rows: dicts
    with image_file/verbatimDate/verbatimLocality (as read from train.csv).
    Returns an ordered list of {crop_filename, source_image_file, card_index,
    candidates} -- one entry per card crop, grouped by specimen in train.csv's
    own row order (then by card_index ascending within a specimen), skipping
    any specimen whose candidate list is empty (both fields "MISSING")."""
    by_source = {}
    for row in manifest_rows:
        by_source.setdefault(row["source_image_file"], []).append(row)

    queue = []
    for train_row in train_rows:
        image_file = train_row["image_file"]
        candidates = build_candidates(train_row)
        if not candidates:
            continue
        cards = sorted(by_source.get(image_file, []), key=lambda r: int(r["card_index"]))
        for card in cards:
            queue.append({
                "crop_filename": card["crop_filename"],
                "source_image_file": image_file,
                "card_index": int(card["card_index"]),
                "candidates": candidates,
            })
    return queue
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge pytest tests/test_match_card_labels_app.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/match_card_labels_app.py tests/test_match_card_labels_app.py
```
Propose commit message: `feat: add build_queue for card/label matching app` -- ask before committing.

---

### Task 3: `next_unmatched` and `unused_candidates` -- resumability and the end-of-specimen check

**Files:**
- Modify: `src/match_card_labels_app.py`
- Test: `tests/test_match_card_labels_app.py`

**Interfaces:**
- Consumes: the `queue` shape produced by `build_queue` (Task 2).
- Produces: `next_unmatched(queue: list[dict], matches: dict) -> dict | None`
  and `unused_candidates(candidates: list[dict], selections_by_card: list[list[dict]]) -> list[dict]`,
  both consumed by the HTTP layer in Task 4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_match_card_labels_app.py (append)
from src.match_card_labels_app import next_unmatched, unused_candidates


def test_next_unmatched_returns_first_not_in_matches():
    queue = [{"crop_filename": "a.jpeg"}, {"crop_filename": "b.jpeg"}]
    assert next_unmatched(queue, matches={"a.jpeg": {}}) == {"crop_filename": "b.jpeg"}


def test_next_unmatched_returns_none_when_all_done():
    queue = [{"crop_filename": "a.jpeg"}]
    assert next_unmatched(queue, matches={"a.jpeg": {}}) is None


def test_unused_candidates_finds_candidates_in_no_cards_selections():
    candidates = [
        {"field": "locality", "index": 0, "text": "Ti"},
        {"field": "locality", "index": 1, "text": "Rørvig"},
    ]
    selections_by_card = [[{"field": "locality", "index": 0, "text": "Ti"}]]
    assert unused_candidates(candidates, selections_by_card) == [
        {"field": "locality", "index": 1, "text": "Rørvig"}
    ]


def test_unused_candidates_empty_when_all_used_across_cards():
    candidates = [
        {"field": "locality", "index": 0, "text": "Ti"},
        {"field": "locality", "index": 1, "text": "Rørvig"},
    ]
    selections_by_card = [
        [{"field": "locality", "index": 0, "text": "Ti"}],
        [{"field": "locality", "index": 1, "text": "Rørvig"}],
    ]
    assert unused_candidates(candidates, selections_by_card) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n museumscat-challenge pytest tests/test_match_card_labels_app.py -v`
Expected: FAIL with `ImportError: cannot import name 'next_unmatched'`

- [ ] **Step 3: Write the implementation**

Append to `src/match_card_labels_app.py`, after `build_queue`:

```python
def next_unmatched(queue, matches):
    """First queue entry whose crop_filename isn't a key in matches, or None
    if every entry has already been matched."""
    for entry in queue:
        if entry["crop_filename"] not in matches:
            return entry
    return None


def unused_candidates(candidates, selections_by_card):
    """candidates: one specimen's full candidate list. selections_by_card: a
    list of that specimen's cards' already-saved selections (each itself a
    list of {field, index, text} dicts). Returns the candidates that don't
    appear in any card's selections, identified by (field, index)."""
    used = {(sel["field"], sel["index"]) for selections in selections_by_card for sel in selections}
    return [c for c in candidates if (c["field"], c["index"]) not in used]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n museumscat-challenge pytest tests/test_match_card_labels_app.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add src/match_card_labels_app.py tests/test_match_card_labels_app.py
```
Propose commit message: `feat: add next_unmatched and unused_candidates helpers` -- ask before committing.

---

### Task 4: HTTP server and browser UI

**Files:**
- Modify: `src/match_card_labels_app.py`

**Interfaces:**
- Consumes: `build_candidates`, `build_queue`, `next_unmatched`,
  `unused_candidates` (Tasks 1-3); reads `MANIFEST_PATH`, `TRAIN_CSV`,
  `IMAGES_DIR`, `MATCHES_PATH` (already defined as module constants in
  Task 1's Step 3).
- Produces: a running `http.server` on `PORT = 8766` serving `GET /`,
  `GET /api/state`, `GET /crop/<filename>`, `POST /api/save`; the JSON
  contract for each is fixed by this task and not expected to change later.

- [ ] **Step 1: Add the JSON persistence helpers**

Append to `src/match_card_labels_app.py`, after `unused_candidates`:

```python
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
```

- [ ] **Step 2: Add the inline HTML/JS page**

Append to `src/match_card_labels_app.py`, after `_save_json`:

```python
INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Card/label matching</title>
<style>
  body { font-family: sans-serif; max-width: 800px; margin: 40px auto; }
  #crop { max-width: 100%; border: 1px solid #ccc; margin-bottom: 10px; }
  #progress { color: #666; margin-bottom: 10px; }
  .candidate { display: inline-block; margin: 4px; padding: 6px 12px; border: 1px solid #888;
               border-radius: 4px; cursor: pointer; background: #f0f0f0; }
  #selected-list { min-height: 40px; border: 1px dashed #999; padding: 8px; margin: 10px 0; }
  .selected-chip { display: inline-block; margin: 2px; padding: 4px 8px; background: #3399ff;
                    color: white; border-radius: 4px; cursor: pointer; }
  button.action { font-size: 16px; margin: 4px 8px 4px 0; padding: 8px 14px; }
  #banner { background: #fff3cd; border: 1px solid #ffe69c; padding: 10px; margin: 10px 0; display: none; }
  #done { display: none; font-size: 24px; }
</style>
</head>
<body>
  <div id="progress"></div>
  <div id="banner"></div>
  <img id="crop">
  <h4>Candidatos</h4>
  <div id="candidates"></div>
  <h4>Seleccionados (el orden = el orden en la tarjeta)</h4>
  <div id="selected-list"></div>
  <div>
    <button class="action" onclick="save('matched')">Guardar y siguiente</button>
    <button class="action" onclick="save('no_relevant_text')">Sin texto relevante</button>
    <button class="action" onclick="save('skipped')">Saltar (mala detección)</button>
  </div>
  <div id="done">Cola completa.</div>

<script>
let state = null;
let selected = [];

function render() {
  if (state.done) {
    document.getElementById('crop').style.display = 'none';
    document.getElementById('candidates').style.display = 'none';
    document.getElementById('selected-list').style.display = 'none';
    document.querySelectorAll('button.action').forEach(b => b.style.display = 'none');
    document.getElementById('done').style.display = 'block';
    document.getElementById('progress').textContent = state.overall_progress;
    return;
  }
  document.getElementById('progress').textContent =
    `tarjeta ${state.specimen_position} del espécimen -- total ${state.overall_progress}`;
  document.getElementById('crop').src = '/crop/' + state.crop_filename;
  selected = [];
  renderCandidates();
  renderSelected();
}

function renderCandidates() {
  const div = document.getElementById('candidates');
  div.innerHTML = '';
  state.candidates.forEach((c) => {
    const btn = document.createElement('div');
    btn.className = 'candidate';
    btn.textContent = c.text;
    btn.onclick = () => {
      selected.push(c);
      renderSelected();
    };
    div.appendChild(btn);
  });
}

function renderSelected() {
  const div = document.getElementById('selected-list');
  div.innerHTML = '';
  selected.forEach((c, i) => {
    const chip = document.createElement('span');
    chip.className = 'selected-chip';
    chip.textContent = (i + 1) + '. ' + c.text;
    chip.onclick = () => {
      selected.splice(i, 1);
      renderSelected();
    };
    div.appendChild(chip);
  });
}

async function refresh() {
  const res = await fetch('/api/state');
  state = await res.json();
  render();
}

async function save(status) {
  if (status === 'matched' && selected.length === 0) {
    alert('Elegí al menos un candidato, o usá "Sin texto relevante".');
    return;
  }
  const res = await fetch('/api/save', {
    method: 'POST',
    body: JSON.stringify({ crop_filename: state.crop_filename, status: status, selections: selected }),
  });
  const result = await res.json();
  if (result.error) { alert(result.error); return; }

  const banner = document.getElementById('banner');
  if (result.unused_candidates && result.unused_candidates.length > 0) {
    const texts = result.unused_candidates.map((c) => '"' + c.text + '"').join(', ');
    banner.textContent = 'Atención: ' + texts + ' no fue usado en ninguna tarjeta de este espécimen.';
    banner.style.display = 'block';
  } else {
    banner.style.display = 'none';
  }

  await refresh();
}

refresh();
</script>
</body>
</html>"""
```

- [ ] **Step 3: Add the `Handler` class**

Append to `src/match_card_labels_app.py`, after `INDEX_HTML`:

```python
class Handler(BaseHTTPRequestHandler):
    """Serves the matching UI and its small JSON/image API."""

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
            matches = _load_json(MATCHES_PATH, {})
            counts = {"matched": 0, "no_relevant_text": 0, "skipped": 0}
            for m in matches.values():
                counts[m["status"]] = counts.get(m["status"], 0) + 1
            total_done = sum(counts.values())

            entry = next_unmatched(QUEUE, matches)
            if entry is None:
                self._send_json({"done": True, "overall_progress": f"{total_done}/{len(QUEUE)}"})
                return

            specimen_cards = [q for q in QUEUE if q["source_image_file"] == entry["source_image_file"]]
            position = [c["crop_filename"] for c in specimen_cards].index(entry["crop_filename"]) + 1
            self._send_json({
                "done": False,
                "crop_filename": entry["crop_filename"],
                "candidates": entry["candidates"],
                "specimen_position": f"{position}/{len(specimen_cards)}",
                "overall_progress": f"{total_done}/{len(QUEUE)}",
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
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        """Route POST /api/save."""
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))

        if self.path == "/api/save":
            crop_filename = payload["crop_filename"]
            status = payload["status"]
            selections = payload.get("selections", []) if status == "matched" else []

            if status == "matched" and not selections:
                self._send_json({"error": "matched status requires at least one selection"}, status=400)
                return

            matches = _load_json(MATCHES_PATH, {})
            matches[crop_filename] = {"status": status, "selections": selections}
            _save_json(MATCHES_PATH, matches)

            entry = next(q for q in QUEUE if q["crop_filename"] == crop_filename)
            specimen_cards = [q for q in QUEUE if q["source_image_file"] == entry["source_image_file"]]
            is_last_card = specimen_cards[-1]["crop_filename"] == crop_filename

            unused = []
            if is_last_card:
                selections_by_card = [
                    matches[c["crop_filename"]]["selections"]
                    for c in specimen_cards if c["crop_filename"] in matches
                ]
                unused = unused_candidates(entry["candidates"], selections_by_card)

            self._send_json({"ok": True, "unused_candidates": unused})
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        """Silence the default request logging (one line per click is noise)."""
```

- [ ] **Step 4: Add `main()`**

Append to `src/match_card_labels_app.py`, after `Handler`:

```python
def main():
    """Build the queue, warn about any relevant specimen with zero detected
    cards, then start the matching server and block until interrupted."""
    global QUEUE

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    with open(TRAIN_CSV, encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))

    QUEUE = build_queue(manifest_rows, train_rows)

    detected_sources = {row["source_image_file"] for row in manifest_rows}
    missing = [row["image_file"] for row in train_rows
               if build_candidates(row) and row["image_file"] not in detected_sources]
    if missing:
        print(f"warning: {len(missing)} specimens have relevant ground truth but zero detected cards:")
        for image_file in missing:
            print(f"  {image_file}")

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Matching app running at http://localhost:{PORT}  ({len(QUEUE)} cards in queue)")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the full test suite once more to confirm nothing broke**

Run: `conda run -n museumscat-challenge pytest tests/test_match_card_labels_app.py -v`
Expected: PASS (14 tests) -- this task only added server/UI code after the
pure functions, so the existing tests must still be untouched and green.

- [ ] **Step 6: Commit**

```bash
git add src/match_card_labels_app.py
```
Propose commit message: `feat: add HTTP server and browser UI for card/label matching app` -- ask before committing.

---

### Task 5: Manual run and verification against real data

This task has no new code -- it's the actual "does it work" checkpoint,
using the real `data/processed/card_crops/` output from the Kaggle
extraction (382 train-set card crops already on disk as of this plan).

- [ ] **Step 1: Start the app**

Run: `conda run -n museumscat-challenge python -m src.match_card_labels_app`
Expected: prints `warning: N specimens have relevant ground truth but zero
detected cards:` followed by a list (some specimens' cards may not have been
detected by SAM -- this is expected information, not a bug), then `Matching
app running at http://localhost:8766 (... cards in queue)`.

- [ ] **Step 2: Open the app and match a handful of specimens**

Open `http://localhost:8766`. Expected: the first queued card's image loads,
its specimen's candidates appear as clickable chips, clicking a few adds
them in order to the "Seleccionados" list (numbered), clicking a selected
chip removes it, "Guardar y siguiente" advances to the next card and updates
the progress line. Match at least one multi-card specimen fully (e.g. one
of the `Ti | ...`-style rows) and confirm:
- Picking the same candidate on two different cards of the same specimen is
  allowed (no error).
- Clicking "Guardar y siguiente" with zero candidates selected shows the
  alert and does not advance.
- Clicking "Sin texto relevante" on a confound-only card advances without
  requiring a selection.
- Clicking "Saltar (mala detección)" advances and records `status:
  "skipped"`.
- After finishing every card of one specimen, if a candidate was never
  picked on any of its cards, the yellow banner appears with that text
  named explicitly.

- [ ] **Step 3: Confirm resumability**

Stop the server (Ctrl-C) after matching a few cards, then restart it (Step
1's command again). Expected: the app resumes at the first not-yet-matched
card, not from the beginning.

- [ ] **Step 4: Inspect the output file**

Run: `conda run -n museumscat-challenge python -c "import json; m = json.load(open('data/processed/card_crops/card_text_matches.json')); print(len(m)); print(list(m.items())[:3])"`
Expected: a count matching how many cards were matched in Steps 2-3, and the
printed sample entries have the `{status, selections}` shape from the spec.
No commit needed for this file -- `data/processed/` is gitignored.

---

## Self-Review

**Spec coverage:** candidate-pool construction (dates then localities,
`MISSING` pieces skipped, whitespace-stripped) -> Task 1. Queue grouping by
specimen in train.csv order, card_index ascending, MISSING-only specimens
excluded -> Task 2. Resumability (`next_unmatched`) and the end-of-specimen
unused-candidate banner -> Task 3. Free reuse of a candidate across cards of
the same specimen -> no code needs to *prevent* it, so this is implicitly
satisfied by Tasks 1-3 never tracking a "consumed" state; Task 5 Step 2
explicitly verifies it's allowed. The three save actions (matched /
no_relevant_text / skipped) and the "Guardar" empty-selection guard -> Task
4's `/api/save` handler + frontend `save()`. Directory-traversal safety on
crop serving -> Task 4's `/crop/<filename>` route. Atomic JSON writes and the
`card_text_matches.json` output shape -> Task 4 Step 1. The
zero-detected-cards startup warning -> Task 4 Step 4 (`main()`).

**Placeholder scan:** no TBD/TODO/"add error handling" phrasing; every code
step has complete, real code.

**Type consistency:** `build_candidates` returns `list[dict]` with
`{field, index, text}` keys, used identically by `build_queue` (Task 2),
`unused_candidates` (Task 3), and the `/api/state`/`/api/save` JSON payloads
(Task 4) -- checked, no renaming drift. `build_queue`'s entries
(`crop_filename`, `source_image_file`, `card_index`, `candidates`) are
consumed with the same key names by `next_unmatched`, `unused_candidates`'s
caller in `/api/save`, and `main()`. `MATCHES_PATH` entries
(`{status, selections}`) match between `/api/save`'s write and Task 5 Step
4's inspection.
