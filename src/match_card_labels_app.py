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
    document.getElementById('progress').textContent = state.overall_progress +
      ` (matched: ${state.counts.matched}, skipped: ${state.counts.skipped}, ` +
      `no_relevant_text: ${state.counts.no_relevant_text})`;
    return;
  }
  document.getElementById('progress').textContent =
    `tarjeta ${state.specimen_position} del espécimen -- total ${state.overall_progress} ` +
    `(matched: ${state.counts.matched}, skipped: ${state.counts.skipped}, ` +
    `no_relevant_text: ${state.counts.no_relevant_text})`;
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
  let res;
  try {
    res = await fetch('/api/save', {
      method: 'POST',
      body: JSON.stringify({ crop_filename: state.crop_filename, status: status, selections: selected }),
    });
  } catch (e) {
    alert('No se pudo conectar con el servidor.');
    return;
  }

  let result;
  try {
    result = await res.json();
  } catch (e) {
    result = null;
  }

  if (!res.ok) {
    alert(result && result.error ? result.error : 'Error del servidor, revisá la consola.');
    return;
  }
  if (result && result.error) { alert(result.error); return; }

  const banner = document.getElementById('banner');
  if (result && result.unused_candidates && result.unused_candidates.length > 0) {
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
                self._send_json({
                    "done": True,
                    "overall_progress": f"{total_done}/{len(QUEUE)}",
                    "counts": counts,
                })
                return

            specimen_cards = [q for q in QUEUE if q["source_image_file"] == entry["source_image_file"]]
            position = [c["crop_filename"] for c in specimen_cards].index(entry["crop_filename"]) + 1
            self._send_json({
                "done": False,
                "crop_filename": entry["crop_filename"],
                "candidates": entry["candidates"],
                "specimen_position": f"{position}/{len(specimen_cards)}",
                "overall_progress": f"{total_done}/{len(QUEUE)}",
                "counts": counts,
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
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "malformed request body"}, status=400)
            return

        if self.path == "/api/save":
            try:
                crop_filename = payload["crop_filename"]
                status = payload["status"]
            except KeyError:
                self._send_json({"error": "malformed request body"}, status=400)
                return

            if status not in ("matched", "no_relevant_text", "skipped"):
                self._send_json({"error": f"unknown status {status!r}"}, status=400)
                return

            entry = next((q for q in QUEUE if q["crop_filename"] == crop_filename), None)
            if entry is None:
                self._send_json(
                    {"error": f"{crop_filename} is not in the queue -- reload the page"}, status=400
                )
                return

            selections = payload.get("selections", []) if status == "matched" else []

            if status == "matched" and not selections:
                self._send_json({"error": "matched status requires at least one selection"}, status=400)
                return

            matches = _load_json(MATCHES_PATH, {})
            matches[crop_filename] = {"status": status, "selections": selections}
            _save_json(MATCHES_PATH, matches)

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
