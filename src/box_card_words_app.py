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
from http.server import BaseHTTPRequestHandler, HTTPServer

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
