"""Local web app for manually transcribing the word crops extracted by
extract_reco_crops.py. Writes labels.json incrementally in doctr's
RecognitionDataset format (https://github.com/mindee/doctr/blob/main/references/recognition/README.md)
so progress is never lost. Also writes confidences.json ({filename: 0-1
float}, default 1.0) alongside it, for the labeler's own self-reported
confidence in each transcription -- kept in a separate file since doctr's
training script requires labels.json to be exactly {filename: text}.
Candidates are served least-doctr-confident first (see next_unlabeled) and
the text field is prefilled with doctr's own guess, so the labeler mostly
confirms or corrects rather than transcribing from scratch.

Usage:
    conda run -n museumscat-challenge python -m src.old.label_crops_app
Then open http://localhost:8765 in a browser (WSL2 forwards localhost to Windows automatically).
"""
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

DATASET_DIR = os.path.join("data", "processed", "reco_finetune", "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ORIGINAL_IMAGES_DIR = os.path.join("data", "raw", "images")
TRAIN_CSV = os.path.join("data", "raw", "train.csv")
CANDIDATES_PATH = os.path.join(DATASET_DIR, "candidates.json")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.json")
SKIPPED_PATH = os.path.join(DATASET_DIR, "skipped.json")
NEEDS_HELP_PATH = os.path.join(DATASET_DIR, "needs_help.json")
CONFIDENCES_PATH = os.path.join(DATASET_DIR, "confidences.json")
DEFAULT_CONFIDENCE = 1.0
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
  #doctr-info { color: #666; font-style: italic; min-height: 1.2em; }
  #ground-truth { color: #364; background: #eef7ee; padding: 4px 8px; border-radius: 4px; min-height: 1.2em; }
  #done { display: none; font-size: 24px; }
</style>
</head>
<body>
  <div id="progress"></div>
  <div id="ground-truth"></div>
  <img id="crop">
  <div id="doctr-info"></div>
  <div>
    <input id="text" autocomplete="off" placeholder="transcribe this word">
  </div>
  <div id="danish-buttons"></div>
  <div>
    <label for="confidence">confidence: <span id="confidence-value">1.0</span></label><br>
    <input id="confidence" type="range" min="0" max="1" step="0.1" value="1">
  </div>
  <button onclick="save()">Save &amp; next (Enter)</button>
  <button onclick="skip()">Skip (illegible)</button>
  <button onclick="needsHelp()">Ask Claude for help</button>
  <button onclick="goBack()">&laquo; Back (fix a mistake)</button>
  <button onclick="openOriginal()">Open original image</button>
  <div id="done">All candidates handled.</div>

<script>
let state = null;
let history = [];
let viewingHistory = false;

function render(prefill) {
  if (!state.filename) {
    document.getElementById('crop').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    return;
  }
  document.getElementById('crop').style.display = '';
  document.getElementById('done').style.display = 'none';
  document.getElementById('progress').textContent =
    `${state.labeled} labeled, ${state.skipped} skipped, ${state.needs_help} need help / ${state.total} total` +
    (viewingHistory ? ' -- editing a previous crop' : '');
  document.getElementById('crop').src = '/crop/' + state.filename;
  document.getElementById('ground-truth').textContent =
    `this card's known transcription -- date: "${state.true_verbatim_date}", locality: "${state.true_verbatim_locality}"`;
  document.getElementById('doctr-info').textContent =
    state.doctr_text !== null && state.doctr_text !== undefined
      ? `doctr's guess: "${state.doctr_text}" (confidence ${state.doctr_confidence.toFixed(2)})`
      : '';
  document.getElementById('text').value = prefill && prefill.text ? prefill.text : (state.doctr_text || '');
  const conf = prefill && prefill.confidence !== null && prefill.confidence !== undefined ? prefill.confidence : 1;
  document.getElementById('confidence').value = String(conf);
  document.getElementById('confidence-value').textContent = parseFloat(conf).toFixed(1);
  document.getElementById('text').focus();
}

async function refresh() {
  viewingHistory = false;
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
  const confidence = parseFloat(document.getElementById('confidence').value);
  const result = await post('/api/label', { filename: state.filename, text: text, confidence: confidence });
  if (result.error) { alert(result.error); return; }
  if (!viewingHistory) history.push(state.filename);
  await refresh();
}

async function skip() {
  if (!viewingHistory) history.push(state.filename);
  await post('/api/skip', { filename: state.filename });
  await refresh();
}

async function needsHelp() {
  if (!viewingHistory) history.push(state.filename);
  await post('/api/needs_help', { filename: state.filename });
  await refresh();
}

function openOriginal() {
  if (!state.source) return;
  window.open('/original/' + encodeURIComponent(state.source), '_blank');
}

async function goBack() {
  if (history.length === 0) {
    alert('No previous crop to go back to.');
    return;
  }
  const filename = history.pop();
  const res = await fetch('/api/item?filename=' + encodeURIComponent(filename));
  const item = await res.json();
  viewingHistory = true;
  state = {
    ...state,
    filename: filename,
    doctr_text: item.doctr_text,
    doctr_confidence: item.doctr_confidence,
    source: item.source,
    true_verbatim_date: item.true_verbatim_date,
    true_verbatim_locality: item.true_verbatim_locality,
  };
  render(item);
}

document.getElementById('text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') save();
});

document.getElementById('confidence').addEventListener('input', (e) => {
  document.getElementById('confidence-value').textContent = parseFloat(e.target.value).toFixed(1);
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

INDEX_HTML = INDEX_HTML.replace("DANISH_CHARS_PLACEHOLDER", json.dumps(DANISH_CHARS))


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


def load_train_lookup(path):
    """Map image_file -> {verbatimDate, verbatimLocality} from train.csv, for
    displaying the known field transcription as a labeling guide. {} if the
    file doesn't exist."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {
            row["image_file"]: {
                "verbatimDate": row["verbatimDate"],
                "verbatimLocality": row["verbatimLocality"],
            }
            for row in csv.DictReader(f)
        }


TRAIN_LOOKUP = load_train_lookup(TRAIN_CSV)


def next_unlabeled(candidates, labels, skipped):
    """Candidate filename not yet in labels or skipped with the lowest doctr
    confidence (least reliable, most worth a human look first), or None if done."""
    remaining = [f for f in candidates if f not in labels and f not in skipped]
    if not remaining:
        return None
    return min(remaining, key=lambda f: candidates[f]["doctr_confidence"])


def is_valid_confidence(value):
    """True if value is a number in [0, 1] (bool excluded, since bool is an int subclass)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


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
        """Route GET / (the page), GET /api/state, GET /api/item, GET /crop/<filename>,
        and GET /original/<filename> (the full source photo the crop was taken from)."""
        parsed = urlsplit(self.path)

        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/state":
            candidates = _load_json(CANDIDATES_PATH, {})
            labels = _load_json(LABELS_PATH, {})
            skipped = _load_json(SKIPPED_PATH, [])
            needs_help = _load_json(NEEDS_HELP_PATH, [])
            filename = next_unlabeled(candidates, labels, skipped)
            candidate = candidates.get(filename, {})
            ground_truth = TRAIN_LOOKUP.get(candidate.get("source"), {})
            self._send_json({
                "filename": filename,
                "source": candidate.get("source"),
                "doctr_text": candidate.get("doctr_text"),
                "doctr_confidence": candidate.get("doctr_confidence"),
                "true_verbatim_date": ground_truth.get("verbatimDate"),
                "true_verbatim_locality": ground_truth.get("verbatimLocality"),
                "labeled": len(labels),
                "skipped": len(skipped),
                "needs_help": len(needs_help),
                "total": len(candidates),
            })
            return

        if parsed.path == "/api/item":
            query = parse_qs(parsed.query)
            filename = query.get("filename", [None])[0]
            labels = _load_json(LABELS_PATH, {})
            confidences = _load_json(CONFIDENCES_PATH, {})
            candidates = _load_json(CANDIDATES_PATH, {})
            candidate = candidates.get(filename, {})
            ground_truth = TRAIN_LOOKUP.get(candidate.get("source"), {})
            self._send_json({
                "filename": filename,
                "text": labels.get(filename),
                "confidence": confidences.get(filename),
                "source": candidate.get("source"),
                "doctr_text": candidate.get("doctr_text"),
                "doctr_confidence": candidate.get("doctr_confidence"),
                "true_verbatim_date": ground_truth.get("verbatimDate"),
                "true_verbatim_locality": ground_truth.get("verbatimLocality"),
            })
            return

        if self.path.startswith("/crop/"):
            filename = self.path[len("/crop/"):]
            self._serve_file(IMAGES_DIR, filename, "image/png")
            return

        if self.path.startswith("/original/"):
            filename = self.path[len("/original/"):]
            self._serve_file(ORIGINAL_IMAGES_DIR, filename, "image/jpeg")
            return

        self.send_response(404)
        self.end_headers()

    def _serve_file(self, directory, filename, content_type):
        """Serve filename from directory, guarding against escaping it via '..' or absolute paths."""
        file_path = os.path.join(directory, filename)
        same_dir = os.path.dirname(os.path.abspath(file_path)) == os.path.abspath(directory)
        if not same_dir or not os.path.isfile(file_path):
            self.send_response(404)
            self.end_headers()
            return
        with open(file_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
            confidence = payload.get("confidence", DEFAULT_CONFIDENCE)
            if not is_valid_confidence(confidence):
                self._send_json({"error": "confidence must be a number between 0 and 1"}, status=400)
                return
            labels = _load_json(LABELS_PATH, {})
            labels[filename] = text
            _save_json(LABELS_PATH, labels)
            confidences = _load_json(CONFIDENCES_PATH, {})
            confidences[filename] = confidence
            _save_json(CONFIDENCES_PATH, confidences)
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
