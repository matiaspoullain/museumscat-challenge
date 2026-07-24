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
