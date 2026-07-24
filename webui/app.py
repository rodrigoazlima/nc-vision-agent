"""Minimal local web UI for nc-vision-agent.

Upload an image, run the agent (run.py) against it, show the result. Stdlib
only - no new dependency for one form + one endpoint.

Usage: python webui/app.py   (serves http://127.0.0.1:8765)
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

_ROOT        = Path(__file__).resolve().parent.parent
_INBOX       = _ROOT / ".knowledge-base" / "00-Inbox" / "webui-uploads"
_PROCESSING  = _ROOT / ".knowledge-base" / "01-Processing"
_STATE_FILE  = _ROOT / "state" / "processed-images.json"
_RUN_SCRIPT  = _ROOT / "run.py"
_STATIC_DIR  = Path(__file__).resolve().parent / "static"
_PORT        = 8765

# run.py's classify_images.py reads/writes state/processed-images.json with
# no locking of its own - two concurrent runs (e.g. two browser tabs, or a
# retry while the first request is still in flight) race on that file and
# corrupt each other's entries. ThreadingHTTPServer gives each request its
# own thread, so serialize agent runs here instead.
_RUN_LOCK = threading.Lock()


def _unique_path(directory: Path, filename: str) -> Path:
    """Same-name upload twice → suffix -1, -2, ... instead of clobbering."""
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    stem, suffix = candidate.stem, candidate.suffix
    n = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{n}{suffix}"
        n += 1
    return candidate


def _container_name(sha: str, state: dict) -> str:
    """agent-vision-{code}: code is the item's existing uuid (already-classified
    image, re-uploaded) or the first 8 hex digits of its sha256 otherwise -
    same uuid-reuse convention classify_images.py's main() applies when
    writing state, so a re-upload lands in the same container as before."""
    entry = (state.get("images") or {}).get(sha) or {}
    code = entry.get("uuid") or sha[:8]
    return f"agent-vision-{code}"


def _find_draft_by_sha(sha: str) -> Optional[Path]:
    """Match the drafted note back to this upload via its sha256 frontmatter
    field - the note's filename (entity slug) has no fixed relation to the
    source filename, but _write_draft always records sha256 when known."""
    if not _PROCESSING.is_dir():
        return None
    for md in _PROCESSING.glob("*.md"):
        if sha in md.read_text(encoding="utf-8", errors="ignore"):
            return md
    return None


def _run_agent() -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(_RUN_SCRIPT)],
        cwd=_ROOT, capture_output=True, text=True, timeout=900,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _classify(filename: str, raw: bytes) -> dict:
    sha = hashlib.sha256(raw).hexdigest()

    with _RUN_LOCK:
        pre_state = json.loads(_STATE_FILE.read_text(encoding="utf-8")) if _STATE_FILE.exists() else {}
        container = _container_name(sha, pre_state)
        _unique_path(_INBOX / container, filename).write_bytes(raw)

        _returncode, stdout, stderr = _run_agent()
        log_tail = (stdout + "\n" + stderr).strip()[-4000:]

        state = json.loads(_STATE_FILE.read_text(encoding="utf-8")) if _STATE_FILE.exists() else {}

    entry = (state.get("images") or {}).get(sha)
    if entry is None:
        return {
            "ok": False,
            "error": "Agent produced no result for this image (LLM offline, "
                     "batch skipped it, or classification failed - see log).",
            "log": log_tail,
            "container": container,
        }

    draft = _find_draft_by_sha(sha)
    return {
        "ok": True,
        "container": container,
        "type": entry.get("type"),
        "entity_type": entry.get("entity_type"),
        "ancestry": entry.get("ancestry"),
        "class": entry.get("class"),
        "creature_type": entry.get("creature_type"),
        "element": entry.get("element"),
        "environment": entry.get("environment"),
        "description": entry.get("description"),
        "tags": entry.get("candidate_tags"),
        "renamedTo": entry.get("originalName"),
        "path": entry.get("path"),
        "draft": draft.read_text(encoding="utf-8") if draft else None,
        "draftName": draft.name if draft else None,
        "log": log_tail,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter default access log
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = (_STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/classify":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            result = _classify(payload["filename"], base64.b64decode(payload["data"]))
            self._send_json(result)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", _PORT), Handler)
    print(f"nc-vision-agent web UI: http://127.0.0.1:{_PORT}/")
    server.serve_forever()


def _selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "a.png").write_bytes(b"x")
        p1 = _unique_path(d, "a.png")
        assert p1.name == "a-1.png", p1
        (d / "a-1.png").write_bytes(b"x")
        assert _unique_path(d, "a.png").name == "a-2.png"

        assert _container_name("abcdef1234", {}) == "agent-vision-abcdef12"
        assert _container_name("abcdef1234", {"images": {"abcdef1234": {"uuid": "u-1"}}}) == "agent-vision-u-1"

        md = d / "note.md"
        md.write_text("---\nsha256: deadbeef\n---\nbody", encoding="utf-8")
        global _PROCESSING
        old = _PROCESSING
        _PROCESSING = d
        try:
            assert _find_draft_by_sha("deadbeef") == md
            assert _find_draft_by_sha("nope") is None
        finally:
            _PROCESSING = old
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        _selftest()
    else:
        main()
