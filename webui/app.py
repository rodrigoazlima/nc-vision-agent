"""Minimal local web UI for nc-vision-agent.

Upload an image, run the agent (run.py) against it, show the result. Stdlib
only - no new dependency for one form + one endpoint.

Usage: python webui/app.py   (serves http://127.0.0.1:8765)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

_ROOT        = Path(__file__).resolve().parent.parent
_INBOX       = _ROOT / ".knowledge-base" / "00-Inbox" / "webui-uploads"
_PROCESSING  = _ROOT / ".knowledge-base" / "01-Processing"
_STATE_FILE  = _ROOT / "state" / "processed-images.json"
_RUN_SCRIPT  = _ROOT / "run.py"
_CLASSIFY_SCRIPT = _ROOT / "src" / "nc_vision_agent" / "tools" / "classify_images.py"
_EXTRACT_SCRIPT  = _ROOT / "src" / "nc_vision_agent" / "tools" / "extract_text.py"
_STATIC_DIR  = Path(__file__).resolve().parent / "static"
_LOG_FILE    = _ROOT / "state" / "logs" / "webui.log"  # same state/logs/ convention as classify_images.py's Logger
_PORT        = 8765

_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(_LOG_FILE, encoding="utf-8")],
)
log = logging.getLogger("nc-vision-webui")

# run.py's classify_images.py reads/writes state/processed-images.json with
# no locking of its own - two concurrent runs (e.g. two browser tabs, or a
# retry while the first request is still in flight) race on that file and
# corrupt each other's entries. ThreadingHTTPServer gives each request its
# own thread, so serialize agent runs here instead.
_RUN_LOCK = threading.Lock()


def _unique_path(directory: Path, filename: str) -> Path:
    """Same-name upload twice → suffix -1, -2, ... instead of clobbering."""
    directory.mkdir(parents=True, exist_ok=True)
    # Browser file names are normally bare names, but do not let a crafted API
    # request escape the upload container.
    candidate = directory / (Path(filename).name or "upload")
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


def _run_command(command: list[str], *, input_text: Optional[str] = None) -> tuple[int, str, str]:
    """Run one agent action and retain its stdout/stderr for the webview."""
    log.debug("agent subprocess: launching %s", command)
    t0 = time.monotonic()
    proc = subprocess.run(
        command, cwd=_ROOT, input=input_text, capture_output=True, text=True, timeout=900,
    )
    log.info("agent subprocess: exit=%s elapsed=%.1fs", proc.returncode, time.monotonic() - t0)
    if proc.stderr.strip():
        log.debug("agent stderr tail: %s", proc.stderr.strip()[-1000:])
    return proc.returncode, proc.stdout, proc.stderr


def _run_agent(*args: str) -> tuple[int, str, str]:
    return _run_command([sys.executable, str(_RUN_SCRIPT), *args])


def _save_upload(filename: str, raw: bytes, state: Optional[dict] = None) -> tuple[str, Path, str]:
    """Save a browser-selected image in its stable agent container."""
    sha = hashlib.sha256(raw).hexdigest()
    state = state if state is not None else (
        json.loads(_STATE_FILE.read_text(encoding="utf-8")) if _STATE_FILE.exists() else {}
    )
    container = _container_name(sha, state)
    dest = _unique_path(_INBOX / container, filename)
    dest.write_bytes(raw)
    log.debug("saved webview upload to %s", dest.relative_to(_ROOT))
    return sha, dest, container


def _run_tool(module: str, name: str, args: dict) -> tuple[int, str, str]:
    """Invoke an agent's public `call_tool` action without a shell command.

    The JSON is sent over stdin so an image path never becomes executable code.
    """
    program = (
        "import json, sys; "
        "from importlib import import_module; "
        "request = json.load(sys.stdin); "
        "print(import_module(request['module']).call_tool(request['name'], request['args'], {}))"
    )
    request = json.dumps({"module": module, "name": name, "args": args})
    return _run_command([sys.executable, "-c", program], input_text=request)


_ACTIONS = {
    "list_pending_images": {
        "label": "List pending images", "module": "nc_vision_agent.tools.classify_images", "tool": "list_pending_images",
    },
    "detect_token": {
        "label": "Detect token", "module": "nc_vision_agent.tools.classify_images", "tool": "detect_token", "needs_image": True,
    },
    "match_token_face": {
        "label": "Match token face", "module": "nc_vision_agent.tools.classify_images", "tool": "match_token_face", "needs_image": True,
    },
    "classify_image": {
        "label": "Classify image (preview)", "module": "nc_vision_agent.tools.classify_images", "tool": "classify_image", "needs_image": True,
    },
    "run_classification_batch": {"label": "Run classification batch", "command": [sys.executable, str(_RUN_SCRIPT)]},
    "retry_failed": {"label": "Retry failed classifications", "command": [sys.executable, str(_RUN_SCRIPT), "--retry-failed"]},
    "extract_image_text": {
        "label": "Extract image text (preview)", "module": "nc_vision_agent.tools.extract_text", "tool": "extract_image_text", "needs_image": True,
    },
    "run_text_extraction_batch": {"label": "Run text extraction batch", "command": [sys.executable, str(_EXTRACT_SCRIPT)]},
    "backfill_short_drafts": {"label": "Backfill short drafts", "command": [sys.executable, "-m", "nc_vision_agent.tools.backfill_short_drafts"]},
}


def _perform_action(action: str, filename: Optional[str] = None, raw: Optional[bytes] = None) -> dict:
    spec = _ACTIONS.get(action)
    if spec is None:
        return {"ok": False, "error": f"Unknown action: {action}"}

    image_path: Optional[Path] = None
    container: Optional[str] = None
    if spec.get("needs_image"):
        if not filename or raw is None:
            return {"ok": False, "error": "Choose an image before running this action."}
        _sha, image_path, container = _save_upload(filename, raw)

    try:
        if "command" in spec:
            rc, stdout, stderr = _run_command(spec["command"])
        else:
            rc, stdout, stderr = _run_tool(
                spec["module"], spec["tool"], {"image_path": str(image_path)} if image_path else {}
            )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Action timed out after 15 minutes."}
    except Exception as exc:
        log.exception("action %s failed", action)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    output = (stdout + ("\n" if stdout and stderr else "") + stderr).strip()
    return {
        "ok": rc == 0,
        "action": action,
        "label": spec["label"],
        "container": container,
        "imagePath": str(image_path.relative_to(_ROOT)) if image_path else None,
        "output": output or "Action completed with no output.",
        "returncode": rc,
    }


def _classify(filename: str, raw: bytes) -> dict:
    sha = hashlib.sha256(raw).hexdigest()
    log.info("classify: filename=%r size=%dB sha=%s", filename, len(raw), sha[:12])

    if _RUN_LOCK.locked():
        log.info("classify: waiting on run-lock (another classification is in progress)")

    with _RUN_LOCK:
        log.debug("classify: run-lock acquired")
        pre_state = json.loads(_STATE_FILE.read_text(encoding="utf-8")) if _STATE_FILE.exists() else {}
        _saved_sha, dest, container = _save_upload(filename, raw, pre_state)

        _returncode, stdout, stderr = _run_agent()
        log_tail = (stdout + "\n" + stderr).strip()[-4000:]

        state = json.loads(_STATE_FILE.read_text(encoding="utf-8")) if _STATE_FILE.exists() else {}
    log.debug("classify: run-lock released")

    entry = (state.get("images") or {}).get(sha)
    if entry is None:
        log.warning("classify: no state entry for sha=%s after run - reporting failure", sha[:12])
        return {
            "ok": False,
            "error": "Agent produced no result for this image (LLM offline, "
                     "batch skipped it, or classification failed - see log).",
            "log": log_tail,
            "container": container,
        }

    draft = _find_draft_by_sha(sha)
    log.info(
        "classify: matched sha=%s type=%s entity_type=%s draft=%s",
        sha[:12], entry.get("type"), entry.get("entity_type"), draft.name if draft else None,
    )
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
    def log_message(self, fmt, *args):  # route stdlib's own access log through ours
        log.debug("http: %s - %s", self.address_string(), fmt % args)

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
        if self.path not in ("/api/classify", "/api/actions"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/classify":
                result = _classify(payload["filename"], base64.b64decode(payload["data"], validate=True))
            else:
                raw = base64.b64decode(payload["data"], validate=True) if payload.get("data") else None
                result = _perform_action(payload.get("action", ""), payload.get("filename"), raw)
            self._send_json(result)
        except Exception as exc:
            log.exception("do_POST: unhandled error")
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", _PORT), Handler)
    log.info("nc-vision-agent web UI: http://127.0.0.1:%d/", _PORT)
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
