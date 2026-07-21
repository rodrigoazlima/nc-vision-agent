"""vision.tools.extract_text

Extracts any readable text visible in classified images (signage, banners,
scrolls, engravings, letters) via Qwen3-VL and appends a "## Text on Image"
section to the matching draft entity in 01-Processing/.

Runs only against images already classified by classify_images.py (reads
processed-images.json). No LLM available → skip gracefully. Batch: 10 per run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

_TOOLS_DIR    = Path(__file__).resolve().parent
_AGENTS_DIR   = _TOOLS_DIR.parents[1]
_PROJECT_ROOT = _AGENTS_DIR.parent

if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from nexus.shared import FrontmatterIO, LLMClient, Logger, LLMOfflineError, image_tag  # noqa: E402
from nexus.shared.config import LLMEndpointConfig  # noqa: E402
from nexus.shared.loaders import load_llm_endpoint  # noqa: E402

TASK_ID         = "vision-agent"
SCRIPT_BASENAME = "extract_text.py"
BATCH_SIZE      = 10

_VAULT_ROOT   = _PROJECT_ROOT / ".knowledge-base"
_PROCESSING   = _VAULT_ROOT / "01-Processing"
_AGENT_STATE  = _AGENTS_DIR / "vision" / "state"
_LOGS_DIR     = _AGENT_STATE / "logs"
_MASTER_LOG   = _AGENTS_DIR / "runtime" / "state" / "logs" / "automation.log"
_PROC_IMAGES  = _AGENT_STATE / "processed-images.json"
_TEXT_STATE   = _AGENT_STATE / "text-extractions.json"

_LLM_CFG = load_llm_endpoint(
    "vision_llm",
    fallback = LLMEndpointConfig(
        url      = "http://localhost:1234/v1/chat/completions",
        model    = "qwen3-vl-4b-instruct",
        type     = "vision",
        provider = "lmstudio",
    ),
    agent_dir    = _AGENTS_DIR / "vision",
    task_id      = TASK_ID,
    project_root = _PROJECT_ROOT,
)

_PROMPT = (
    "Look for any readable text rendered in this image (signage, banners, scrolls, "
    "engravings, letters, tattoos, book pages). Return ONLY valid JSON: "
    '{"has_text": true|false, "text": "verbatim text found, or empty string"}'
)
_SYSTEM = "You are an OCR assistant for a Dungeon Master's campaign vault. Return ONLY valid JSON."


# ---------------------------------------------------------------------------
# State I/O (atomic writes per G8)
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    _AGENT_STATE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------

def _candidate_images(processed: dict, done: dict) -> list[tuple[str, str]]:
    """Return (sha256, rel_path) pairs for classified images not yet text-extracted."""
    out: list[tuple[str, str]] = []
    for sha, entry in processed.get("images", {}).items():
        if entry.get("status") != "ok":
            continue
        if sha in done:
            continue
        rel = entry.get("path")
        if rel and (_PROJECT_ROOT / rel).exists():
            out.append((sha, rel))
    return sorted(out, key=lambda t: t[1])


def _find_draft(rel_img: str) -> Optional[Path]:
    """Locate the draft .md in 01-Processing/ whose frontmatter `source` lists rel_img."""
    io_ = FrontmatterIO()
    for path in _PROCESSING.glob("*.md"):
        fm, _ = io_.read(path)
        if rel_img in (fm.get("source") or []):
            return path
    return None


# ---------------------------------------------------------------------------
# Draft update
# ---------------------------------------------------------------------------

def _append_text_section(path: Path, text: str) -> None:
    io_ = FrontmatterIO()
    fm, body = io_.read(path)
    section = f"\n## Text on Image\n\n{text}\n"
    io_.write(path, fm, body.rstrip("\n") + "\n" + section)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log = Logger(
        task_id=TASK_ID,
        script_basename=SCRIPT_BASENAME,
        logs_dir=_LOGS_DIR,
        master_log=_MASTER_LOG,
    )
    t0 = log.start()

    client = LLMClient(_LLM_CFG)
    if not client.is_available():
        log.warning("Qwen3-VL (localhost:1234) offline - skipping batch")
        log.done(t0, key="extracted", count=0, failed=0)
        sys.exit(0)

    processed = _load_state(_PROC_IMAGES)
    done      = _load_state(_TEXT_STATE)
    candidates = _candidate_images(processed, done)

    if not candidates:
        log.info("No pending images for text extraction")
        log.done(t0, key="extracted", count=0, failed=0)
        sys.exit(0)

    batch  = candidates[:BATCH_SIZE]
    count  = 0
    failed = 0
    log.info(f"Batch: {len(batch)} of {len(candidates)} image(s)")

    for sha, rel in batch:
        img_path = _PROJECT_ROOT / rel
        tag = image_tag(sha256=sha, path=rel)
        try:
            result = client.vision_chat(img_path, _PROMPT, system=_SYSTEM, max_tokens=400)
        except LLMOfflineError:
            log.warning(f"LLM offline while processing {img_path.name} - aborting batch{tag}")
            break
        except Exception as exc:
            log.error(f"Text extraction failed for {img_path.name}: {exc}{tag}")
            failed += 1
            continue

        text = (result.get("text") or "").strip() if result.get("has_text") else ""
        if text:
            draft = _find_draft(rel)
            if draft is not None:
                _append_text_section(draft, text)
                log.info(f"Text found in {img_path.name} → appended to {draft.name}{tag}")
            else:
                log.warning(f"Text found in {img_path.name} but no matching draft in 01-Processing/{tag}")
        else:
            log.info(f"No text found in {img_path.name}{tag}")

        done[sha] = {"path": rel, "hasText": bool(text)}
        _save_state(_TEXT_STATE, done)
        count += 1

    log.done(t0, key="extracted", count=count, failed=failed)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Agentic tool interface (claude-api dispatch)
# ---------------------------------------------------------------------------

from nexus.shared.agent_tools import SELF_MANAGEMENT_TOOLS, call_self_management_tool  # noqa: E402

_MODULE_FILE = Path(__file__)

TOOLS = SELF_MANAGEMENT_TOOLS + [
    {
        "name": "extract_image_text",
        "description": (
            "Run OCR-style text extraction via the local vision LLM on a single image. "
            "Returns {has_text, text}. Does not modify any draft file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Absolute path to the image file"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "run_batch",
        "description": (
            "Extract text from up to BATCH_SIZE pending classified images and append "
            "a '## Text on Image' section to their matching draft in 01-Processing/."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def call_tool(name: str, args: dict, context: dict) -> str:
    import io
    import contextlib

    result = call_self_management_tool(
        name, args, context, module_file=_MODULE_FILE, task_id=TASK_ID
    )
    if result is not None:
        return result

    if name == "extract_image_text":
        p = Path(args["image_path"])
        if not p.exists():
            return json.dumps({"error": f"Image not found: {p}"})
        client = LLMClient(_LLM_CFG)
        if not client.is_available():
            return json.dumps({"error": "LLM offline (localhost:1234)"})
        try:
            result = client.vision_chat(p, _PROMPT, system=_SYSTEM, max_tokens=400)
            return json.dumps(result)
        except LLMOfflineError:
            return json.dumps({"error": "LLM offline"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    if name == "run_batch":
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                main()
        except SystemExit:
            pass
        return buf.getvalue().strip() or "Batch run complete"

    raise ValueError(f"Unknown tool: {name!r}")
