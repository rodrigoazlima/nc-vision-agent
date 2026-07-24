"""nc_vision_agent.tools.backfill_short_drafts

One-time migration: enrich existing short draft files in 01-Processing/ that were
written by the old minimal template. Reads classification data from processed-images.json
and rewrites the body using the new type-specific rich templates from classify_images.py.

Run once:
    python -m nc_vision_agent.tools.backfill_short_drafts

Safe to re-run: files are only updated if body_lines < 15.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

_TOOLS_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _TOOLS_DIR.parents[2]  # repo root (parent of src/)

_LOG_FILE = _PROJECT_ROOT / "state" / "logs" / "backfill-short-drafts-debug.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(_LOG_FILE, encoding="utf-8")],
)
log = logging.getLogger("nc-vision-backfill")

# nexus.shared is copied into .system/ at deploy/runtime - see install.py.
_SYSTEM_SRC = _PROJECT_ROOT / ".system" / "src"
if str(_SYSTEM_SRC) not in sys.path:
    sys.path.insert(0, str(_SYSTEM_SRC))

from nexus.shared import FrontmatterIO  # noqa: E402
from nexus.shared.models import Element, Environment, ImageType, VisionClassification  # noqa: E402

# Import the rich body builders from classify_images (own package, not nexus.shared)
from nc_vision_agent.tools.classify_images import (  # noqa: E402
    _battlemap_body,
    _portrait_body,
    _scene_body,
    _token_body,
)

_VAULT_ROOT  = _PROJECT_ROOT / ".knowledge-base"
_PROCESSING  = _VAULT_ROOT / "01-Processing"
_AGENT_STATE = _PROJECT_ROOT / "state"
_PROC_IMAGES = _AGENT_STATE / "processed-images.json"

SHORT_LINE_THRESHOLD = 15


def _count_body_lines(body: str) -> int:
    n = sum(1 for line in body.splitlines() if line.strip())
    log.debug("_count_body_lines: %d non-blank line(s)", n)
    return n


def _load_state() -> dict[str, Any]:
    if not _PROC_IMAGES.exists():
        log.debug("_load_state: %s missing - returning empty state", _PROC_IMAGES)
        return {"images": {}, "pathIndex": {}}
    state = json.loads(_PROC_IMAGES.read_text(encoding="utf-8"))
    log.debug("_load_state: loaded %d image entr(y/ies) from %s", len(state.get("images", {})), _PROC_IMAGES)
    return state


def _clf_from_entry(entry: dict) -> VisionClassification:
    log.debug("_clf_from_entry: type=%s ancestry=%s class=%s", entry.get("type"), entry.get("ancestry"), entry.get("class"))
    return VisionClassification(
        type          = ImageType(entry.get("type", "body")),
        ancestry      = entry.get("ancestry", "none"),
        **{"class":   entry.get("class", "none")},
        creature_type = entry.get("creature_type", "none"),
        element       = Element(entry.get("element", "none")),
        environment   = Environment(entry.get("environment", "none")),
        description   = entry.get("description", ""),
    )


def _build_body(clf: VisionClassification) -> str:
    t = clf.type.value
    log.debug("_build_body: dispatching on type=%s", t)
    if t == "battlemap":
        return _battlemap_body(clf)
    if t == "scene":
        return _scene_body(clf)
    if t == "token":
        return _token_body(clf)
    return _portrait_body(clf)


def main() -> None:
    log.debug("main: starting, processing dir=%s", _PROCESSING)
    fio   = FrontmatterIO()
    state = _load_state()

    sha_to_entry: dict[str, dict] = state.get("images", {})
    path_index:   dict[str, str]  = state.get("pathIndex", {})
    log.debug("main: %d sha entr(y/ies), %d pathIndex entr(y/ies)", len(sha_to_entry), len(path_index))

    # Build sha → entry lookup (skip failed entries)
    sha_map: dict[str, dict] = {
        sha: entry
        for sha, entry in sha_to_entry.items()
        if not sha.startswith("path:") and entry.get("status") == "ok"
    }
    log.debug("main: %d usable (status=ok) sha entr(y/ies)", len(sha_map))

    updated = 0
    skipped = 0
    missing = 0

    md_paths = sorted(_PROCESSING.glob("*.md"))
    log.debug("main: scanning %d draft file(s)", len(md_paths))
    for md_path in md_paths:
        log.debug("main: examining %s", md_path.name)
        try:
            fm, body = fio.read(md_path)
        except Exception as exc:
            log.debug("main: %s parse error - %r", md_path.name, exc)
            print(f"  SKIP {md_path.name}: parse error - {exc}")
            skipped += 1
            continue

        body_lines = _count_body_lines(body)
        log.debug("main: %s has %d body line(s) (threshold=%d)", md_path.name, body_lines, SHORT_LINE_THRESHOLD)
        if body_lines >= SHORT_LINE_THRESHOLD:
            continue  # already rich enough

        sha256: Optional[str] = fm.get("sha256")
        if not sha256:
            log.debug("main: %s has no sha256 in frontmatter", md_path.name)
            print(f"  SKIP {md_path.name}: no sha256 in frontmatter")
            missing += 1
            continue

        entry = sha_map.get(sha256)
        if not entry:
            log.debug("main: %s sha256=%s not found in state", md_path.name, sha256[:12])
            print(f"  SKIP {md_path.name}: sha256 {sha256[:12]}… not in state")
            missing += 1
            continue

        try:
            clf      = _clf_from_entry(entry)
            new_body = _build_body(clf)
        except Exception as exc:
            log.debug("main: %s body build failed - %r", md_path.name, exc)
            print(f"  SKIP {md_path.name}: body build failed - {exc}")
            skipped += 1
            continue

        fm["updated"] = date.today().isoformat()

        # Add environment tag if missing
        env = clf.environment.value
        if env and env != "none" and env not in fm.get("tags", []):
            log.debug("main: %s adding missing environment tag %r", md_path.name, env)
            fm.setdefault("tags", []).append(env)

        fio.write(md_path, fm, new_body)
        new_lines = _count_body_lines(new_body)
        log.debug("main: %s written, %d -> %d body lines", md_path.name, body_lines, new_lines)
        print(f"  UPDATED {md_path.name}: {body_lines} → {new_lines} body lines")
        updated += 1

    log.debug("main: done - updated=%d skipped=%d missing=%d", updated, skipped, missing)
    print(f"\nDone: {updated} updated, {skipped} skipped, {missing} missing state")


if __name__ == "__main__":
    main()
