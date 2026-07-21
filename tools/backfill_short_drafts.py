"""vision.tools.backfill_short_drafts

One-time migration: enrich existing short draft files in 01-Processing/ that were
written by the old minimal template. Reads classification data from processed-images.json
and rewrites the body using the new type-specific rich templates from classify_images.py.

Run once:
    python agents/vision/tools/backfill_short_drafts.py

Safe to re-run: files are only updated if body_lines < 15.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

_TOOLS_DIR    = Path(__file__).resolve().parent
_AGENTS_DIR   = _TOOLS_DIR.parents[1]
_PROJECT_ROOT = _AGENTS_DIR.parent

if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from nexus.shared import FrontmatterIO  # noqa: E402
from nexus.shared.models import Element, Environment, ImageType, VisionClassification  # noqa: E402

# Import the rich body builders from classify_images
from vision.tools.classify_images import (  # noqa: E402
    _battlemap_body,
    _portrait_body,
    _scene_body,
    _token_body,
)

_VAULT_ROOT  = _PROJECT_ROOT / ".knowledge-base"
_PROCESSING  = _VAULT_ROOT / "01-Processing"
_AGENT_STATE = _AGENTS_DIR / "vision" / "state"
_PROC_IMAGES = _AGENT_STATE / "processed-images.json"

SHORT_LINE_THRESHOLD = 15


def _count_body_lines(body: str) -> int:
    return sum(1 for line in body.splitlines() if line.strip())


def _load_state() -> dict[str, Any]:
    if not _PROC_IMAGES.exists():
        return {"images": {}, "pathIndex": {}}
    return json.loads(_PROC_IMAGES.read_text(encoding="utf-8"))


def _clf_from_entry(entry: dict) -> VisionClassification:
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
    if t == "battlemap":
        return _battlemap_body(clf)
    if t == "scene":
        return _scene_body(clf)
    if t == "token":
        return _token_body(clf)
    return _portrait_body(clf)


def main() -> None:
    fio   = FrontmatterIO()
    state = _load_state()

    sha_to_entry: dict[str, dict] = state.get("images", {})
    path_index:   dict[str, str]  = state.get("pathIndex", {})

    # Build sha → entry lookup (skip failed entries)
    sha_map: dict[str, dict] = {
        sha: entry
        for sha, entry in sha_to_entry.items()
        if not sha.startswith("path:") and entry.get("status") == "ok"
    }

    updated = 0
    skipped = 0
    missing = 0

    for md_path in sorted(_PROCESSING.glob("*.md")):
        try:
            fm, body = fio.read(md_path)
        except Exception as exc:
            print(f"  SKIP {md_path.name}: parse error - {exc}")
            skipped += 1
            continue

        body_lines = _count_body_lines(body)
        if body_lines >= SHORT_LINE_THRESHOLD:
            continue  # already rich enough

        sha256: Optional[str] = fm.get("sha256")
        if not sha256:
            print(f"  SKIP {md_path.name}: no sha256 in frontmatter")
            missing += 1
            continue

        entry = sha_map.get(sha256)
        if not entry:
            print(f"  SKIP {md_path.name}: sha256 {sha256[:12]}… not in state")
            missing += 1
            continue

        try:
            clf      = _clf_from_entry(entry)
            new_body = _build_body(clf)
        except Exception as exc:
            print(f"  SKIP {md_path.name}: body build failed - {exc}")
            skipped += 1
            continue

        fm["updated"] = date.today().isoformat()

        # Add environment tag if missing
        env = clf.environment.value
        if env and env != "none" and env not in fm.get("tags", []):
            fm.setdefault("tags", []).append(env)

        fio.write(md_path, fm, new_body)
        new_lines = _count_body_lines(new_body)
        print(f"  UPDATED {md_path.name}: {body_lines} → {new_lines} body lines")
        updated += 1

    print(f"\nDone: {updated} updated, {skipped} skipped, {missing} missing state")


if __name__ == "__main__":
    main()
