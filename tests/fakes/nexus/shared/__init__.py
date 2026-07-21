"""Minimal stand-in for the monorepo's nexus.shared library.

nc-vision-agent depends on nexus.shared but doesn't vendor it - normally it's
copied into .system/src at deploy time (see install.py). That copy isn't
present in this dev/test environment, so this fake package provides just the
surface classify_images.py / extract_text.py / backfill_short_drafts.py
import, with real behavior where it's cheap (slugify, hashing, YAML
frontmatter) and dumb stubs where tests inject their own fakes instead
(LLMClient - see conftest.FakeLLMClient).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import yaml
from slugify import slugify

from .models import Element, Environment, ImageType, VisionClassification

__all__ = [
    "FrontmatterIO",
    "LLMClient",
    "Logger",
    "LLMOfflineError",
    "LLMResponseError",
    "SignalEmitter",
    "VisionClassification",
    "build_entity_slug",
    "image_tag",
    "locked_update_queue_entry",
    "sha256_of_file",
    "to_slug",
]


class LLMOfflineError(Exception):
    pass


class LLMResponseError(Exception):
    pass


def to_slug(text: str) -> str:
    return slugify(text or "")


def build_entity_slug(entity_type: str, descriptor: str) -> str:
    parts = [p for p in (to_slug(entity_type), to_slug(descriptor)) if p]
    return "-".join(parts)


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_tag(*, sha256: Optional[str] = None, uuid: Optional[str] = None, path: Optional[str] = None) -> str:
    bits = []
    if sha256:
        bits.append(f"sha256={sha256[:12]}")
    if uuid:
        bits.append(f"uuid={uuid}")
    if path:
        bits.append(f"path={path}")
    return f" [{' '.join(bits)}]" if bits else ""


def locked_update_queue_entry(queue_file: Path, rel: str, fn) -> None:
    data = json.loads(Path(queue_file).read_text(encoding="utf-8")) if Path(queue_file).exists() else {}
    entry = data.get(rel, {})
    new_key = fn(entry)
    data[rel] = entry
    if new_key:
        data[new_key] = data.pop(rel)
    Path(queue_file).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


class FrontmatterIO:
    """Real (if simplified) YAML-frontmatter read/write - `---\\nYAML\\n---\\nbody`."""

    def read(self, path: Path) -> tuple[dict, str]:
        text = Path(path).read_text(encoding="utf-8")
        if not text.startswith("---"):
            return {}, text
        _, _, rest = text.partition("---\n")
        fm_text, _, body = rest.partition("\n---\n")
        fm = yaml.safe_load(fm_text) or {}
        return fm, body

    def write(self, path: Path, frontmatter: dict, body: str) -> None:
        fm_text = yaml.safe_dump(frontmatter, sort_keys=False, default_flow_style=False, allow_unicode=True)
        Path(path).write_text(f"---\n{fm_text}---\n{body}", encoding="utf-8")


class Logger:
    def __init__(self, *, task_id: str, script_basename: str, logs_dir: Path, master_log: Path) -> None:
        self.task_id = task_id
        self.script_basename = script_basename
        self.logs_dir = logs_dir
        self.master_log = master_log

    def start(self) -> float:
        import time
        return time.time()

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass

    def done(self, t0: float, *, key: str, count: int, failed: int) -> None:
        pass


class SignalEmitter:
    def __init__(self, signals_dir: Path) -> None:
        self.signals_dir = signals_dir

    def emit(self, *, signal_type: str, emitter: str, ref: str) -> None:
        pass


from .llm_client import LLMClient  # noqa: E402
