#!/usr/bin/env python3
"""Pre-flight check for nc-vision-agent.

Validates that .system/ (the nexus.shared library, copied in from the
monorepo at deploy/runtime - see CLAUDE.md) is present and version-compatible
before the agent's own imports are attempted. Does not copy anything itself.

Usage: python install.py [--selftest]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent
_SYSTEM_SRC = _ROOT / ".system" / "src"
_TOOLS_DIR = _ROOT / "src" / "nc_vision_agent" / "tools"
_PROMPT_DIR = _ROOT / "src" / "nc_vision_agent" / "prompts"

# Keep in lockstep with nexus's own pyproject.toml `version =` - bump when
# this agent starts depending on a newer nexus.shared feature.
MIN_NEXUS_VERSION = (0, 1, 0)

# run.py always subprocesses this script; tools/*.py always load these
# prompt files (classify_images.py's _load_step_prompts() reads .exists()
# and silently substitutes "" per-file - a missing prompt fails LATE, mid-LLM-call,
# not here) - so check both here instead of leaving them to surface as
# scattered ImportError/FileNotFoundError.
_REQUIRED_FILES = [
    _TOOLS_DIR / "classify_images.py",   # run.py's subprocess target
    _TOOLS_DIR / "extract_text.py",
    _TOOLS_DIR / "backfill_short_drafts.py",
    _PROMPT_DIR / "classify-step1-type.txt",
    _PROMPT_DIR / "classify-step2-visual.txt",
    _PROMPT_DIR / "classify-step3-pf2e-character.txt",
    _PROMPT_DIR / "classify-step3-pf2e-environment.txt",
    _PROMPT_DIR / "classify-step4-description.txt",
]


def _agent_version() -> str:
    sys.path.insert(0, str(_ROOT / "src"))
    from nc_vision_agent import __version__
    return __version__


def _dist_info_version(system_src: Path) -> Optional[str]:
    """Read Version: out of nexus's PKG-INFO/METADATA directly - works even
    when system/ was raw-copied rather than pip-installed at destination."""
    for pattern in ("nexus*.egg-info/PKG-INFO", "nexus*.dist-info/METADATA"):
        for meta in system_src.glob(pattern):
            for line in meta.read_text(encoding="utf-8").splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    return None


def _parse_version(v: str) -> tuple[int, ...]:
    parts = []
    for chunk in v.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _selftest() -> None:
    assert _parse_version("0.1.0") == (0, 1, 0)
    assert _parse_version("1.2") == (1, 2)
    assert _parse_version("0.10.0") > _parse_version("0.9.9")
    assert _parse_version("0.0.9") < MIN_NEXUS_VERSION
    print("selftest OK")


def main() -> int:
    print(f"nc-vision-agent v{_agent_version()}")

    problems: list[str] = []
    warnings: list[str] = []

    for f in _REQUIRED_FILES:
        if not f.exists():
            problems.append(f"Missing required file: {f.relative_to(_ROOT)}")

    if not _SYSTEM_SRC.exists():
        problems.append(
            f"Missing {_SYSTEM_SRC} - copy the nexus shared library's system/ "
            "folder into the repo root as .system/ before running the agent."
        )
    else:
        sys.path.insert(0, str(_SYSTEM_SRC))
        try:
            import nexus.shared  # noqa: F401
        except Exception as exc:
            problems.append(f"system/src/nexus is present but failed to import: {exc!r}")
        else:
            version = _dist_info_version(_SYSTEM_SRC)
            if version is None:
                warnings.append(
                    "nexus version unknown (no egg-info/dist-info under system/src) "
                    "- skipping compatibility check"
                )
            elif _parse_version(version) < MIN_NEXUS_VERSION:
                problems.append(
                    f"nexus {version} is older than the minimum required "
                    f"{'.'.join(map(str, MIN_NEXUS_VERSION))}"
                )
            else:
                print(f"nexus v{version} OK")

    for w in warnings:
        print(f"WARNING: {w}")
    for p in problems:
        print(f"ERROR: {p}")

    return 1 if problems else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        _selftest()
        sys.exit(0)
    sys.exit(main())
