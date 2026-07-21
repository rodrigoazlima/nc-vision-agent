from __future__ import annotations

from typing import Any, Optional

SELF_MANAGEMENT_TOOLS: list[dict] = []


def call_self_management_tool(name: str, args: dict, context: dict, *, module_file, task_id) -> Optional[str]:
    """Stub: never handles a tool, always defers to the caller's own dispatch."""
    return None
