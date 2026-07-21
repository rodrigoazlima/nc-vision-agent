from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import LLMEndpointConfig


def load_llm_endpoint(
    name: str,
    *,
    fallback: LLMEndpointConfig,
    agent_dir: Optional[Path] = None,
    task_id: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> LLMEndpointConfig:
    """Stub: always returns fallback (no registry.yaml/agent.json resolution -
    that precedence logic lives in the real nexus.shared, not under test here)."""
    return fallback
