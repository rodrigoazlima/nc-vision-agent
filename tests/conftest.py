"""nc-vision-agent depends on nexus.shared, which is normally copied into
.system/src at deploy time (see repo CLAUDE.md / install.py) and isn't
vendored here. For tests we point sys.path at tests/fakes/ instead, which
provides a minimal stand-in package (see tests/fakes/nexus/shared/) - real
behavior where it's cheap (slugify, YAML frontmatter, hashing), dumb stubs
where tests inject their own fake (LLMClient, via FakeLLMClient below).

This must run before nc_vision_agent.tools.* is imported anywhere, which is
why it lives at module scope in conftest.py rather than in a fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Optional

_FAKES_DIR = Path(__file__).resolve().parent / "fakes"
if str(_FAKES_DIR) not in sys.path:
    sys.path.insert(0, str(_FAKES_DIR))

import pytest


class FakeLLMClient:
    """Duck-typed stand-in for nexus.shared.LLMClient. classify_image_full()
    and friends only ever call .chat(messages, max_tokens=...) and
    .is_available() on the client they're given - Python doesn't enforce the
    LLMClient type hint at runtime, so this is a legal substitute.

    `responses` is a list of either a JSON string or an Exception instance,
    consumed one per .chat() call in order (so a test can script a required
    step failing twice then succeeding, etc).
    """

    def __init__(self, responses: list[Any], available: bool = True) -> None:
        self.responses = list(responses)
        self.available = available
        self.calls: list[list[dict]] = []

    def is_available(self) -> bool:
        return self.available

    def chat(self, messages: list[dict], max_tokens: int = 1024) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_llm_client() -> Callable[..., FakeLLMClient]:
    return FakeLLMClient
