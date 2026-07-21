from __future__ import annotations

import base64
from pathlib import Path


def _resize_and_encode(path: Path) -> str:
    """Stub: base64-encode raw bytes, no actual resize/JPEG re-encode."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


class LLMClient:
    """Stub - tests inject their own fake client into classify_image_full()
    etc. directly rather than exercising this class's HTTP behavior."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def is_available(self) -> bool:
        raise NotImplementedError("tests should inject a fake LLMClient")

    def chat(self, messages, max_tokens: int = 1024) -> str:
        raise NotImplementedError("tests should inject a fake LLMClient")

    def vision_chat(self, img_path, prompt, system=None, max_tokens: int = 400):
        raise NotImplementedError("tests should inject a fake LLMClient")
