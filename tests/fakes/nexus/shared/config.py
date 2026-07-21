from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMEndpointConfig:
    url: str
    model: str
    type: str
    provider: str
