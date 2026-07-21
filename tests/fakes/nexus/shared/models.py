"""Minimal stand-in for nexus.shared.models - real thing lives in the
Nexus Campaigns monorepo (not vendored here, see nc-vision-agent CLAUDE.md).
Only the members classify_images.py / backfill_short_drafts.py import."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ImageType(str, Enum):
    portrait = "portrait"
    body = "body"
    battlemap = "battlemap"
    scene = "scene"
    token = "token"


class Element(str, Enum):
    fire = "fire"
    water = "water"
    earth = "earth"
    air = "air"
    metal = "metal"
    wood = "wood"
    nature = "nature"
    dark = "dark"
    light = "light"
    void = "void"
    vitality = "vitality"
    none = "none"


class Environment(str, Enum):
    dungeon = "dungeon"
    cave = "cave"
    forest = "forest"
    city = "city"
    tavern = "tavern"
    desert = "desert"
    snow = "snow"
    sea = "sea"
    swamp = "swamp"
    ruins = "ruins"
    temple = "temple"
    castle = "castle"
    plains = "plains"
    mountain = "mountain"
    volcano = "volcano"
    underwater = "underwater"
    sky = "sky"
    astral = "astral"
    shadow = "shadow"
    abyss = "abyss"
    interior = "interior"
    exterior = "exterior"
    none = "none"


class VisionClassification(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    type: ImageType
    ancestry: str = "none"
    char_class: str = Field(default="none", alias="class")
    creature_type: str = "none"
    element: Element = Element.none
    environment: Environment = Environment.none
    description: str = ""
    candidate_tags: list[str] = Field(default_factory=list)
    entity_type: str = "none"
