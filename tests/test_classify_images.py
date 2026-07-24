from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from nc_vision_agent.tools import classify_images as ci
from nc_vision_agent.tools.classify_images import (
    LLMOfflineError,
    LLMResponseError,
    VisionClassification,
)


def _clf(**overrides) -> VisionClassification:
    data = {
        "type": "portrait",
        "ancestry": "none",
        "class": "none",
        "creature_type": "none",
        "element": "none",
        "environment": "none",
        "description": "",
        "candidate_tags": [],
        "entity_type": "none",
    }
    data.update(overrides)
    return VisionClassification.model_validate(data)


# ---------------------------------------------------------------------------
# _is_concrete_tag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag,expected", [
    ("sword", True),
    ("stylized sword", True),
    ("a b c d e f", True),  # exactly 6 words
    ("a b c d e f g", False),  # 7 words - too long
    ("", False),
    ("   ", False),
])
def test_is_concrete_tag(tag, expected):
    assert ci._is_concrete_tag(tag) is expected


# ---------------------------------------------------------------------------
# _extract_candidate_tags
# ---------------------------------------------------------------------------

def test_extract_candidate_tags_flattens_and_dedupes():
    raw = {
        "visual_analysis": {
            "equipment": {"weapons": ["Sword", "sword", "  Bow  "], "shield": ["Round Shield"]},
            "clothing": {"materials": ["leather"]},
            "fantasy_features": {"wings": ["feathered wings"]},
            "environment_details": {"architecture": ["stone arch"]},
        }
    }
    tags = ci._extract_candidate_tags(raw)
    assert tags == ["sword", "bow", "round shield", "leather", "feathered wings", "stone arch"]


def test_extract_candidate_tags_rejects_sentence_length():
    raw = {
        "visual_analysis": {
            "equipment": {"weapons": [
                "a very long sentence describing an elaborate weapon in far too many words to be a tag"
            ]},
        }
    }
    assert ci._extract_candidate_tags(raw) == []


def test_extract_candidate_tags_handles_dict_items():
    raw = {"visual_analysis": {"equipment": {"other": [{"type": "amulet"}, {"name": "ring"}, {}]}}}
    assert ci._extract_candidate_tags(raw) == ["amulet", "ring"]


def test_extract_candidate_tags_degrades_on_bad_shape():
    assert ci._extract_candidate_tags({"visual_analysis": "not a dict"}) == []
    assert ci._extract_candidate_tags({}) == []


# ---------------------------------------------------------------------------
# _is_object_in_scene_bucket / slug builders
# ---------------------------------------------------------------------------

def test_object_in_scene_bucket_true_for_item_or_artifact():
    assert ci._is_object_in_scene_bucket(_clf(type="scene", entity_type="item"))
    assert ci._is_object_in_scene_bucket(_clf(type="battlemap", entity_type="artifact"))


def test_object_in_scene_bucket_false_for_creature_or_npc():
    # A monster/character genuinely standing in an environment shot is still
    # a scene, not a misclassified object photo.
    assert not ci._is_object_in_scene_bucket(_clf(type="scene", entity_type="creature"))
    assert not ci._is_object_in_scene_bucket(_clf(type="scene", entity_type="npc"))


def test_object_in_scene_bucket_false_for_portrait():
    assert not ci._is_object_in_scene_bucket(_clf(type="portrait", entity_type="item"))


def test_object_slug_descriptor_prefers_first_slugifiable_tag():
    clf = _clf(type="scene", entity_type="item", candidate_tags=["a sentence too long to slug nicely at all here"],
               environment="dungeon")
    # the sentence-length tag still slugifies to *something* via to_slug, so
    # it wins over falling back to environment - descriptor picks the first
    # tag that survives to_slug, not the first *concrete* tag.
    assert ci._object_slug_descriptor(clf)


def test_object_slug_descriptor_falls_back_to_environment():
    clf = _clf(type="scene", entity_type="item", candidate_tags=[], environment="dungeon")
    assert ci._object_slug_descriptor(clf) == "dungeon"


def test_object_slug_descriptor_none_when_nothing_available():
    clf = _clf(type="scene", entity_type="item", candidate_tags=[], environment="none")
    assert ci._object_slug_descriptor(clf) is None


def test_image_filename_slug_portrait():
    clf = _clf(type="portrait", ancestry="human", **{"class": "fighter"}, element="fire")
    assert ci._image_filename_slug(clf) == "human-fighter-fire.portrait"


def test_image_filename_slug_portrait_creature_fallback():
    clf = _clf(type="body", ancestry="none", creature_type="dragon", element="none")
    assert ci._image_filename_slug(clf) == "dragon.body"


def test_image_filename_slug_portrait_unknown_when_no_parts():
    clf = _clf(type="portrait")
    assert ci._image_filename_slug(clf) == "unknown.portrait"


def test_image_filename_slug_scene_environment():
    clf = _clf(type="scene", environment="forest")
    assert ci._image_filename_slug(clf) == "scene-forest"


def test_image_filename_slug_scene_unknown_environment():
    clf = _clf(type="scene", environment="none")
    assert ci._image_filename_slug(clf) == "scene-unknown"


def test_image_filename_slug_object_in_scene_bucket():
    clf = _clf(type="scene", entity_type="item", candidate_tags=["Rusty Sword"], environment="dungeon")
    assert ci._image_filename_slug(clf) == "item-rusty-sword"


def test_entity_slug_matches_filename_slug_shape():
    clf = _clf(type="portrait", ancestry="elf", **{"class": "wizard"})
    assert ci._entity_slug(clf) == "portrait-elf-wizard"

    clf2 = _clf(type="battlemap", environment="cave")
    assert ci._entity_slug(clf2) == "battlemap-cave"

    clf3 = _clf(type="scene", entity_type="artifact", candidate_tags=["ancient tome"])
    assert ci._entity_slug(clf3) == "artifact-ancient-tome"


# ---------------------------------------------------------------------------
# Path resolution (collision bumping)
# ---------------------------------------------------------------------------

def test_resolve_image_target_no_collision(tmp_path):
    src = tmp_path / "raw.png"
    src.write_bytes(b"x")
    target = ci._resolve_image_target(src, "human-fighter-fire.portrait")
    assert target == tmp_path / "human-fighter-fire.portrait.png"


def test_resolve_image_target_bumps_on_collision(tmp_path):
    src = tmp_path / "raw.png"
    src.write_bytes(b"x")
    (tmp_path / "slug.png").write_bytes(b"taken")
    target = ci._resolve_image_target(src, "slug")
    assert target == tmp_path / "slug-01.png"


def test_resolve_image_target_same_path_is_a_noop(tmp_path):
    src = tmp_path / "slug.png"
    src.write_bytes(b"x")
    assert ci._resolve_image_target(src, "slug") == src


def test_resolve_entity_path_bumps_on_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_PROCESSING", tmp_path)
    (tmp_path / "human-fighter-fire.md").write_text("x", encoding="utf-8")
    out = ci._resolve_entity_path("human-fighter-fire")
    assert out == tmp_path / "human-fighter-fire-01.md"


# ---------------------------------------------------------------------------
# retry_failed_images - pure dict logic, no filesystem
# ---------------------------------------------------------------------------

def test_retry_failed_images_clears_only_failed_entries():
    state = {
        "images": {
            "path:bad.png": {"status": "failed"},
            "abc123": {"status": "ok"},
        },
        "pathIndex": {
            "bad.png": "path:bad.png",
            "good.png": "abc123",
        },
    }
    cleared = ci.retry_failed_images(state)
    assert cleared == 1
    assert "bad.png" not in state["pathIndex"]
    assert "path:bad.png" not in state["images"]
    assert state["pathIndex"]["good.png"] == "abc123"
    assert state["images"]["abc123"]["status"] == "ok"


def test_retry_failed_images_noop_when_nothing_failed():
    state = {"images": {"abc": {"status": "ok"}}, "pathIndex": {"good.png": "abc"}}
    assert ci.retry_failed_images(state) == 0
    assert state["pathIndex"] == {"good.png": "abc"}


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------

def test_load_state_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_PROC_IMAGES", tmp_path / "processed-images.json")
    state = ci._load_state()
    assert state == {"version": 2, "images": {}, "pathIndex": {}}


def test_save_state_then_load_state_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_AGENT_STATE", tmp_path)
    monkeypatch.setattr(ci, "_PROC_IMAGES", tmp_path / "processed-images.json")
    state = {"version": 2, "images": {"a": {"status": "ok"}}, "pathIndex": {"p": "a"}}
    ci._save_state(state)
    assert ci._load_state() == state


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------

def _setup_inbox(tmp_path, monkeypatch, files):
    inbox = tmp_path / "00-Inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    for rel in files:
        p = inbox / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake")
    monkeypatch.setattr(ci, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ci, "_INBOX_IMAGES", inbox)
    monkeypatch.setattr(ci, "_GEN_TOKENS", tmp_path / "nonexistent-generated-tokens.json")
    return inbox


def test_candidate_images_skips_already_processed(tmp_path, monkeypatch):
    _setup_inbox(tmp_path, monkeypatch, ["a.png", "b.jpg"])
    state = {"pathIndex": {"00-Inbox/a.png": "somesha"}}
    queue: dict = {}
    result = ci._candidate_images(state, queue)
    assert [p.name for p in result] == ["b.jpg"]


def test_candidate_images_skips_done_in_queue(tmp_path, monkeypatch):
    _setup_inbox(tmp_path, monkeypatch, ["a.png"])
    state = {"pathIndex": {}}
    queue = {"00-Inbox/a.png": {"agents": {"vision": "done"}}}
    assert ci._candidate_images(state, queue) == []


def test_candidate_images_non_png_before_png(tmp_path, monkeypatch):
    _setup_inbox(tmp_path, monkeypatch, ["z.png", "a.jpg"])
    result = ci._candidate_images({"pathIndex": {}}, {})
    assert [p.suffix for p in result] == [".jpg", ".png"]


def test_candidate_images_excludes_generated_tokens(tmp_path, monkeypatch):
    inbox = _setup_inbox(tmp_path, monkeypatch, ["a.png", "b.png"])
    gen_file = tmp_path / "generated-tokens.json"
    gen_file.write_text(json.dumps({"x": {"tokenPath": "00-Inbox/a.png"}}), encoding="utf-8")
    monkeypatch.setattr(ci, "_GEN_TOKENS", gen_file)
    result = ci._candidate_images({"pathIndex": {}}, {})
    assert [p.name for p in result] == ["b.png"]


def test_candidate_images_excludes_token_filename_convention(tmp_path, monkeypatch):
    _setup_inbox(tmp_path, monkeypatch, ["hero-token.png", "hero.png"])
    result = ci._candidate_images({"pathIndex": {}}, {})
    assert [p.name for p in result] == ["hero.png"]


def test_candidate_images_skips_duplicate_content_at_new_path(tmp_path, monkeypatch):
    # a.jpg is already classified (its sha recorded under state["images"]);
    # b.jpg is a byte-identical re-upload at a different path (e.g. a repeat
    # webui "classify" click) and must not be reclassified from scratch.
    _setup_inbox(tmp_path, monkeypatch, ["a.jpg", "b.jpg"])
    sha = ci._sha256(tmp_path / "00-Inbox" / "a.jpg")
    state = {
        "pathIndex": {"00-Inbox/a.jpg": sha},
        "images": {sha: {"status": "ok"}},
    }
    result = ci._candidate_images(state, {})
    assert result == []


def test_is_token_file_matches_suffix_and_infix():
    assert ci._is_token_file(Path("hero-token.png"))
    assert ci._is_token_file(Path("hero.token.png"))
    assert not ci._is_token_file(Path("hero.png"))


# ---------------------------------------------------------------------------
# _get_folder_candidates
# ---------------------------------------------------------------------------

def test_get_folder_candidates_filters_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_PROJECT_ROOT", tmp_path)
    folder = tmp_path / "00-Inbox"
    folder.mkdir()
    good = folder / "good.png"
    good.write_bytes(b"x")
    bad_type = folder / "tok.png"
    bad_type.write_bytes(b"x")
    other_folder = tmp_path / "other"
    other_folder.mkdir()
    elsewhere = other_folder / "elsewhere.png"
    elsewhere.write_bytes(b"x")

    state = {
        "pathIndex": {
            "00-Inbox/good.png": "sha_good",
            "00-Inbox/tok.png": "sha_tok",
            "other/elsewhere.png": "sha_else",
            "00-Inbox/missing.png": "path:00-Inbox/missing.png",  # failed
        },
        "images": {
            "sha_good": {"status": "ok", "type": "portrait"},
            "sha_tok": {"status": "ok", "type": "token"},
            "sha_else": {"status": "ok", "type": "portrait"},
        },
    }
    result = ci._get_folder_candidates(folder, state)
    assert result == [good]


# ---------------------------------------------------------------------------
# _is_token (real PIL images)
# ---------------------------------------------------------------------------

def _make_png(path: Path, corner_alphas):
    img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
    img.putpixel((0, 0), (0, 0, 0, corner_alphas[0]))
    img.putpixel((9, 0), (0, 0, 0, corner_alphas[1]))
    img.putpixel((0, 9), (0, 0, 0, corner_alphas[2]))
    img.putpixel((9, 9), (0, 0, 0, corner_alphas[3]))
    img.save(path)


def test_is_token_true_with_two_transparent_corners(tmp_path):
    p = tmp_path / "t.png"
    _make_png(p, [0, 0, 255, 255])
    assert ci._is_token(p) is True


def test_is_token_false_with_one_transparent_corner(tmp_path):
    p = tmp_path / "t.png"
    _make_png(p, [0, 255, 255, 255])
    assert ci._is_token(p) is False


def test_is_token_false_for_non_png(tmp_path):
    p = tmp_path / "t.jpg"
    Image.new("RGB", (10, 10)).save(p)
    assert ci._is_token(p) is False


def test_is_token_false_on_corrupt_file(tmp_path):
    p = tmp_path / "t.png"
    p.write_bytes(b"not a real png")
    assert ci._is_token(p) is False


# ---------------------------------------------------------------------------
# _try_face_match
# ---------------------------------------------------------------------------

def _make_face_png(path: Path, seed: int):
    """Random per-pixel noise, not a flat color: a uniform-color image is a
    constant vector for any shade, so cosine similarity would be ~1.0
    between *any* two solid-color images regardless of "who matches whom" -
    useless for distinguishing candidates. Independent noise seeds give
    near-zero similarity for unrelated images and exactly 1.0 for the same
    seed, which is what the match/no-match assertions below actually need."""
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    Image.fromarray(arr, mode="L").convert("RGBA").save(path)


def test_try_face_match_finds_identical_candidate(tmp_path):
    token = tmp_path / "token.png"
    match = tmp_path / "match.png"
    other = tmp_path / "other.png"
    _make_face_png(token, seed=1)
    _make_face_png(match, seed=1)  # same seed as token - identical image
    _make_face_png(other, seed=2)  # different noise - unrelated
    result = ci._try_face_match(token, [other, match])
    assert result == match


def test_try_face_match_none_when_no_candidates(tmp_path):
    token = tmp_path / "token.png"
    _make_face_png(token, seed=1)
    assert ci._try_face_match(token, []) is None


def test_try_face_match_none_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_FACE_SIMILARITY_THRESHOLD", 1.01)  # unreachable
    token = tmp_path / "token.png"
    match = tmp_path / "match.png"
    _make_face_png(token, seed=1)
    _make_face_png(match, seed=1)
    assert ci._try_face_match(token, [match]) is None


# ---------------------------------------------------------------------------
# Body builders - spot-check structure, not full prose
# ---------------------------------------------------------------------------

def test_battlemap_body_contains_expected_sections():
    clf = _clf(type="battlemap", environment="cave", element="fire", description="A dank cave.")
    body = ci._battlemap_body(clf)
    for heading in ("## Description", "## Atmosphere", "## Tactical Notes", "## Encounter Hooks", "## Details"):
        assert heading in body
    assert "A dank cave." in body


def test_scene_body_uses_environment_specific_hooks():
    clf = _clf(type="scene", environment="forest", description="desc")
    body = ci._scene_body(clf)
    assert "arrow" in body.lower()  # forest-specific hook text


def test_scene_body_falls_back_to_default_hooks_for_unknown_environment():
    clf = _clf(type="scene", environment="abyss", description="desc")
    body = ci._scene_body(clf)
    assert "pivotal moment" in body


def test_token_body_uses_creature_role_hints():
    clf = _clf(type="token", creature_type="dragon", description="desc")
    body = ci._token_body(clf)
    assert "Boss encounter" in body


def test_token_body_default_hints_when_no_ancestry_or_creature():
    clf = _clf(type="token", description="desc")
    body = ci._token_body(clf)
    assert "Unnamed encounter participant" in body


def test_portrait_body_is_minimal():
    clf = _clf(type="portrait", ancestry="human", description="desc")
    body = ci._portrait_body(clf)
    assert "Pending NPC sheet generation" in body


def test_item_body_excludes_type_and_environment_tags():
    clf = _clf(type="scene", entity_type="item", environment="dungeon",
               candidate_tags=["scene", "dungeon", "rusty sword", "iron shield"],
               description="An old blade.")
    body = ci._item_body(clf, "item")
    assert "rusty sword" in body
    assert "iron shield" in body
    # type/environment values are filtered out of the visual-details bullets
    assert "- scene\n" not in body
    assert "- dungeon\n" not in body


# ---------------------------------------------------------------------------
# _atmosphere_lines
# ---------------------------------------------------------------------------

def test_atmosphere_lines_includes_lighting_mood_sound_when_known():
    clf = _clf(environment="cave", element="dark")
    lines = ci._atmosphere_lines(clf)
    assert "**Setting**: Cave environment" in lines
    assert "**Lighting**" in lines
    assert "**Mood**" in lines
    assert "**Sounds**" in lines


def test_atmosphere_lines_minimal_when_none_known():
    clf = _clf(environment="none", element="none")
    lines = ci._atmosphere_lines(clf)
    assert lines == "- **Setting**: Unknown environment"


# ---------------------------------------------------------------------------
# _write_draft
# ---------------------------------------------------------------------------

def test_write_draft_portrait_frontmatter_and_body(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_PROJECT_ROOT", tmp_path)
    image_path = tmp_path / "00-Inbox" / "human-fighter-fire.portrait.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"x")
    out_path = tmp_path / "01-Processing" / "portrait-human-fighter-fire.md"
    out_path.parent.mkdir(parents=True)

    clf = _clf(type="portrait", ancestry="human", **{"class": "fighter"}, element="fire",
               environment="none", description="A stoic human fighter.",
               candidate_tags=["scarred", "longsword"])

    entity_type = ci._write_draft(out_path, clf, image_path, sha256="deadbeef")

    assert entity_type == "npc"  # portrait/body/token default when entity_type is "none"
    fm, body = ci.FrontmatterIO().read(out_path)
    assert fm["type"] == "npc"
    assert fm["status"] == "draft"
    assert fm["quality"] == 0
    assert fm["reviewed"] is False
    assert fm["sha256"] == "deadbeef"
    assert "human" in fm["tags"] and "fire" in fm["tags"] and "scarred" in fm["tags"]
    assert fm["source"] == ["00-Inbox/human-fighter-fire.portrait.png"]
    assert "Pending NPC sheet generation" in body


def test_write_draft_records_original_source_when_renamed(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_PROJECT_ROOT", tmp_path)
    orig = tmp_path / "00-Inbox" / "IMG_0001.png"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"x")
    renamed = tmp_path / "00-Inbox" / "human-fighter-fire.portrait.png"
    renamed.write_bytes(b"x")
    out_path = tmp_path / "portrait.md"

    clf = _clf(type="portrait", ancestry="human", description="d")
    ci._write_draft(out_path, clf, renamed, original_image_path=orig)

    fm, _ = ci.FrontmatterIO().read(out_path)
    assert fm["originalSource"] == ["00-Inbox/IMG_0001.png"]


def test_write_draft_object_in_scene_bucket_uses_item_body(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_PROJECT_ROOT", tmp_path)
    image_path = tmp_path / "item.png"
    image_path.write_bytes(b"x")
    out_path = tmp_path / "item.md"
    clf = _clf(type="scene", entity_type="item", environment="dungeon",
               candidate_tags=["rusty sword"], description="An old blade.")

    entity_type = ci._write_draft(out_path, clf, image_path)
    assert entity_type == "item"
    fm, body = ci.FrontmatterIO().read(out_path)
    assert fm["type"] == "item"
    assert "Detected as **item**" in body


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

def test_parse_json_response_plain():
    assert ci._parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_strips_json_fence():
    raw = '```json\n{"a": 1}\n```'
    assert ci._parse_json_response(raw) == {"a": 1}


def test_parse_json_response_strips_bare_fence():
    raw = '```\n{"a": 1}\n```'
    assert ci._parse_json_response(raw) == {"a": 1}


# ---------------------------------------------------------------------------
# Cycle prompt builders
# ---------------------------------------------------------------------------

def test_cycle2_prompt_asks_for_remaining_count(monkeypatch):
    monkeypatch.setattr(ci, "_MIN_TAGS_TARGET", 6)
    prompt = ci._cycle2_prompt(["a", "b"])
    assert "aim for at least 4 more" in prompt


def test_cycle2_prompt_no_remaining_clause_once_target_met(monkeypatch):
    monkeypatch.setattr(ci, "_MIN_TAGS_TARGET", 2)
    prompt = ci._cycle2_prompt(["a", "b", "c"])
    assert "aim for at least" not in prompt


def test_cycle3_prompt_includes_entity_type_guidance():
    prompt = ci._cycle3_prompt(["a"])
    assert "CHARACTERS & NPCS" in prompt


def test_cycle4_prompt_includes_known_tags_and_hint():
    prompt = ci._cycle4_prompt(["sword"], ["blade", "shield"], "item")
    assert "blade, shield" in prompt
    assert "current best guess: item" in prompt


# ---------------------------------------------------------------------------
# _run_required_step
# ---------------------------------------------------------------------------

def test_run_required_step_succeeds_first_try(fake_llm_client):
    client = fake_llm_client(['{"type": "portrait"}'])
    messages = [{"role": "system", "content": "sys"}]
    result = ci._run_required_step(messages, client, "do step 1", 256, validate=ci._validate_step1)
    assert result == {"type": "portrait"}
    assert len(client.calls) == 1


def test_run_required_step_retries_then_succeeds(monkeypatch, fake_llm_client):
    monkeypatch.setattr(ci, "_STEP_RETRY_BACKOFF_S", 0)
    client = fake_llm_client(["not json at all", '{"type": "scene"}'])
    messages = [{"role": "system", "content": "sys"}]
    result = ci._run_required_step(messages, client, "do step 1", 256, validate=ci._validate_step1)
    assert result == {"type": "scene"}
    assert len(client.calls) == 2


def test_run_required_step_raises_after_exhausting_retries(monkeypatch, fake_llm_client):
    monkeypatch.setattr(ci, "_STEP_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(ci, "_STEP_MAX_RETRIES", 1)
    client = fake_llm_client(["bad json", "still bad"])
    messages = [{"role": "system", "content": "sys"}]
    with pytest.raises(LLMResponseError):
        ci._run_required_step(messages, client, "do step 1", 256, validate=ci._validate_step1)
    assert len(client.calls) == 2  # initial + 1 retry


def test_run_required_step_propagates_offline_error_without_retry(fake_llm_client):
    client = fake_llm_client([LLMOfflineError("offline")])
    messages = [{"role": "system", "content": "sys"}]
    with pytest.raises(LLMOfflineError):
        ci._run_required_step(messages, client, "do step 1", 256)
    assert len(client.calls) == 1  # no retry attempted


def test_run_required_step_respects_message_budget(monkeypatch, fake_llm_client):
    monkeypatch.setattr(ci, "_MAX_CONVERSATION_MESSAGES", 2)
    client = fake_llm_client(['{"type": "portrait"}'])
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "x"}]
    with pytest.raises(LLMResponseError, match="message budget"):
        ci._run_required_step(messages, client, "do step 1", 256)
    assert len(client.calls) == 0


# ---------------------------------------------------------------------------
# refine_tags_with_library
# ---------------------------------------------------------------------------

def test_refine_tags_with_library_merges_never_replaces(tmp_path, fake_llm_client):
    img = tmp_path / "img.png"
    Image.new("RGBA", (4, 4)).save(img)
    client = fake_llm_client(['{"final_tags": ["blade"], "entity_type": "item"}'])
    library = {"tags": {}}
    final_tags, entity_type = ci.refine_tags_with_library(
        img, client, ["sword", "shield"], "none", library,
    )
    # "blade" merged in, "sword"/"shield" survive even though the model's
    # "final_tags" didn't repeat them - see docstring: never a replace.
    assert final_tags == ["sword", "shield", "blade"]
    assert entity_type == "item"


def test_refine_tags_with_library_degrades_on_bad_response(tmp_path, fake_llm_client):
    img = tmp_path / "img.png"
    Image.new("RGBA", (4, 4)).save(img)
    client = fake_llm_client(["not json"])
    final_tags, entity_type = ci.refine_tags_with_library(
        img, client, ["sword"], "item", {"tags": {}},
    )
    assert final_tags == ["sword"]
    assert entity_type == "item"


def test_refine_tags_with_library_propagates_offline_error(tmp_path, fake_llm_client):
    img = tmp_path / "img.png"
    Image.new("RGBA", (4, 4)).save(img)
    client = fake_llm_client([LLMOfflineError("offline")])
    with pytest.raises(LLMOfflineError):
        ci.refine_tags_with_library(img, client, ["sword"], "item", {"tags": {}})


def test_refine_tags_with_library_stops_at_message_budget(monkeypatch, tmp_path, fake_llm_client):
    monkeypatch.setattr(ci, "_MAX_CONVERSATION_MESSAGES", 1)
    img = tmp_path / "img.png"
    Image.new("RGBA", (4, 4)).save(img)
    client = fake_llm_client(['{"final_tags": ["blade"]}'])
    history = [{"role": "system", "content": "sys"}]
    final_tags, entity_type = ci.refine_tags_with_library(
        img, client, ["sword"], "item", {"tags": {}}, history=history,
    )
    assert final_tags == ["sword"]
    assert len(client.calls) == 0


# ---------------------------------------------------------------------------
# classify_image_full - full happy-path conversation
# ---------------------------------------------------------------------------

def test_classify_image_full_character_branch(tmp_path, fake_llm_client):
    img = tmp_path / "img.png"
    Image.new("RGBA", (4, 4)).save(img)
    client = fake_llm_client([
        '{"type": "portrait"}',                                    # step 1
        '{"visual_analysis": {"equipment": {"weapons": ["sword"]}}}',  # step 2
        '{"ancestry": "elf", "class": "wizard", "creature_type": "none"}',  # step 3 (character)
        '{"description": "An elven wizard."}',                      # step 4
        '{"category": "portrait", "additional_tags": ["robe"]}',    # cycle 2
        '{"entity_type": "npc", "additional_tags": []}',            # cycle 3
        '{"final_tags": ["sword", "robe"], "entity_type": "npc"}',  # cycle 4
    ])
    clf = ci.classify_image_full(img, client, ci._load_step_prompts(), is_tk=False)
    assert clf.type.value == "portrait"
    assert clf.ancestry == "elf"
    assert clf.char_class == "wizard"
    assert clf.entity_type == "npc"
    assert "sword" in clf.candidate_tags and "robe" in clf.candidate_tags


def test_classify_image_full_environment_branch(tmp_path, fake_llm_client):
    img = tmp_path / "img.png"
    Image.new("RGBA", (4, 4)).save(img)
    client = fake_llm_client([
        '{"type": "battlemap"}',                                   # step 1
        '{"visual_analysis": {}}',                                 # step 2
        '{"environment": "cave", "element": "fire"}',              # step 3 (environment)
        '{"description": "A fiery cave."}',                        # step 4
        '{"category": "battlemap", "additional_tags": []}',        # cycle 2
        '{"entity_type": "location", "additional_tags": []}',      # cycle 3
        '{"final_tags": [], "entity_type": "location"}',           # cycle 4
    ])
    clf = ci.classify_image_full(img, client, ci._load_step_prompts(), is_tk=False)
    assert clf.type.value == "battlemap"
    assert clf.environment.value == "cave"
    assert clf.element.value == "fire"
    assert clf.ancestry == "none"  # character fields stay "none" on this branch


def test_classify_image_full_forces_token_type_when_is_tk(tmp_path, fake_llm_client):
    img = tmp_path / "img.png"
    Image.new("RGBA", (4, 4)).save(img)
    client = fake_llm_client([
        '{"type": "portrait"}',
        '{"visual_analysis": {}}',
        '{"ancestry": "human", "class": "none", "creature_type": "none"}',
        '{"description": "d"}',
        '{"category": "body", "additional_tags": []}',  # cycle 2 tries to change type...
        '{"entity_type": "npc", "additional_tags": []}',
        '{"final_tags": [], "entity_type": "npc"}',
    ])
    clf = ci.classify_image_full(img, client, ci._load_step_prompts(), is_tk=True)
    # ...but is_tk overrides it - a token stays a token regardless of cycle 2.
    assert clf.type.value == "token"


def test_classify_image_full_raises_when_required_step_never_parses(monkeypatch, tmp_path, fake_llm_client):
    monkeypatch.setattr(ci, "_STEP_RETRY_BACKOFF_S", 0)
    img = tmp_path / "img.png"
    Image.new("RGBA", (4, 4)).save(img)
    client = fake_llm_client(["nope"] * (ci._STEP_MAX_RETRIES + 1))
    with pytest.raises(LLMResponseError):
        ci.classify_image_full(img, client, ci._load_step_prompts(), is_tk=False)
