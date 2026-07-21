from __future__ import annotations

from nc_vision_agent.tools import backfill_short_drafts as bsd


def test_count_body_lines_ignores_blank_lines():
    body = "\n## Heading\n\nSome text\n\n\n- bullet\n"
    assert bsd._count_body_lines(body) == 3


def test_count_body_lines_empty_body():
    assert bsd._count_body_lines("") == 0
    assert bsd._count_body_lines("\n\n\n") == 0


def test_clf_from_entry_builds_classification():
    entry = {
        "type": "portrait",
        "ancestry": "human",
        "class": "fighter",
        "creature_type": "none",
        "element": "fire",
        "environment": "none",
        "description": "A fighter.",
    }
    clf = bsd._clf_from_entry(entry)
    assert clf.type.value == "portrait"
    assert clf.ancestry == "human"
    assert clf.char_class == "fighter"
    assert clf.element.value == "fire"


def test_clf_from_entry_defaults_missing_fields():
    clf = bsd._clf_from_entry({})
    assert clf.type.value == "body"  # entry.get("type", "body") default
    assert clf.ancestry == "none"


def test_build_body_dispatches_by_type():
    battlemap_clf = bsd._clf_from_entry({"type": "battlemap", "environment": "cave"})
    assert "## Tactical Notes" in bsd._build_body(battlemap_clf)

    scene_clf = bsd._clf_from_entry({"type": "scene", "environment": "forest"})
    assert "## Story Hooks" in bsd._build_body(scene_clf)

    token_clf = bsd._clf_from_entry({"type": "token", "creature_type": "dragon"})
    assert "## VTT Usage" in bsd._build_body(token_clf)

    portrait_clf = bsd._clf_from_entry({"type": "portrait"})
    assert "Pending NPC sheet generation" in bsd._build_body(portrait_clf)


def test_main_updates_short_drafts_and_skips_rich_ones(tmp_path, monkeypatch):
    processing = tmp_path / "01-Processing"
    processing.mkdir()
    monkeypatch.setattr(bsd, "_PROCESSING", processing)

    state_file = tmp_path / "processed-images.json"
    monkeypatch.setattr(bsd, "_PROC_IMAGES", state_file)
    import json
    state_file.write_text(json.dumps({
        "images": {
            "sha_short": {
                "status": "ok", "type": "portrait", "ancestry": "human",
                "class": "fighter", "creature_type": "none", "element": "fire",
                "environment": "none", "description": "A fighter.",
            },
        },
        "pathIndex": {},
    }), encoding="utf-8")

    io_ = bsd.FrontmatterIO()
    io_.write(processing / "short.md", {"sha256": "sha_short", "tags": ["portrait"]}, "\nshort body\n")
    io_.write(processing / "no-sha.md", {"tags": []}, "\nshort body\n")
    rich_body = "\n".join(f"line {i}" for i in range(20))
    io_.write(processing / "rich.md", {"sha256": "sha_short"}, rich_body)

    bsd.main()

    fm, body = io_.read(processing / "short.md")
    assert "Pending NPC sheet generation" in body
    # environment is "none" for this entry, so main() has nothing to append -
    # only a non-"none" environment value gets added to tags.
    assert fm["tags"] == ["portrait"]

    _, rich_body_after = io_.read(processing / "rich.md")
    assert rich_body_after == rich_body  # untouched - already rich enough
