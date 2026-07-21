from __future__ import annotations

from nc_vision_agent.tools import extract_text as et


def test_candidate_images_filters_status_done_and_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "exists.png").write_bytes(b"x")

    processed = {
        "images": {
            "sha_ok":      {"status": "ok", "path": "exists.png"},
            "sha_failed":  {"status": "failed", "path": "exists.png"},
            "sha_missing": {"status": "ok", "path": "gone.png"},
            "sha_done":    {"status": "ok", "path": "exists.png"},
        }
    }
    done = {"sha_done": {"path": "exists.png", "hasText": False}}

    result = et._candidate_images(processed, done)
    assert result == [("sha_ok", "exists.png")]


def test_candidate_images_sorted_by_path(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    processed = {
        "images": {
            "sha_b": {"status": "ok", "path": "b.png"},
            "sha_a": {"status": "ok", "path": "a.png"},
        }
    }
    result = et._candidate_images(processed, {})
    assert result == [("sha_a", "a.png"), ("sha_b", "b.png")]


def test_find_draft_matches_on_source_frontmatter(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "_PROCESSING", tmp_path)
    io_ = et.FrontmatterIO()
    io_.write(tmp_path / "note.md", {"source": ["00-Inbox/a.png"]}, "body")
    io_.write(tmp_path / "other.md", {"source": ["00-Inbox/b.png"]}, "body")

    found = et._find_draft("00-Inbox/a.png")
    assert found == tmp_path / "note.md"


def test_find_draft_returns_none_when_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(et, "_PROCESSING", tmp_path)
    io_ = et.FrontmatterIO()
    io_.write(tmp_path / "other.md", {"source": ["00-Inbox/b.png"]}, "body")
    assert et._find_draft("00-Inbox/a.png") is None


def test_append_text_section_adds_heading_and_preserves_body(tmp_path):
    path = tmp_path / "note.md"
    io_ = et.FrontmatterIO()
    io_.write(path, {"id": "note"}, "\n## Description\n\nSome body.\n")

    et._append_text_section(path, "Beware the depths.")

    fm, body = io_.read(path)
    assert fm["id"] == "note"
    assert "## Description" in body
    assert "## Text on Image" in body
    assert "Beware the depths." in body
    # original body content wasn't clobbered
    assert "Some body." in body
