# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This is **nc-vision-agent**, a standalone extraction of the Nexus Campaigns monorepo's vision agent (originally `agents/vision/`). It runs independently, but still depends on that monorepo's shared library (`nexus.shared`): not vendored here, expected at `.system/src` (see `install.py` and "Architecture within this folder" below). Full pipeline architecture (multi-agent runner, vault folder rules, metadata standard) lives in the monorepo's root `CLAUDE.md`/`AGENTS.md`, not duplicated here.

## What this agent does

Classifies RPG images dropped in `.knowledge-base/00-Inbox/` (any subfolder, not just `images/`) using a local Qwen3-VL vision model served by LM Studio, then:
1. Detects image type: `portrait` / `body` / `battlemap` / `scene` / `token`
2. Renames the image in-place to a canonical slug (`{ancestry}-{class}-{element}.{type}` for portrait/body/token, `{type}-{environment}` for battlemap/scene - except when `entity_type` says a scene/battlemap image is really a photographed object, see `_is_object_in_scene_bucket` below, in which case it's `{entity_type}-{descriptor}`)
3. Writes a draft entity `.md` to `.knowledge-base/01-Processing/` with type-specific body content (`status: draft`, `quality: 0`, `reviewed: false` - always)
4. For token PNGs (≥2 transparent corners), attempts face-match against other classified images in the same folder via cosine similarity on a center-crop grayscale vector (PIL + numpy only, no ML deps) and inherits the matched portrait's metadata instead of re-calling the LLM
5. Emits an `image-classified` signal for the downstream Lore agent and updates `system/state/inbox-queue.json`
6. `classify_image_full()` (public - importable by other agents/system code) runs the full classification per image, image held in one conversation for up to `max_conversation_messages` (default 20, configurable - see below). Split into 4 **required** steps, each its own single-purpose LLM turn (`_run_required_step`), then 3 **optional** follow-up cycles:
   - **STEP 1 - type** (`prompts/classify-step1-type.txt`) - portrait/body/battlemap/scene, camera-angle-based (straight-down = battlemap even with no grid, vs. angled/eye-level = scene).
   - **STEP 2 - visual analysis** (`prompts/classify-step2-visual.txt`) - the large nested `visual_analysis` JSON; seeds a free-form `candidate_tags` list via `_extract_candidate_tags` (equipment/clothing/fantasy_features/environment_details arrays).
   - **STEP 3 - PF2e classification** - two prompt variants (`prompts/classify-step3-pf2e-character.txt` for ancestry/class/creature_type, `prompts/classify-step3-pf2e-environment.txt` for environment/element). Which one is sent is decided **in code** from STEP 1's already-parsed `type` (`classify_image_full`'s `is_character` branch) - never asked as an LLM if/else.
   - **STEP 4 - flavor description** (`prompts/classify-step4-description.txt`) - three sentences.
   - A required step that fails to parse/validate retries up to `step_max_retries` times (corrective nudge + `step_retry_backoff_seconds` backoff) before raising and aborting just that image; `LLMOfflineError` is never retried here, it propagates immediately.
   - Then 3 optional cycles in the same conversation, degrading gracefully instead of retrying: confirming image type + more tags; inferring `entity_type` (same 18-value taxonomy as the classification agent's `_ALLOWED_TYPES`) + more tags; `refine_tags_with_library()` (also public, independently callable) aligning the running tag list against classification agent's `state/tag-library.json` (read-only here) - its `final_tags` are **merged** into the running list, never used to replace it, so a rephrase (e.g. "weapon" → "stylized sword") can't silently drop a tag an earlier cycle already grounded.

   Every tag appended at any step/cycle passes `_is_concrete_tag()` (1-6 words) first - guards against a full `visual_analysis` sentence landing in `tags:` as if it were one tag. Target: ≥`min_tags_target` tags (default 6) + both categories set. Final `candidate_tags`/`entity_type` land in the note's frontmatter via `_write_draft` - see `agents/classification/CLAUDE.md` for the library side.

   `type` (structural: portrait/body/battlemap/scene/token) and `entity_type` (semantic, 18-value taxonomy) come from separate, uncoordinated steps/cycles and can disagree - most notably, STEP 1 has no "isolated object" option, so a photographed weapon/artifact with no real environment gets forced into `scene`/`battlemap` by elimination. `_is_object_in_scene_bucket(clf)` detects this narrowly (type is scene/battlemap AND entity_type is `item` or `artifact` specifically - NOT creature/npc/anything else, since a monster or character standing in an environment shot is still genuinely a scene, not a misclassification) and both the slug builders and `_write_draft`'s body selection switch to entity_type-driven output (`_object_slug_descriptor` + `_item_body`) instead of the generic `{type}-{environment}` bucket - otherwise every unrelated item/artifact photo with no clear environment collides into the same meaningless, collision-numbered name.

A second, independent tool (`tools/extract_text.py`) runs OCR-style text extraction over images already classified above: it asks Qwen3-VL whether an image contains any readable text (signage, banners, scrolls, engravings, letters), and if so appends a `## Text on Image` section to the matching draft in `01-Processing/` (matched via the draft's `source:` frontmatter field). It never classifies, renames, or touches any other frontmatter field - body append only.

Full contract (inputs/outputs/responsibilities/restrictions) is in `AGENT.md` frontmatter - read it before changing behavior, it's the source of truth, not this file.

## Commands

```powershell
# Pre-flight: validates .system/ (nexus.shared) is present and version-compatible
python install.py

# Run this agent once, standalone
python run.py
# equivalently:
python src\nc_vision_agent\tools\classify_images.py

# One-time migration for old short-body drafts in 01-Processing/ (safe to re-run)
python -m nc_vision_agent.tools.backfill_short_drafts

# Extract text-on-image ("## Text on Image") for already-classified images
python src\nc_vision_agent\tools\extract_text.py

# Editable install (registers console_scripts: nc-vision-agent, nc-vision-agent-extract-text)
pip install -e .
```

LM Studio must be running at `http://localhost:1234/v1`. Vision model is selected via `./registry.yaml` (optional, repo root) → `llm_endpoints.vision_llm.model`, falling back to `qwen3-vl-4b-instruct` if missing/malformed. `agent.json` (optional, repo root) is unrelated - it only governs `claude-code`/`lm-studio` dispatch of the agent process itself, not this script's internal vision LLM calls. If the LLM is offline, the batch aborts silently - no image is marked failed.

## Architecture within this folder

- `src/nc_vision_agent/tools/classify_images.py` - the whole agent: state I/O, token detection, face-match, slug builders, per-type draft body generation, `main()` batch loop, the public `classify_image_full()`/`refine_tags_with_library()` multi-step/cycle classification functions, `_load_pipeline_config()` (registry.yaml/agent.json config loader), and the `TOOLS`/`call_tool()` agentic interface consumed by `agent.json`'s dispatch.
- `src/nc_vision_agent/tools/backfill_short_drafts.py` - one-off migration importing body builders from `classify_images.py`; only rewrites drafts with `body_lines < 15`.
- `src/nc_vision_agent/tools/extract_text.py` - OCR-style text extraction over already-classified images; appends `## Text on Image` to the matching draft. Own `TOOLS`/`call_tool()` agentic interface (`extract_image_text`, `run_batch`), own state file, own log.
- `src/nc_vision_agent/prompts/system.md` - role/workflow instructions for `claude-code`/agentic dispatch, covering both `classify_images.py` (tool-call sequence: `list_pending_images` → `run_batch` → `write_log`) and `extract_text.py` (`extract_image_text` / `run_batch`).
- `src/nc_vision_agent/prompts/classify-step{1..4}-*.txt` - the 5 raw prompt files sent to the vision LLM, one per `classify_image_full` required step (STEP 3 has a `-character` and `-environment` variant - see above); together they define the JSON schema and full PF2e vocabulary the model must answer with (kept in sync with the monorepo's `nexus.shared.models` `PF2E_*` constants - update both together). See `docs/agents/vision/image-classification-workflow.md` (monorepo) for the full flow.
- `state/processed-images.json` - `{version, images: {sha256: entry}, pathIndex: {relpath: sha256}}`. Failed images are keyed `path:{relpath}` in `pathIndex` (no sha), so `_get_folder_candidates` filters them out by that prefix, not by a `status` field alone. Each entry also carries `candidate_tags` (list[str], possibly empty) - the free-form brainstorm harvested from that image's `visual_analysis`.
- `state/token-links.json` - token→source-portrait face-match links, keyed by token sha256.
- `state/text-extractions.json` - `{sha256: {path, hasText}}`, tracks which classified images `extract_text.py` has already checked so they aren't re-checked every run.
- `.system/` - **not checked in** (gitignored); the monorepo's `system/` folder (containing `src/nexus/...`, the shared agent library) is copied or bind-mounted here at deploy/runtime. `install.py` validates it's present and version-compatible before any tool script's imports run. `.system/state/` also doubles as the cross-agent bus (`inbox-queue.json`, signals, `automation.log`) - see `_SHARED_STATE` in `classify_images.py`.
- `state/tag-library.json`, `./registry.yaml`, `./agent.json` - optional local overrides for state the monorepo's classification agent / registry / dispatch config would otherwise provide; all three are absent by default and every read degrades gracefully (defaults / `{"tags": {}}`) when missing.

## Conventions specific to this agent

- Batch size defaults to 10 images per run (`batch_size` - see `AGENT.md`'s Pipeline Configuration table for the full list of tunables and where they're sourced); within a batch, non-PNG files are processed before PNG (tokens are typically PNG, so this pushes them last, after their source portraits are already classified and in `state`).
- Never mark an image `status: failed` on an LLM *connection* error (offline LM Studio) - only on repeated response/parsing errors after retries. Connection errors abort the whole batch; response errors fail just that one image and continue the batch.
- Generated token PNGs from `nexus.workers.token` (tracked in `system/state/workers/token/generated-tokens.json`) are excluded from `_candidate_images` - they are derived artifacts, not new source images, and re-ingesting them causes filename-bump duplication.
- This agent inherits the vault-wide rule that only `01-Processing/` may be written for drafts; it must never touch `02-Library/`, and `reviewed`/approved `status` are human-only fields it must never set (see root `AGENTS.md` → Directory Rules / Metadata Standard).
