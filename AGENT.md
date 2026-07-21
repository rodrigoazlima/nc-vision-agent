---
name: vision
purpose: >
  Classifies RPG images in 00-Inbox/images/ using Qwen3-VL vision model via LM Studio,
  via a bounded multi-step/cycle conversation (classify_image_full - up to
  max_conversation_messages, default 20: 4 required steps - image type, visual analysis,
  PF2e classification, flavor description, each retried up to step_max_retries times
  before aborting the image - then 3 optional follow-up cycles that degrade gracefully -
  image-type confirmation, entity-type inference, tag-library refinement).
  Detects image type (portrait/body/battlemap/scene/token) and entity type, renames files
  to canonical slug format, writes AGENTS.md-compliant draft entities to 01-Processing/,
  runs face-match to link tokens to their source portraits, maintains processed-images.json.
inputs:
  - vault://00-Inbox/images/**/*.{png,jpg,jpeg,webp}
  - state/processed-images.json
  - state/token-links.json
  - system/state/inbox-queue.json
  - LLM: http://localhost:1234/v1/chat/completions (Qwen3-VL)
outputs:
  - vault://01-Processing/{slug}.md (draft entity per image)
  - vault://01-Processing/Images Index.md (appended row)
  - vault://00-Inbox/images/{folder}/{slug}.{ext} (renamed image)
  - vault://01-Processing/{slug}.md (## Text on Image section appended, when text found)
  - state/processed-images.json (updated index)
  - state/token-links.json (face match links)
  - state/text-extractions.json (text-extraction index)
  - system/state/inbox-queue.json (agents.vision = done)
  - state/logs/classify_images.py_YYYY-MM-DD.log
  - state/logs/extract_text.py_YYYY-MM-DD.log
dependencies:
  - ingestion
dispatch_config: agent.json
owned_tools:
  - tools/classify_images.py
  - tools/extract_text.py
responsibilities:
  - Pre-flight check LM Studio at localhost:1234 before processing batch
  - Detect transparent-corner PNGs as tokens (≥2 of 4 corners with alpha < 128)
  - For tokens: run face-match via cosine similarity against non-token images in same folder
  - Inherit classification metadata from matched source portrait if face-match succeeds
  - Call Qwen3-VL with base64-encoded image (resize to max 1024px on longest side, JPEG 85%)
  - Validate LLM JSON response against PF2e vocabulary (see models.py PF2E_* constants)
  - Harvest visual_analysis arrays (equipment/clothing/fantasy_features/environment_details)
    into candidate_tags - free-form (never validated against a vocabulary), but every
    appended tag must pass _is_concrete_tag (1-6 words) so a full sentence can't land
    in tags: as if it were one tag
  - classify_image_full() (public - importable by other agents/system code) runs 4
    required steps, each its own single-purpose LLM turn, image held in one conversation
    throughout: (1) image type, (2) visual analysis, (3) PF2e classification - a
    character variant (ancestry/class/creature_type) or environment variant
    (environment/element), the branch picked in code from step 1's own parsed type,
    never asked as an LLM if/else, (4) flavor description. Each required step retries
    up to step_max_retries times (short corrective nudge + backoff) before giving up;
    exhausting retries raises and aborts just that image. Then 3 optional follow-up
    turns in the SAME conversation: confirm image type + more tags, infer entity type
    (18-value taxonomy, same as classification agent's) + more tags, then
    refine_tags_with_library() (also public, independently callable) aligns the running
    tag list against classification agent's state/tag-library.json (read-only here)
    before finalizing. Target: >=min_tags_target tags (default 6) + both categories set;
    an optional cycle whose response fails to parse keeps the prior cycle's values
    rather than retrying - no budget for corrective turns on those three. An
    LLMOfflineError is the exception: it propagates so main() applies its existing
    one-time model-swap retry rather than silently writing degraded output.
  - Final tags (candidate_tags) and entity_type land directly in the note's frontmatter
    via _write_draft - no longer state-only once the multi-cycle conversation finishes
  - Build target filename slug; bump existing same-named file to counter suffix (e.g. -01, -02).
    Scene/battlemap images whose entity_type resolves to item or artifact (never legitimately
    the scene itself - unlike creature/npc, which can genuinely appear within one) slug off
    entity_type + a content tag instead of {type}-{environment}, and get an item-style draft
    body - otherwise unrelated object photos collapse into one generic scene-interior-NN name
    family (see _is_object_in_scene_bucket)
  - Write AGENTS.md-compliant draft to 01-Processing/ (status: draft, quality: 0, reviewed: false)
  - Append row to Images Index.md
  - Save result to processed-images.json after each classified image (not at batch end)
  - Update inbox-queue.json agents.vision = done for each processed file
  - Emit image-classified signal to signals dir (fire-and-forget per G5)
  - Abort batch silently on connection-error (do not mark image as failed)
  - `python agents/vision/tools/classify_images.py --retry-failed` clears only
    failed `path:` state entries, then reclassifies the normal batch; successful
    entries are never reset
  - Batch: 10 images per run; process non-PNG before PNG (tokens last)
  - (extract_text.py) For each already-classified image not yet text-extracted, call
    Qwen3-VL for readable text (signage, banners, scrolls, engravings, letters)
  - (extract_text.py) If text found, find the matching draft in 01-Processing/ via
    `source:` frontmatter and append a `## Text on Image` section
  - (extract_text.py) Mark every checked image in text-extractions.json (hasText true/false)
    so it is never re-checked
restrictions:
  - Must not approve content (reviewed: false, status: draft always)
  - Must not modify 02-Library/
  - Must not mark images as failed on connection-error (only on repeated API errors)
  - Max step_max_retries (default 2) retries per required step, step_retry_backoff_seconds
    (default 3) backoff between attempts - separate from main()'s own one-time
    LLM-offline retry at the batch level
  - extract_text.py must not classify, rename, or re-write frontmatter fields - body append only
state_files:
  - state/processed-images.json
  - state/token-links.json
  - state/text-extractions.json
commit_scope:
  - .knowledge-base/00-Inbox/images
  - .knowledge-base/01-Processing
---

## Pipeline Configuration

All tunables below live in `agents/registry.yaml` → `agents.vision.options` (shared
default) and are overridable per-machine via `agents/vision/agent.json` →
`tasks.vision-agent.pipeline` (same precedence as `llm_endpoints`: agent.json >
registry.yaml > `classify_images.py`'s hardcoded `_PIPELINE_DEFAULTS` fallback).

| Key | Default | Meaning |
|---|---|---|
| `batch_size` | 10 | images per run |
| `min_tags_target` | 6 | tag-count goal the optional cycles aim for |
| `max_conversation_messages` | 20 | hard cap on the classify_image_full conversation |
| `step_max_retries` | 2 | retries per required step (+1 initial attempt) before aborting the image |
| `step_retry_backoff_seconds` | 3 | sleep between a required step's retry attempts |
| `followup_max_tokens` | 1024 | token budget for the 3 optional cycles |
| `face_similarity_threshold` | 0.85 | cosine similarity for token→portrait face-match |
| `step_max_tokens.type` | 256 | token budget for STEP 1 (image type) |
| `step_max_tokens.visual` | 4096 | token budget for STEP 2 (visual analysis - the large nested JSON) |
| `step_max_tokens.pf2e` | 512 | token budget for STEP 3 (character or environment branch) |
| `step_max_tokens.description` | 512 | token budget for STEP 4 (flavor description) |

See `docs/agents/vision/image-classification-workflow.md` for the full step-by-step
conversation flow.

## Valid Classification Values

### Image Types
portrait · body · battlemap · scene · token

### PF2e Ancestries
human · elf · dwarf · halfling · gnome · goblin · leshy · lizardfolk · ratfolk · tengu ·
catfolk · orc · shoony · anadi · grippli · automaton · fleshwarp · fetchling · sprite ·
kitsune · kobold · android · strix · vanara · gnoll · goloma · hobgoblin · poppet · shisk ·
conrasu · beastkin · azarketi · dhampir · aasimar · tiefling · changeling · half-elf ·
half-orc · duskwalker · geniekin · ifrit · oread · sylph · undine · nagaji · vishkanya ·
dark-elf · none

### PF2e Classes
alchemist · barbarian · bard · champion · cleric · druid · fighter · gunslinger · inventor ·
investigator · kineticist · magus · monk · oracle · psychic · ranger · rogue · sorcerer ·
summoner · swashbuckler · thaumaturge · witch · wizard · animist · exemplar · commander · none

### PF2e Creature Types (monsters without a class)
aberration · animal · astral · beast · celestial · construct · dragon · dream · elemental ·
fey · fiend · fungus · giant · humanoid · monitor · ooze · phantom · plant · spirit ·
time · undead · none

### Elements
fire · water · earth · air · metal · wood · nature · dark · light · void · vitality · none

### Environments (battlemap/scene only)
dungeon · cave · forest · city · tavern · desert · snow · sea · swamp · ruins · temple ·
castle · plains · mountain · volcano · underwater · sky · astral · shadow · abyss ·
interior · exterior · none
