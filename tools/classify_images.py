"""vision.tools.classify_images

Classifies RPG images in 00-Inbox/images/ via Qwen3-VL.
Renames images to canonical slug format in-place.
Writes AGENTS.md-compliant draft entities to 01-Processing/.
No LLM available → skip gracefully (no images marked failed). Batch: 10 per run.
"""

from __future__ import annotations

import json
import re
import sys
import time
import uuid as _uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

_TOOLS_DIR    = Path(__file__).resolve().parent
_AGENTS_DIR   = _TOOLS_DIR.parents[1]
_PROJECT_ROOT = _AGENTS_DIR.parent

if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from nexus.shared import (  # noqa: E402
    FrontmatterIO,
    LLMClient,
    Logger,
    LLMOfflineError,
    LLMResponseError,
    SignalEmitter,
    VisionClassification,
    build_entity_slug,
    image_tag,
    locked_update_queue_entry,
    sha256_of_file,
    to_slug,
)
from nexus.shared.config import LLMEndpointConfig  # noqa: E402
from nexus.shared.llm_client import _resize_and_encode  # noqa: E402
from nexus.shared.loaders import load_llm_endpoint  # noqa: E402
from nexus.shared.models import Element, Environment, ImageType  # noqa: E402

# classification.tools.enrich_tags owns state/tag-library.json - read-only
# here (cycle 4 aligns against it, never writes it; classification is the
# sole writer of canonical entries/aliases/counts).
_CLASSIFICATION_TAG_LIBRARY = _AGENTS_DIR / "classification" / "state" / "tag-library.json"

# Same 18-value taxonomy as classification agent's _ALLOWED_TYPES
# (agents/classification/tools/enrich_tags.py) - kept in sync manually, same
# convention already used for the PF2E_* vocab lists.
_ENTITY_TYPES: frozenset[str] = frozenset({
    "npc", "character", "faction", "location", "city", "village", "dungeon",
    "item", "artifact", "quest", "encounter", "creature", "monster", "event",
    "religion", "organization", "timeline", "lore",
})

# Adapted from agents/classification/prompts/enrich-tags.txt's dashboard-tab
# guidance - duplicated per this codebase's existing manual-sync convention
# for prompt text (see vision CLAUDE.md's PF2E_* sync note); update both
# together if the taxonomy changes.
_ENTITY_TYPE_GUIDANCE = """Entity types by dashboard tab:

CHARACTERS & NPCS - npc (named non-player character, default for people), character (playable/significant individual)
BESTIARY - creature (reusable monster stat-block), monster (unique/legendary named monster), encounter (a placed fight)
PLACES - location (generic site/region), city, village, dungeon (underground ruin/cave/tomb)
FACTIONS & POWERS - faction (group with a goal/agenda), organization (formal group, no antagonist role), religion
QUESTS & EVENTS - quest (adventure hook), event (past/ongoing event), timeline (sequence of events)
ITEMS & LORE - item (mundane/magical object), artifact (unique powerful relic), lore (world knowledge, not a physical object)

Choose the MOST SPECIFIC type. Prefer dungeon over location for a tomb/cave, creature over npc for a monster, artifact over item for relics."""

_VISION_SYSTEM_PROMPT = "You are a Pathfinder 2e image classifier. Return ONLY valid JSON."

# Every prompt that asks the LLM for tags asks for "short (1-3 word)" concrete
# tags, but nothing enforced that - a full visual_analysis sentence (e.g. "a
# large sword with a black blade and hilt, featuring white geometric
# patterns...") could slip straight into frontmatter tags as if it were one
# tag. Guard at the single append point every tag-collection site shares.
_MAX_TAG_WORDS = 6


def _is_concrete_tag(tag: str) -> bool:
    """Reject sentence-length strings the LLM sometimes returns instead of a
    short tag - keeps frontmatter `tags:` scannable/searchable rather than
    carrying paragraph-length prose."""
    return bool(tag) and 1 <= len(tag.split()) <= _MAX_TAG_WORDS

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)

TASK_ID         = "vision-agent"
SCRIPT_BASENAME = "classify_images.py"

# ---------------------------------------------------------------------------
# Pipeline config - registry.yaml agents.vision.options is the shared default;
# agent.json tasks.vision-agent.pipeline overrides it per-machine (same
# precedence as load_llm_endpoint's agent.json > registry.yaml > hardcoded
# fallback). Raw dict reads, not RegistryConfig/pydantic - same convention
# workers already use for their own per-worker `options:` blocks
# (nexus.workers.base.WorkerConfig.options) since this is free-form, not part
# of the agent-registry.spec.md schema.
# ---------------------------------------------------------------------------
_PIPELINE_DEFAULTS: dict[str, Any] = {
    "batch_size":                 10,
    "min_tags_target":            6,
    "max_conversation_messages":  20,
    "step_max_retries":           2,   # + 1 initial attempt = 3 tries per required step
    "step_retry_backoff_seconds": 3,
    "followup_max_tokens":        1024,
    "face_similarity_threshold":  0.85,
    # Vision-model token budgets per required step - generous on purpose
    # (assertive over cost-efficient). "visual" (STEP 2) asks for a large
    # nested visual_analysis JSON that a small budget can truncate mid-object;
    # the other steps return a handful of short fields.
    "step_max_tokens": {"type": 256, "visual": 4096, "pf2e": 512, "description": 512},
}

_REGISTRY_FILE = _AGENTS_DIR / "registry.yaml"
_AGENT_JSON_FILE = _AGENTS_DIR / "vision" / "agent.json"


def _load_pipeline_config() -> dict[str, Any]:
    cfg = json.loads(json.dumps(_PIPELINE_DEFAULTS))  # cheap deep copy

    try:
        registry = yaml.safe_load(_REGISTRY_FILE.read_text(encoding="utf-8")) or {}
        opts = registry["agents"]["vision"]["options"]
        for key in cfg:
            if key in opts:
                cfg[key] = opts[key]
    except Exception:
        pass  # registry.yaml missing/malformed - keep _PIPELINE_DEFAULTS

    try:
        agent_cfg = json.loads(_AGENT_JSON_FILE.read_text(encoding="utf-8"))
        overrides = agent_cfg["tasks"][TASK_ID]["pipeline"]
        for key in cfg:
            if key in overrides:
                cfg[key] = overrides[key]
    except Exception:
        pass  # no agent.json / no pipeline block - registry default applies

    return cfg


_PIPELINE                   = _load_pipeline_config()
BATCH_SIZE                  = int(_PIPELINE["batch_size"])
_MIN_TAGS_TARGET            = int(_PIPELINE["min_tags_target"])
_MAX_CONVERSATION_MESSAGES  = int(_PIPELINE["max_conversation_messages"])
_STEP_MAX_RETRIES           = int(_PIPELINE["step_max_retries"])
_STEP_RETRY_BACKOFF_S       = float(_PIPELINE["step_retry_backoff_seconds"])
_FOLLOWUP_MAX_TOKENS        = int(_PIPELINE["followup_max_tokens"])
_FACE_SIMILARITY_THRESHOLD  = float(_PIPELINE["face_similarity_threshold"])
_STEP_MAX_TOKENS: dict[str, int] = {k: int(v) for k, v in _PIPELINE["step_max_tokens"].items()}

_VAULT_ROOT   = _PROJECT_ROOT / ".knowledge-base"
_INBOX        = _VAULT_ROOT / "00-Inbox"
_INBOX_IMAGES = _INBOX  # scan entire inbox, not just images/ subdir
_PROCESSING   = _VAULT_ROOT / "01-Processing"
_AGENT_STATE  = _AGENTS_DIR / "vision" / "state"
_LOGS_DIR     = _AGENT_STATE / "logs"
_SHARED_STATE = _PROJECT_ROOT / "system" / "state"
_MASTER_LOG   = _AGENTS_DIR / "runtime" / "state" / "logs" / "automation.log"
_PROC_IMAGES  = _AGENT_STATE / "processed-images.json"
_TOKEN_LINKS  = _AGENT_STATE / "token-links.json"
_QUEUE_FILE   = _SHARED_STATE / "inbox-queue.json"
_GEN_TOKENS   = _PROJECT_ROOT / "system" / "state" / "workers" / "token" / "generated-tokens.json"
_PROMPT_DIR   = _AGENTS_DIR / "vision" / "prompts"
_SIGNALS_DIR  = _AGENTS_DIR / "runtime" / "state" / "signals"

# The 4 required classify_image_full steps, each its own single-purpose LLM
# turn. "pf2e" has two variants - the branch is picked in code from step 1's
# already-parsed type (see classify_image_full), never asked as an LLM
# if/else.
_STEP_PROMPT_FILES: dict[str, str] = {
    "type":             "classify-step1-type.txt",
    "visual":           "classify-step2-visual.txt",
    "pf2e_character":   "classify-step3-pf2e-character.txt",
    "pf2e_environment": "classify-step3-pf2e-environment.txt",
    "description":      "classify-step4-description.txt",
}

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})

# Types excluded from face-match candidate pool
_EXCLUDE_TYPES = frozenset({"token", "battlemap", "scene"})

# Model selectable via agents/registry.yaml -> llm_endpoints.vision_llm.model
_FALLBACK_LLM_CFG = LLMEndpointConfig(
    url      = "http://localhost:1234/v1/chat/completions",
    model    = "qwen3-vl-4b-instruct",
    type     = "vision",
    provider = "lmstudio",
)


_LLM_CFG = load_llm_endpoint(
    "vision_llm",
    fallback     = _FALLBACK_LLM_CFG,
    agent_dir    = _AGENTS_DIR / "vision",
    task_id      = TASK_ID,
    project_root = _PROJECT_ROOT,
)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

# Despite the name (kept for on-disk field compatibility), this is blake2b,
# not SHA-256 - see nexus.shared.hashing's module docstring.
_sha256 = sha256_of_file


# ---------------------------------------------------------------------------
# Token detection
# ---------------------------------------------------------------------------

def _is_token(path: Path) -> bool:
    """Return True if PNG has ≥2 transparent corners - canonical token detection."""
    if path.suffix.lower() != ".png":
        return False
    try:
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        corners = [
            img.getpixel((0, 0)),
            img.getpixel((w - 1, 0)),
            img.getpixel((0, h - 1)),
            img.getpixel((w - 1, h - 1)),
        ]
        return sum(1 for c in corners if c[3] < 128) >= 2
    except Exception:
        return False


# ---------------------------------------------------------------------------
# State I/O  (atomic writes per G8)
# ---------------------------------------------------------------------------

def _load_state() -> dict[str, Any]:
    if not _PROC_IMAGES.exists():
        return {"version": 2, "images": {}, "pathIndex": {}}
    return json.loads(_PROC_IMAGES.read_text(encoding="utf-8"))


def _save_state(state: dict) -> None:
    _AGENT_STATE.mkdir(parents=True, exist_ok=True)
    tmp = _PROC_IMAGES.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(_PROC_IMAGES)


def retry_failed_images(state: dict) -> int:
    """Remove failed pseudo-entries so their unchanged source paths can run again.

    Successful and migrated entries remain untouched.  The caller must persist
    the returned state with ``_save_state`` after this operation.
    """
    images = state.setdefault("images", {})
    path_index = state.setdefault("pathIndex", {})
    failed_paths = {
        rel for rel, key in path_index.items()
        if isinstance(key, str) and key.startswith("path:")
        and isinstance(images.get(key), dict)
        and images[key].get("status") == "failed"
    }
    for rel in failed_paths:
        key = path_index.pop(rel)
        images.pop(key, None)
    return len(failed_paths)


def _load_token_links() -> dict[str, Any]:
    if not _TOKEN_LINKS.exists():
        return {}
    return json.loads(_TOKEN_LINKS.read_text(encoding="utf-8"))


def _save_token_links(links: dict) -> None:
    _AGENT_STATE.mkdir(parents=True, exist_ok=True)
    tmp = _TOKEN_LINKS.with_suffix(".tmp")
    tmp.write_text(json.dumps(links, indent=2, default=str), encoding="utf-8")
    tmp.replace(_TOKEN_LINKS)


def _load_queue() -> dict[str, Any]:
    if not _QUEUE_FILE.exists():
        return {}
    return json.loads(_QUEUE_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------

def _generated_token_paths() -> set[str]:
    """Project-relative paths of dashboard-generated tokens.

    These artifacts live next to their source in 00-Inbox but are NOT raw input -
    re-ingesting them as new images is what duplicates tokens (each gm/view edit
    writes a differently-named token file; _resolve_image_target then bumps each to
    -01, -02…). Skip them here so they are never treated as source images.
    """
    if not _GEN_TOKENS.exists():
        return set()
    try:
        gen = json.loads(_GEN_TOKENS.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {v["tokenPath"] for v in gen.values() if isinstance(v, dict) and v.get("tokenPath")}


def _is_token_file(path: Path) -> bool:
    """Filename-convention fallback for _generated_token_paths().

    Same predicate as nexus.workers.ingestion._is_token_file (and token.py's
    own _is_token_stem) - ingestion already uses it to keep generated tokens
    out of inbox-queue.json in the first place. Vision's candidate scan walks
    00-Inbox directly, independent of that queue, and until now only trusted
    generated-tokens.json's path index to exclude them here - an index keyed
    by whatever path token.py wrote at generation time. A worker output_dir
    change (token.py moved its default from 00-Inbox to 01-Processing, see
    that module's _DEFAULT_CFG comment) orphans any token generated under the
    old location: the index no longer lists it, so the lookup above silently
    stops excluding it and it gets reclassified as a "new" source image.
    Checking the filename convention directly - like ingestion already does -
    catches those orphans regardless of index state.
    """
    stem = path.stem
    return stem.endswith("-token") or ".token" in stem


def _candidate_images(state: dict, queue: dict) -> list[Path]:
    """Return unprocessed images, non-PNG first (tokens last per spec)."""
    path_index: set[str] = set(state.get("pathIndex", {}).keys())
    gen_tokens: set[str] = _generated_token_paths()
    images: list[Path] = []
    for path in sorted(_INBOX_IMAGES.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
            continue
        rel = path.relative_to(_PROJECT_ROOT).as_posix()
        if rel in path_index:
            continue
        if rel in gen_tokens:
            continue
        if _is_token_file(path):
            continue
        agents = queue.get(rel, {}).get("agents", {})
        if isinstance(agents, dict) and agents.get("vision") in ("done", "paused"):
            continue
        images.append(path)
    return sorted(images, key=lambda p: (p.suffix.lower() == ".png", p))


# ---------------------------------------------------------------------------
# Token face matching
# ---------------------------------------------------------------------------

def _get_folder_candidates(folder: Path, state: dict) -> list[Path]:
    """Return processed non-token, non-battlemap image paths in the same folder.

    Only returns images with a successful state entry (status != failed).
    Used to build the candidate pool for token face matching.
    """
    path_index = state.get("pathIndex", {})
    images     = state.get("images", {})
    candidates: list[Path] = []

    for rel_str, sha_or_key in path_index.items():
        if sha_or_key.startswith("path:"):
            continue  # failed image
        path = _PROJECT_ROOT / rel_str
        if path.parent != folder:
            continue
        if path.suffix.lower() not in _IMAGE_EXTS:
            continue
        entry = images.get(sha_or_key)
        if not entry or not isinstance(entry, dict):
            continue
        if entry.get("status") == "failed":
            continue
        if entry.get("type") in _EXCLUDE_TYPES:
            continue
        if path.exists():
            candidates.append(path)

    return candidates


def _try_face_match(token_path: Path, candidates: list[Path]) -> Optional[Path]:
    """Match a token to its source portrait via center-crop pixel similarity.

    Uses PIL + numpy only - no heavy dependencies. Falls back to None if
    either dep is missing or any image fails to load.
    Returns the best match above _FACE_SIMILARITY_THRESHOLD, or None.
    """
    if not candidates:
        return None

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    def _face_vector(path: Path) -> Optional[Any]:
        try:
            img = Image.open(path).convert("L")
            w, h = img.size
            # Heuristic face region: horizontal center 40%, vertical upper 50%
            x0, x1 = int(w * 0.3), int(w * 0.7)
            y0, y1 = int(h * 0.1), int(h * 0.6)
            crop = img.crop((x0, y0, x1, y1)).resize((32, 32))
            vec  = np.array(crop, dtype=np.float32).flatten()
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else None
        except Exception:
            return None

    token_vec = _face_vector(token_path)
    if token_vec is None:
        return None

    best_path:  Optional[Path] = None
    best_score: float          = 0.0

    for cand in candidates:
        cand_vec = _face_vector(cand)
        if cand_vec is None:
            continue
        score = float((token_vec * cand_vec).sum())  # cosine sim (vecs are unit-norm)
        if score > best_score:
            best_score = score
            best_path  = cand

    return best_path if best_score >= _FACE_SIMILARITY_THRESHOLD else None


def _inherit_clf_from_state(entry: dict) -> VisionClassification:
    """Rebuild a VisionClassification from a processed-images.json entry.

    Overrides type to `token` - the token inherits all other metadata,
    including the source portrait's final tags/entity_type (same character,
    so both are just as applicable to the token), from its matched source.
    """
    return VisionClassification(
        type           = ImageType.token,
        ancestry       = entry.get("ancestry", "none"),
        **{"class": entry.get("class", "none")},
        creature_type  = entry.get("creature_type", "none"),
        element        = Element(entry.get("element",      "none")),
        environment    = Environment(entry.get("environment", "none")),
        description    = entry.get("description", ""),
        candidate_tags = entry.get("candidate_tags", []),
        entity_type    = entry.get("entity_type", "none"),
    )


# ---------------------------------------------------------------------------
# Slug builders
# ---------------------------------------------------------------------------

# Entity types that can never legitimately BE a scene/battlemap themselves -
# always a standalone physical object, not a place, creature, or character
# that might reasonably appear within one. classify-step1-type.txt only
# offers 4 structural types (portrait/body/battlemap/scene) - there is no
# "isolated object" option - so a photographed weapon/artifact with little
# visible environment gets forced into "scene" by elimination. Deliberately
# narrow (not "everything except place-like types"): a scene/battlemap
# landing on entity_type "creature" or "npc" is an accepted, expected
# disagreement (a monster or character standing in an environment shot is
# still genuinely a scene) - only item/artifact are unambiguous.
_OBJECT_ENTITY_TYPES: frozenset[str] = frozenset({"item", "artifact"})


def _is_object_in_scene_bucket(clf: VisionClassification) -> bool:
    """True when the structural type (scene/battlemap) and the grounded
    entity_type disagree in a way that means this is really a photographed
    object, not an environment (or a creature/character within one)."""
    return clf.type.value in ("scene", "battlemap") and clf.entity_type in _OBJECT_ENTITY_TYPES


def _object_slug_descriptor(clf: VisionClassification) -> Optional[str]:
    """Most concrete single descriptor for an object-in-scene-bucket image:
    the first candidate tag that survives slugification, else the
    environment (still better than nothing), else None."""
    for tag in clf.candidate_tags:
        slugged = to_slug(tag)
        if slugged:
            return slugged
    if clf.environment and clf.environment.value != "none":
        return clf.environment.value
    return None


def _image_filename_slug(clf: VisionClassification) -> str:
    """Canonical image filename slug per agent-vision.spec.md.

    portrait / body / token       : {ancestry}-{class}-{element}.{type}
    battlemap / scene             : {type}-{environment}
    battlemap / scene (as object) : {entity_type}-{descriptor}
    """
    t = clf.type.value
    if t in ("portrait", "body", "token"):
        parts: list[str] = []
        if clf.ancestry and clf.ancestry != "none":
            parts.append(to_slug(clf.ancestry))
        elif clf.creature_type and clf.creature_type != "none":
            parts.append(to_slug(clf.creature_type))
        if clf.char_class and clf.char_class != "none":
            parts.append(to_slug(clf.char_class))
        if clf.element and clf.element.value != "none":
            parts.append(clf.element.value)
        return ("-".join(parts) if parts else "unknown") + f".{t}"
    elif _is_object_in_scene_bucket(clf):
        return build_entity_slug(clf.entity_type, _object_slug_descriptor(clf) or "")
    else:
        env = clf.environment.value if clf.environment.value != "none" else "unknown"
        return f"{t}-{env}"


def _entity_slug(clf: VisionClassification) -> str:
    """Entity slug for the MD file per data-contracts.spec.md: {type}-{descriptors}."""
    t = clf.type.value
    if t in ("portrait", "body", "token"):
        parts = [t]
        if clf.ancestry and clf.ancestry != "none":
            parts.append(to_slug(clf.ancestry))
        elif clf.creature_type and clf.creature_type != "none":
            parts.append(to_slug(clf.creature_type))
        if clf.char_class and clf.char_class != "none":
            parts.append(to_slug(clf.char_class))
        if clf.element and clf.element.value != "none":
            parts.append(clf.element.value)
        return "-".join(parts)
    elif _is_object_in_scene_bucket(clf):
        return build_entity_slug(clf.entity_type, _object_slug_descriptor(clf) or "")
    else:
        parts = [t]
        if clf.environment and clf.environment.value != "none":
            parts.append(clf.environment.value)
        return "-".join(parts)


def _resolve_image_target(src: Path, img_slug: str) -> Path:
    """Resolve target path for in-place image rename. Bumps filename on collision."""
    ext    = src.suffix.lower()
    target = src.parent / f"{img_slug}{ext}"
    if not target.exists() or target == src:
        return target
    counter = 1
    while True:
        target = src.parent / f"{img_slug}-{counter:02d}{ext}"
        if not target.exists():
            return target
        counter += 1


def _resolve_entity_path(slug: str) -> Path:
    """Resolve output MD path in 01-Processing/. Bumps on collision."""
    _PROCESSING.mkdir(parents=True, exist_ok=True)
    candidate = _PROCESSING / f"{slug}.md"
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = _PROCESSING / f"{slug}-{counter:02d}.md"
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# Image rename
# ---------------------------------------------------------------------------

def _rename_image(src: Path, clf: VisionClassification) -> Path:
    """Rename image to canonical slug format in-place. Returns new path."""
    img_slug = _image_filename_slug(clf)
    target   = _resolve_image_target(src, img_slug)
    if target != src:
        src.rename(target)
    return target


# ---------------------------------------------------------------------------
# Draft writing
# ---------------------------------------------------------------------------

def _atmosphere_lines(clf: VisionClassification) -> str:
    """Generate atmosphere bullet points from classification data."""
    env  = clf.environment.value if clf.environment.value != "none" else None
    elem = clf.element.value     if clf.element.value     != "none" else None

    lighting_map = {
        "dark":     "Dim, shadowy - torchlight or moonlight only",
        "light":    "Bright and open, natural or magical illumination",
        "fire":     "Flickering orange glow, heat haze near the source",
        "void":     "Absolute darkness with pinpricks of cold starlight",
        "vitality": "Warm golden light, life energy visible as soft radiance",
    }
    mood_map = {
        "dark":     "Dread and tension - something watches from the shadows",
        "fire":     "Urgency and danger - heat presses in from all sides",
        "water":    "Eerie calm, reflections distort what is real",
        "earth":    "Solid and ancient - the weight of stone above is palpable",
        "air":      "Vertiginous openness - wind howls and pulls at equipment",
        "void":     "Cosmic isolation - existence feels fragile here",
        "vitality": "Hopeful and sacred - wounds ache less, resolve strengthens",
        "nature":   "Alive and watching - rustles and growths respond to presence",
        "metal":    "Cold and mechanical - every footstep rings against hard floors",
        "wood":     "Organic and overgrown - roots and vines creep through cracks",
    }
    sound_map = {
        "cave":      "Dripping water, distant echoes, the crunch of gravel underfoot",
        "dungeon":   "Stone settling, faint moans, the scrape of old iron chains",
        "forest":    "Wind through canopy, birdsong that abruptly goes silent",
        "city":      "Crowd murmur, hawkers calling, distant bells",
        "tavern":    "Lute and laughter, the clink of mugs, a fire popping",
        "temple":    "Hollow silence broken by chanting, incense thick in the air",
        "volcano":   "Deep rumbles, hissing steam vents, cracking lava crust",
        "ruins":     "Wind through broken walls, loose stone underfoot, owls",
        "castle":    "Echoing boots on stone, distant orders shouted across battlements",
        "sea":       "Constant wave crash, salt spray, creaking rigging",
        "swamp":     "Frogs, insect chorus, the suck of mud at every step",
        "desert":    "Wind-driven sand, vast silence, the creak of parched wood",
        "snow":      "Muffled sound, breath visible, the creak of ice underfoot",
        "mountain":  "Howling updrafts, distant rockfall, thin air that burns the lungs",
        "underwater": "Muffled rushing, pressure in the ears, bubbles rising",
    }

    atm_lines = [f"- **Setting**: {env.capitalize() if env else 'Unknown'} environment"]
    if elem and elem in lighting_map:
        atm_lines.append(f"- **Lighting**: {lighting_map[elem]}")
    if elem and elem in mood_map:
        atm_lines.append(f"- **Mood**: {mood_map[elem]}")
    if env and env in sound_map:
        atm_lines.append(f"- **Sounds**: {sound_map[env]}")
    return "\n".join(atm_lines)


def _battlemap_body(clf: VisionClassification) -> str:
    env  = clf.environment.value if clf.environment.value != "none" else "unknown"
    elem = clf.element.value     if clf.element.value     != "none" else "none"

    tactical_map: dict[str, list[str]] = {
        "cave":      ["Narrow chokepoints force single-file movement",
                      "Stalagmites provide half-cover; stalactites can be dropped as hazards",
                      "Darkness beyond 30 ft unless light sources are carried"],
        "dungeon":   ["Corridors limit flanking opportunities",
                      "Doorways create fatal-funnel chokepoints",
                      "Rubble and debris create difficult terrain patches"],
        "forest":    ["Trees grant three-quarters cover at range",
                      "Dense undergrowth is difficult terrain (5 ft of movement per 5 ft)",
                      "High canopy may allow flying or climbing ambush"],
        "city":      ["Rooftops accessible for archers - watch vertical threats",
                      "Alleyways split groups; coordinating between lanes is difficult",
                      "Civilians scatter - area spells risk collateral consequences"],
        "ruins":     ["Unstable floors - failing checks drop combatants one level",
                      "Partial walls give half-cover without blocking movement",
                      "Rubble fields are difficult terrain throughout"],
        "volcano":   ["Lava pools deal 4d10 fire damage on contact (or per round)",
                      "Steam vents are hazardous terrain - DC 15 Acrobatics to avoid",
                      "Ground cracks can open as environmental hazards"],
        "temple":    ["Pillars grant cover and can be toppled as an action",
                      "Raised dais gives +1 circumstance bonus on attacks",
                      "Ritual circles on the floor may interact with spells"],
        "sea":       ["Open water = difficult terrain for non-swimmers",
                      "Ship deck limits large creature movement significantly",
                      "Masts and rigging allow Climb DC 15 for high-ground advantage"],
        "castle":    ["Arrow slits grant near-total cover to defenders inside",
                      "Portcullis can be dropped to split the party",
                      "Parapets grant cover and advantage on ranged attacks"],
    }
    hooks_map: dict[str, list[str]] = {
        "cave":    ["Something moves in the deeper darkness - bones crunch underfoot",
                    "A hidden fissure leads to an unexplored passage",
                    "Ancient markings on the walls predate the current inhabitants"],
        "dungeon": ["A cell block holds something that shouldn't still be alive",
                    "The lock mechanism on the far door is set to trap the next opener",
                    "Rations and gear scattered - the previous expedition ended badly here"],
        "forest":  ["Tracks converge on a specific tree - a den or lair above or below",
                    "The silence radius suggests a predator is active nearby",
                    "Fallen shrine in the clearing hints at a forgotten pact"],
        "ruins":   ["A sealed vault door bears a sigil that matches a faction crest",
                    "Fresh campfire ash - someone camped here very recently",
                    "The collapse pattern suggests this wasn't accidental"],
        "volcano": ["Cultists performing a ritual at the caldera edge at midnight",
                    "A heat-warped chest juts from the cooled lava - survivors' cache?",
                    "The eruption cycle is accelerating; the party has limited time"],
        "temple":  ["The altar responds to blood - what does it summon?",
                    "A hidden reliquary behind the main idol holds something valuable",
                    "The priests here serve a god whose name causes the walls to vibrate"],
        "city":    ["A crowd has gathered around something - or someone - on the ground",
                    "Wanted posters on every wall bearing a face that looks familiar",
                    "A merchant's stall is a front; the real business is in the back room"],
    }

    default_tactics = ["Open terrain dominates - movement and positioning are key",
                        "Identify the highest point - height advantage is decisive here",
                        "Flanking lanes are wide; spread out or be surrounded"]
    default_hooks   = ["Something has been disturbed here recently - evidence of passage",
                        "An item of interest is visible but retrieving it is the challenge",
                        "The environment itself becomes an antagonist as the fight progresses"]

    tactics = tactical_map.get(env, default_tactics)
    hooks   = hooks_map.get(env,   default_hooks)

    return (
        f"\n## Description\n\n{clf.description}\n\n"
        f"## Atmosphere\n\n{_atmosphere_lines(clf)}\n\n"
        f"## Tactical Notes\n\n"
        + "\n".join(f"- {t}" for t in tactics) + "\n\n"
        f"## Encounter Hooks\n\n"
        + "\n".join(f"- {h}" for h in hooks) + "\n\n"
        f"## Details\n\n"
        f"- **Type**: Battlemap\n"
        f"- **Environment**: {env}\n"
        f"- **Element**: {elem}\n\n"
        f"## Related\n\n"
    )


def _scene_body(clf: VisionClassification) -> str:
    env  = clf.environment.value if clf.environment.value != "none" else "unknown"
    elem = clf.element.value     if clf.element.value     != "none" else "none"

    hooks_map: dict[str, list[str]] = {
        "cave":      ["A figure stands at the cave mouth - ally, enemy, or something else?",
                      "The light source in this scene will not last much longer",
                      "What the figure sees ahead should terrify any sane adventurer"],
        "dungeon":   ["This is the moment before the door opens - what waits beyond?",
                      "The party must choose: press on or fall back with what they have",
                      "Someone in this image knows more than they are saying"],
        "forest":    ["The encounter began with an arrow - from which direction?",
                      "A figure emerges from the treeline; their intent is unclear",
                      "The beast here is not acting like prey - it is hunting"],
        "volcano":   ["The ritual can still be stopped - but the window is closing",
                      "Escape routes are being cut off by the eruption",
                      "The antagonist has planned for this terrain; the party has not"],
        "city":      ["The crowd turns hostile - who gave the signal?",
                      "The chase leads somewhere the party did not expect",
                      "A witness to something they should not have seen"],
        "interior":  ["The object that matters is somewhere in this room",
                      "The conversation happening here will change everything",
                      "Someone in this scene is lying"],
    }
    default_hooks = [
        "The scene captures a pivotal moment - what came just before?",
        "A detail in the background tells a different story than the foreground",
        "The emotional stakes here can anchor a session's dramatic peak",
    ]
    notes_map: dict[str, list[str]] = {
        "dark":  ["Use this scene at a low point in the campaign arc",
                  "Dim the lights at the table when describing this moment"],
        "fire":  ["Read aloud the heat and urgency - time pressure is the mechanic",
                  "This works as an action climax or an emotional confrontation"],
        "light": ["Scene suggests revelation or triumph - pair with a party achievement",
                  "High contrast lighting suggests a moral choice moment"],
    }
    default_notes = ["Scene works as an establishing shot for a new location or encounter",
                     "Use the description to set the tone before the players act"]

    hooks = hooks_map.get(env,  default_hooks)
    notes = notes_map.get(elem, default_notes)

    return (
        f"\n## Description\n\n{clf.description}\n\n"
        f"## Atmosphere\n\n{_atmosphere_lines(clf)}\n\n"
        f"## Story Hooks\n\n"
        + "\n".join(f"- {h}" for h in hooks) + "\n\n"
        f"## DM Notes\n\n"
        + "\n".join(f"- {n}" for n in notes) + "\n\n"
        f"## Details\n\n"
        f"- **Type**: Scene\n"
        f"- **Environment**: {env}\n"
        f"- **Element**: {elem}\n\n"
        f"## Related\n\n"
    )


def _token_body(clf: VisionClassification) -> str:
    ancestry      = clf.ancestry      if clf.ancestry      != "none" else None
    creature_type = clf.creature_type if clf.creature_type != "none" else None
    char_class    = clf.char_class    if clf.char_class    != "none" else None
    elem          = clf.element.value if clf.element.value != "none" else None

    subject = ancestry or creature_type or "Unknown"
    role_hints: list[str] = []
    if creature_type:
        creature_roles = {
            "dragon":    ["Boss encounter", "Ancient rival", "Unlikely ally with a price"],
            "undead":    ["Recurring villain", "Tragic cursed NPC", "Guardian of a sealed tomb"],
            "fiend":     ["Patron in disguise", "Antagonist pulling strings from afar", "Deal-maker"],
            "celestial": ["Divine messenger", "Quest giver", "Moral compass NPC"],
            "construct": ["Guard automaton", "Arcane experiment escaped", "Loyal servitor"],
            "beast":     ["Wilderness encounter", "Companion animal", "Territorial hazard"],
            "humanoid":  ["Recurring NPC", "Faction representative", "Ambiguous moral agent"],
        }
        role_hints = creature_roles.get(creature_type, ["Encounter creature", "Faction-aligned being"])
    elif ancestry:
        role_hints = ["Player character", "Named NPC ally or rival", "Recurring face in the campaign"]
    else:
        role_hints = ["Unnamed encounter participant", "Background NPC", "Crowd member with a secret"]

    vtt_notes = [
        "512×512 circular portrait - drop directly into Foundry VTT or Roll20",
        "Scale to 1×1 grid square for Medium creatures; 2×2 for Large",
    ]
    if elem:
        vtt_notes.append(f"Consider a {elem}-coloured aura overlay for visual identification")

    return (
        f"\n## Description\n\n{clf.description}\n\n"
        f"## Visual Notes\n\n"
        f"- **Subject**: {subject.capitalize()}"
        + (f" {char_class}" if char_class else "") + "\n"
        + (f"- **Element**: {elem.capitalize()}\n" if elem else "")
        + "\n"
        f"## Suggested Roles\n\n"
        + "\n".join(f"- {r}" for r in role_hints) + "\n\n"
        f"## VTT Usage\n\n"
        + "\n".join(f"- {n}" for n in vtt_notes) + "\n\n"
        f"## Details\n\n"
        f"- **Ancestry**: {clf.ancestry}\n"
        f"- **Class**: {clf.char_class}\n"
        f"- **Creature type**: {clf.creature_type}\n"
        f"- **Element**: {clf.element.value}\n\n"
        f"## Related\n\n"
    )


def _portrait_body(clf: VisionClassification) -> str:
    """Minimal body for portrait/body types - lore agent will enrich these."""
    return (
        f"\n## Description\n\n{clf.description}\n\n"
        f"## Visual Classification\n\n"
        f"- **Ancestry**: {clf.ancestry}\n"
        f"- **Class**: {clf.char_class}\n"
        f"- **Creature type**: {clf.creature_type}\n"
        f"- **Element**: {clf.element.value}\n"
        f"- **Environment**: {clf.environment.value}\n\n"
        f"## Lore Status\n\n"
        f"- Pending NPC sheet generation by Lore Agent\n"
        f"- Classification complete - awaiting scenario pairing\n\n"
        f"## Related\n\n"
    )


def _item_body(clf: VisionClassification, entity_type: str) -> str:
    """Body for an object-in-scene-bucket image (see _is_object_in_scene_bucket):
    structurally 'scene'/'battlemap' (no dedicated isolated-object image type
    exists) but entity_type says this is really a photographed item/artifact/
    creature/etc, not a place. Scene's Atmosphere/Story-Hooks template would
    misleadingly describe an object photo as an establishing shot; this is a
    minimal, honest body instead - same spirit as _portrait_body."""
    concrete_tags = [t for t in clf.candidate_tags if t not in (clf.type.value, clf.environment.value)]
    return (
        f"\n## Description\n\n{clf.description}\n\n"
        f"## Visual Details\n\n"
        + "".join(f"- {t}\n" for t in concrete_tags[:8])
        + "\n"
        f"## DM Notes\n\n"
        f"- Detected as **{entity_type}** from an image the vision model could not "
        f"place in a specific environment - confirm type/name on review\n\n"
        f"## Details\n\n"
        f"- **Type**: {entity_type.capitalize()}\n"
        f"- **Element**: {clf.element.value}\n\n"
        f"## Related\n\n"
    )


def _write_draft(
    path: Path,
    clf: VisionClassification,
    image_path: Path,
    sha256: Optional[str] = None,
    original_image_path: Optional[Path] = None,
    entity_uuid: Optional[str] = None,
) -> str:
    """Writes the draft entity. Returns the resolved entity_type (frontmatter
    'type') so callers can log it distinctly from clf.type (the image's
    structural portrait/body/battlemap/scene/token category) - the two are
    decided by separate, uncoordinated LLM cycles in classify_image_full and
    can disagree (e.g. a confirmed 'scene' image landing on entity_type
    'creature'); logging only clf.type previously hid which value actually
    became the note's type field.

    `original_image_path` is the pre-rename path (_rename_image renames the
    image in-place to its canonical slug before this is called) - when it
    differs from `image_path`, the draft records both under `source`
    (current path) and `originalSource` (as-dropped path/filename), so the
    provenance trail survives the rename instead of only being reconstructable
    after the fact from automation.log timestamps.

    `entity_uuid`, when given, becomes the note's `uuid:` frontmatter instead
    of a freshly minted one - callers pass the same uuid they're about to
    (or already did) record against this image's sha256 in
    processed-images.json, so the note (dashboard URL) and the state ledger
    agree on one identifier instead of each minting their own.
    """
    today   = date.today().isoformat()
    slug    = path.stem
    rel_img = image_path.relative_to(_PROJECT_ROOT).as_posix()

    tags: list[str] = [clf.type.value]
    if clf.ancestry and clf.ancestry != "none":
        tags.append(clf.ancestry)
    if clf.creature_type and clf.creature_type != "none":
        tags.append(clf.creature_type)
    if clf.element and clf.element.value != "none":
        tags.append(clf.element.value)
    if clf.environment and clf.environment.value != "none":
        tags.append(clf.environment.value)
    # Final, library-aligned brainstorm from classify_image_full's multi-cycle
    # conversation (state-only until now - this is the one place it becomes
    # the note's actual tags).
    for tag in clf.candidate_tags:
        if tag not in tags:
            tags.append(tag)

    # Grounded guess from the multi-cycle conversation's entity-type cycle
    # takes priority; the coarse portrait/body/token->npc, else->location
    # placeholder only applies when that cycle didn't run or came back "none".
    entity_type = clf.entity_type if clf.entity_type != "none" else (
        "npc" if clf.type.value in ("portrait", "body", "token") else "location"
    )

    frontmatter: dict[str, Any] = {
        "id":            slug,
        "uuid":          entity_uuid or str(_uuid.uuid4()),
        "type":          entity_type,
        "status":        "draft",
        "quality":       0,
        "created":       today,
        "updated":       today,
        "tags":          tags,
        "source":        [rel_img],
        "reviewed":      False,
        "relationships": [],
    }
    if sha256:
        frontmatter["sha256"] = sha256
    if original_image_path is not None:
        orig_rel = original_image_path.relative_to(_PROJECT_ROOT).as_posix()
        if orig_rel != rel_img:
            frontmatter["originalSource"] = [orig_rel]

    t = clf.type.value
    if _is_object_in_scene_bucket(clf):
        body = _item_body(clf, entity_type)
    elif t == "battlemap":
        body = _battlemap_body(clf)
    elif t == "scene":
        body = _scene_body(clf)
    elif t == "token":
        body = _token_body(clf)
    else:
        body = _portrait_body(clf)

    FrontmatterIO().write(path, frontmatter, body)
    return entity_type


# ---------------------------------------------------------------------------
# Candidate tag harvesting
# ---------------------------------------------------------------------------

# Array leaves under the prompt's `visual_analysis` block that name concrete,
# visible things worth surfacing as tag candidates (classify-step2-visual.txt
# already asks the LLM for all of these - they were previously parsed and
# discarded, since VisionClassification has no field for them).
_CANDIDATE_TAG_PATHS: tuple[tuple[str, str], ...] = (
    ("equipment", "weapons"),
    ("equipment", "shield"),
    ("equipment", "focus_items"),
    ("equipment", "tools"),
    ("equipment", "musical_instruments"),
    ("equipment", "books"),
    ("equipment", "other"),
    ("clothing", "materials"),
    ("clothing", "ornaments"),
    ("fantasy_features", "creatures"),
    ("fantasy_features", "wings"),
    ("fantasy_features", "horns"),
    ("fantasy_features", "tail"),
    ("fantasy_features", "halo"),
    ("environment_details", "architecture"),
    ("environment_details", "vegetation"),
)


def _extract_candidate_tags(raw: dict) -> list[str]:
    """Flatten visual_analysis array leaves into a deduped raw tag list.

    Best-effort: any shape surprise in the LLM's free-text visual_analysis
    block yields [] rather than failing classification over a tagging extra.
    """
    try:
        analysis = raw.get("visual_analysis") or {}
        seen: dict[str, None] = {}
        for section, key in _CANDIDATE_TAG_PATHS:
            for item in (analysis.get(section) or {}).get(key) or []:
                if isinstance(item, dict):
                    item = item.get("type") or item.get("name")
                if not isinstance(item, str):
                    continue
                norm = item.strip().lower()
                if norm and _is_concrete_tag(norm):
                    seen.setdefault(norm, None)
        return list(seen)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Multi-step/cycle classification - image stays in one conversation across up
# to _MAX_CONVERSATION_MESSAGES messages: 4 required steps (type, visual
# analysis, PF2e classification, description - see _run_required_step, each
# retried up to _STEP_MAX_RETRIES times before aborting the image), then
# dedicated image-type, entity-type, and tag-library-refinement follow-up
# cycles that degrade gracefully instead of retrying. Completion goal for
# those follow-ups: >= _MIN_TAGS_TARGET tags + both categories set.
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict:
    """Strip a leading/trailing ```json fence before parsing, if present.

    Observed live this session: LocalRouter's `auto` model sometimes wraps
    otherwise-valid JSON in a markdown code fence. Cheap insurance, not a
    correctness risk - falls through to plain json.loads when there's no fence.
    """
    text = raw.strip()
    m = _JSON_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _read_tag_library() -> dict[str, Any]:
    """Read-only view of classification agent's canonical tag library."""
    try:
        return json.loads(_CLASSIFICATION_TAG_LIBRARY.read_text(encoding="utf-8"))
    except Exception:
        return {"tags": {}}


def _image_content_block(img_path: Path) -> dict:
    b64 = _resize_and_encode(img_path)
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _load_step_prompts() -> dict[str, str]:
    """Read the 4 required-step prompt files (5 on disk - "pf2e" has a
    character/environment variant, see _STEP_PROMPT_FILES) fresh each call so
    an edited prompt file is picked up without restarting the process."""
    prompts: dict[str, str] = {}
    for key, filename in _STEP_PROMPT_FILES.items():
        p = _PROMPT_DIR / filename
        prompts[key] = p.read_text(encoding="utf-8") if p.exists() else ""
    return prompts


def _cycle2_prompt(tags_so_far: list[str]) -> str:
    remaining = max(0, _MIN_TAGS_TARGET - len(tags_so_far))
    return (
        "Confirm the image category and continue tagging.\n\n"
        "Category - respond with exactly one of: portrait, body, battlemap, scene, token.\n\n"
        f"We have {len(tags_so_far)} tag(s) so far: {', '.join(tags_so_far) or 'none'}. "
        "List additional short (1-3 word) concrete tags describing objects, materials, "
        f"subjects, or themes visible in the image, not already listed"
        + (f" - aim for at least {remaining} more so the total reaches {_MIN_TAGS_TARGET}." if remaining else ".")
        + '\n\nReturn ONLY valid JSON:\n{"category": "...", "additional_tags": ["...", "..."]}'
    )


def _cycle3_prompt(tags_so_far: list[str]) -> str:
    remaining = max(0, _MIN_TAGS_TARGET - len(tags_so_far))
    return (
        "Now determine the entity type this image will become once promoted "
        "into our Dungeon Master's knowledge base.\n\n"
        f"{_ENTITY_TYPE_GUIDANCE}\n\n"
        f"We have {len(tags_so_far)} tag(s) so far: {', '.join(tags_so_far) or 'none'}. "
        "List any additional concrete tags not yet mentioned"
        + (f" - aim for at least {remaining} more so the total reaches {_MIN_TAGS_TARGET}." if remaining else ".")
        + '\n\nReturn ONLY valid JSON:\n{"entity_type": "...", "additional_tags": ["...", "..."]}'
    )


def _cycle4_prompt(current_tags: list[str], known_tags: list[str], entity_type_hint: str) -> str:
    return (
        "Here is our tag library - canonical tags already in use elsewhere in "
        f"this knowledge base: {', '.join(known_tags) or 'none yet'}\n\n"
        f"Our current tag list for this image: {', '.join(current_tags) or 'none'}\n\n"
        "Align these tags: where a known tag above means the same thing as one of "
        "ours, prefer the known spelling. Finalize the complete tag list for this "
        f"image (merge, dedupe, keep concrete and specific), aiming for at least "
        f"{_MIN_TAGS_TARGET} total if the image supports it. Confirm the entity "
        f"type (current best guess: {entity_type_hint}).\n\n"
        'Return ONLY valid JSON:\n{"final_tags": ["...", "..."], "entity_type": "..."}'
    )


def refine_tags_with_library(
    img_path: Path,
    client: LLMClient,
    current_tags: list[str],
    entity_type_hint: str,
    library: dict[str, Any],
    *,
    history: Optional[list[dict]] = None,
) -> tuple[list[str], str]:
    """Cycle 4, standalone-callable: align current_tags against the tag
    library's known canonical tags and finalize tags + entity type, image
    still in context. Public so other agents/system code can re-run just
    this refinement later (e.g. after the library has grown) without
    redoing the full classification - pass history=None for a fresh
    conversation, or an existing message list to extend it in place.
    """
    known_tags = sorted(
        library.get("tags", {}),
        key=lambda t: -library["tags"][t].get("count", 0),
    )[:20]
    prompt_text = _cycle4_prompt(current_tags, known_tags, entity_type_hint)

    if history is not None:
        messages = history
        messages.append({"role": "user", "content": prompt_text})
    else:
        messages = [
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {"role": "user", "content": [_image_content_block(img_path), {"type": "text", "text": prompt_text}]},
        ]

    if len(messages) + 1 > _MAX_CONVERSATION_MESSAGES:
        # Over budget even before this turn's reply - accept what we have
        # rather than exceed the message cap.
        return current_tags, entity_type_hint

    final_tags, entity_type = current_tags, entity_type_hint
    try:
        raw_text = client.chat(messages, max_tokens=_FOLLOWUP_MAX_TOKENS)
        messages.append({"role": "assistant", "content": raw_text})
        resp = _parse_json_response(raw_text)
        candidate = resp.get("final_tags")
        if isinstance(candidate, list) and candidate:
            # Merge, never replace: cycle 4 asks the LLM to "finalize the
            # complete tag list", and a model that just repeats a subset of
            # what it was given (or rephrases "weapon" as "stylized sword")
            # would otherwise silently erase tags earlier cycles already
            # grounded in the image's own visual_analysis.
            merged = list(current_tags)
            for t in candidate:
                if isinstance(t, str) and t.strip():
                    norm = t.strip().lower()
                    if norm not in merged and _is_concrete_tag(norm):
                        merged.append(norm)
            final_tags = merged
        et = resp.get("entity_type")
        if isinstance(et, str) and et in _ENTITY_TYPES:
            entity_type = et
    except LLMOfflineError:
        raise
    except Exception:
        pass  # graceful degrade - keep current_tags/entity_type_hint as-is

    return final_tags, entity_type


_STRUCTURAL_TYPES = frozenset({"portrait", "body", "battlemap", "scene"})


def _validate_step1(d: dict) -> None:
    if d.get("type") not in _STRUCTURAL_TYPES:
        raise ValueError(f"step 1: invalid/missing type {d.get('type')!r}")


def _validate_step2(d: dict) -> None:
    if not isinstance(d.get("visual_analysis"), dict):
        raise ValueError("step 2: missing visual_analysis object")


def _validate_step4(d: dict) -> None:
    if not isinstance(d.get("description"), str) or not d["description"].strip():
        raise ValueError("step 4: missing/empty description")


def _run_required_step(
    messages: list[dict],
    client: LLMClient,
    content: Any,
    max_tokens: int,
    validate: Optional[Any] = None,
) -> dict:
    """Send one required-step turn to completion, image already in
    conversation context. Unlike cycles 2-4 below (which degrade gracefully
    on any hiccup), a required step retries up to _STEP_MAX_RETRIES times -
    same JSON turn re-sent with a short corrective nudge and backoff - before
    giving up. LLMOfflineError is never retried here; it propagates
    immediately so the caller's caller (main's per-image try/except) applies
    its own connection-error handling instead of this step swallowing it.
    Exhausting retries raises LLMResponseError, which aborts just this one
    image (same contract the old single-call cycle 1 had).
    """
    messages.append({"role": "user", "content": content})
    last_err: Optional[Exception] = None
    for attempt in range(_STEP_MAX_RETRIES + 1):
        if len(messages) + 1 > _MAX_CONVERSATION_MESSAGES:
            raise LLMResponseError(
                f"message budget ({_MAX_CONVERSATION_MESSAGES}) exhausted before a required step completed"
            )
        try:
            raw_text = client.chat(messages, max_tokens=max_tokens)
            messages.append({"role": "assistant", "content": raw_text})
            parsed = _parse_json_response(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
            if validate is not None:
                validate(parsed)
            return parsed
        except LLMOfflineError:
            raise
        except Exception as exc:
            last_err = exc
            if attempt < _STEP_MAX_RETRIES:
                time.sleep(_STEP_RETRY_BACKOFF_S)
                messages.append({
                    "role": "user",
                    "content": "That was not valid JSON, or was missing a required field. "
                                "Return ONLY valid JSON matching the requested schema.",
                })
    raise LLMResponseError(
        f"required step never returned valid JSON after {_STEP_MAX_RETRIES + 1} attempt(s): {last_err}"
    )


def classify_image_full(
    img_path: Path,
    client: LLMClient,
    step_prompts: dict[str, str],
    is_tk: bool,
) -> VisionClassification:
    """Full classification, image held in one conversation throughout:

    4 required steps, each its own single-purpose LLM turn (see
    _run_required_step) - image type, visual analysis, PF2e classification,
    flavor description - followed by 3 optional follow-up cycles (confirm
    type + tags, entity type + tags, tag-library refinement) that degrade
    gracefully on failure instead of aborting. Public - the single entry
    point other agents/system code should call for a from-image
    classification.

    STEP 3 (PF2e classification) has a character variant (ancestry/class/
    creature_type) and an environment variant (environment/element) - the
    branch is picked here, in code, from step 1's own already-parsed type,
    never asked as an LLM if/else the way the old monolithic prompt did.

    Raises LLMOfflineError immediately (no retry - the caller's caller
    handles that) or LLMResponseError once a required step exhausts its
    retries. Optional cycles keep prior values on parse/format errors, but
    propagate LLMOfflineError so main() can retry the model swap once.
    """
    step_prompts = step_prompts or _load_step_prompts()
    messages: list[dict] = [{"role": "system", "content": _VISION_SYSTEM_PROMPT}]

    # --- Step 1: image type ---
    step1 = _run_required_step(
        messages, client,
        [_image_content_block(img_path), {"type": "text", "text": step_prompts.get("type", "")}],
        _STEP_MAX_TOKENS["type"], validate=_validate_step1,
    )

    # --- Step 2: visual analysis ---
    step2 = _run_required_step(
        messages, client, step_prompts.get("visual", ""), _STEP_MAX_TOKENS["visual"], validate=_validate_step2,
    )

    # --- Step 3: PF2e classification - branch decided from step 1's type ---
    is_character = step1["type"] in ("portrait", "body")
    step3 = _run_required_step(
        messages, client,
        step_prompts.get("pf2e_character" if is_character else "pf2e_environment", ""),
        _STEP_MAX_TOKENS["pf2e"],
    )

    # --- Step 4: flavor description ---
    step4 = _run_required_step(
        messages, client, step_prompts.get("description", ""), _STEP_MAX_TOKENS["description"],
        validate=_validate_step4,
    )

    raw: dict[str, Any] = {
        "type":           step1["type"],
        "ancestry":       step3.get("ancestry", "none") if is_character else "none",
        "class":          step3.get("class", "none") if is_character else "none",
        "creature_type":  step3.get("creature_type", "none") if is_character else "none",
        "element":        step3.get("element", "none") if not is_character else "none",
        "environment":    step3.get("environment", "none") if not is_character else "none",
        "description":    step4.get("description", ""),
        "visual_analysis": step2.get("visual_analysis", {}),
    }
    clf = VisionClassification.model_validate(raw)
    if is_tk:
        clf = clf.model_copy(update={"type": ImageType.token})
    tags: list[str] = list(dict.fromkeys(_extract_candidate_tags(raw)))

    # --- Cycle 2: image type + more tags ---
    try:
        messages.append({"role": "user", "content": _cycle2_prompt(tags)})
        raw_text = client.chat(messages, max_tokens=_FOLLOWUP_MAX_TOKENS)
        messages.append({"role": "assistant", "content": raw_text})
        resp = _parse_json_response(raw_text)
        cat = resp.get("category")
        if not is_tk and isinstance(cat, str) and cat in {"portrait", "body", "battlemap", "scene", "token"}:
            clf = clf.model_copy(update={"type": ImageType(cat)})
        for t in resp.get("additional_tags") or []:
            if isinstance(t, str) and t.strip():
                norm = t.strip().lower()
                if norm not in tags and _is_concrete_tag(norm):
                    tags.append(norm)
    except LLMOfflineError:
        raise
    except Exception:
        pass  # keep the required steps' type/tags - never fail the image over this

    # --- Cycle 3: entity type + more tags ---
    entity_type = "none"
    try:
        messages.append({"role": "user", "content": _cycle3_prompt(tags)})
        raw_text = client.chat(messages, max_tokens=_FOLLOWUP_MAX_TOKENS)
        messages.append({"role": "assistant", "content": raw_text})
        resp = _parse_json_response(raw_text)
        et = resp.get("entity_type")
        if isinstance(et, str) and et in _ENTITY_TYPES:
            entity_type = et
        for t in resp.get("additional_tags") or []:
            if isinstance(t, str) and t.strip():
                norm = t.strip().lower()
                if norm not in tags and _is_concrete_tag(norm):
                    tags.append(norm)
    except LLMOfflineError:
        raise
    except Exception:
        pass

    # --- Cycle 4: tag-library refinement (in-conversation) ---
    final_tags, entity_type = refine_tags_with_library(
        img_path, client, tags, entity_type, _read_tag_library(), history=messages,
    )

    return clf.model_copy(update={"candidate_tags": final_tags, "entity_type": entity_type})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(*, retry_failed: bool = False) -> None:
    log = Logger(
        task_id=TASK_ID,
        script_basename=SCRIPT_BASENAME,
        logs_dir=_LOGS_DIR,
        master_log=_MASTER_LOG,
    )
    t0 = log.start()

    _AGENT_STATE.mkdir(parents=True, exist_ok=True)
    _PROCESSING.mkdir(parents=True, exist_ok=True)

    client = LLMClient(_LLM_CFG)
    if not client.is_available():
        log.warning("Qwen3-VL (localhost:1234) offline - skipping batch")
        log.done(t0, key="classified", count=0, failed=0)
        sys.exit(0)

    step_prompts = _load_step_prompts()
    state       = _load_state()
    if retry_failed:
        retried = retry_failed_images(state)
        if retried:
            _save_state(state)
        log.info(f"Cleared {retried} failed image entr{'y' if retried == 1 else 'ies'} for retry")
    token_links = _load_token_links()
    queue       = _load_queue()
    candidates  = _candidate_images(state, queue)

    if not candidates:
        log.info("No unprocessed images found")
        log.done(t0, key="classified", count=0, failed=0)
        sys.exit(0)

    batch   = candidates[:BATCH_SIZE]
    count   = 0
    failed  = 0
    emitter = SignalEmitter(_SIGNALS_DIR)
    log.info(f"Batch: {len(batch)} of {len(candidates)} image(s)")

    for img_path in batch:
        orig_img_path = img_path  # pre-rename Path - _rename_image reassigns img_path below
        orig_rel = img_path.relative_to(_PROJECT_ROOT).as_posix()
        is_tk    = _is_token(img_path)

        # --- Classification: face match for tokens, LLM for everything else ---
        clf: Optional[VisionClassification] = None
        matched_source: Optional[Path]      = None

        if is_tk:
            folder_candidates = _get_folder_candidates(img_path.parent, state)
            matched_source    = _try_face_match(img_path, folder_candidates)
            if matched_source is not None:
                matched_rel = matched_source.relative_to(_PROJECT_ROOT).as_posix()
                matched_sha = state["pathIndex"].get(matched_rel)
                if matched_sha and not matched_sha.startswith("path:"):
                    entry = state["images"].get(matched_sha)
                    if entry:
                        clf = _inherit_clf_from_state(entry)
                        log.info(
                            f"Token {img_path.name} → face-matched to "
                            f"{matched_source.name} (inherited meta)"
                        )

        if clf is None:
            try:
                clf = classify_image_full(img_path, client, step_prompts, is_tk)
            except LLMOfflineError:
                # LM Studio can only hold one model loaded at a time; a
                # concurrent request for another model briefly evicts this
                # one. One retry after a short wait rides out that swap
                # instead of aborting the whole batch.
                log.warning(f"LLM offline while processing {img_path.name} - retrying once")
                time.sleep(10)
                try:
                    clf = classify_image_full(img_path, client, step_prompts, is_tk)
                except LLMOfflineError:
                    log.warning(f"LLM still offline for {img_path.name} - aborting batch")
                    break
            except (LLMResponseError, Exception) as exc:
                sha_fail  = _sha256(img_path)
                fail_uuid = str(_uuid.uuid4())
                log.error(f"Classification failed for {img_path.name}: {exc}{image_tag(sha256=sha_fail, uuid=fail_uuid, path=orig_rel)}")
                state["images"][f"path:{orig_rel}"] = {
                    "uuid":          fail_uuid,
                    "path":          orig_rel,
                    "processedAt":   datetime.now(timezone.utc).isoformat(),
                    "originalName":  img_path.name,
                    "type":          "unknown",
                    "ancestry":      "none",
                    "class":         "none",
                    "creature_type": "none",
                    "element":       "none",
                    "environment":   "none",
                    "description":   "",
                    "candidate_tags": [],
                    "sha256":        sha_fail,
                    "isToken":       is_tk,
                    "status":        "failed",
                }
                state["pathIndex"][orig_rel] = f"path:{orig_rel}"
                _save_state(state)
                failed += 1
                continue

        # --- Rename image (steps 4–6) ---
        try:
            img_path = _rename_image(img_path, clf)
        except Exception as exc:
            log.warning(f"Could not rename {img_path.name}: {exc} - using original path{image_tag(path=orig_rel)}")

        new_rel = img_path.relative_to(_PROJECT_ROOT).as_posix()
        sha     = _sha256(img_path)
        # Reuse the uuid already on record for this hash (e.g. a stale-token
        # regeneration re-classifying the same source) instead of minting a
        # second one - the note frontmatter and processed-images.json must
        # agree on a single identifier for the same image.
        entity_uuid = state.get("images", {}).get(sha, {}).get("uuid") or str(_uuid.uuid4())

        # --- Write draft (step 7) ---
        e_slug   = _entity_slug(clf)
        out_path = _resolve_entity_path(e_slug)
        entity_type = _write_draft(
            out_path, clf, img_path, sha256=sha, original_image_path=orig_img_path,
            entity_uuid=entity_uuid,
        )

        log.info(
            f"Classified: {img_path.name} → {out_path.name} "
            f"(image_type={clf.type.value}, entity_type={entity_type})"
            f"{image_tag(sha256=sha, uuid=entity_uuid, path=new_rel)}"
        )

        # --- Update processed-images.json (step 9) ---
        state["images"][sha] = {
            "uuid":          entity_uuid,
            "path":          new_rel,
            "processedAt":   datetime.now(timezone.utc).isoformat(),
            "originalName":  img_path.name,
            "type":          clf.type.value,
            "ancestry":      clf.ancestry,
            "class":         clf.char_class,
            "creature_type": clf.creature_type,
            "element":       clf.element.value,
            "environment":   clf.environment.value,
            "description":   clf.description,
            "candidate_tags": clf.candidate_tags,
            "entity_type":   clf.entity_type,
            "sha256":        sha,
            "isToken":       is_tk,
            "status":        "ok",
        }
        state["pathIndex"][new_rel] = sha
        _save_state(state)

        # --- Track face-match link in token-links.json ---
        if is_tk and matched_source is not None:
            matched_rel = matched_source.relative_to(_PROJECT_ROOT).as_posix()
            matched_sha = state["pathIndex"].get(matched_rel, "")
            token_links[sha] = {
                "tokenPath":   new_rel,
                "sourcePath":  matched_rel,
                "sourceSha256": matched_sha,
                "linkedAt":    datetime.now(timezone.utc).isoformat(),
            }
            _save_token_links(token_links)

        # --- Update inbox-queue.json (step 10) - locked, one item at a time ---
        if orig_rel in queue and isinstance(queue[orig_rel].get("agents"), dict):
            def _mark_vision_done(entry: dict) -> Optional[str]:
                entry.setdefault("agents", {})["vision"] = "done"
                entry["candidate_tags"] = clf.candidate_tags
                # Rename queue key to match new image path so classification/lore
                # can trace source frontmatter back to the correct queue entry.
                return new_rel if new_rel != orig_rel else None

            locked_update_queue_entry(_QUEUE_FILE, orig_rel, _mark_vision_done)
            queue = _load_queue()

        # --- Emit signal for Lore agent (fire-and-forget per G5) ---
        emitter.emit(
            signal_type="image-classified",
            emitter=TASK_ID,
            ref=img_path.name,
        )

        count += 1

    log.done(t0, key="classified", count=count, failed=failed)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main(retry_failed="--retry-failed" in sys.argv[1:])


# ---------------------------------------------------------------------------
# Agentic tool interface (claude-api dispatch)
# ---------------------------------------------------------------------------

from nexus.shared.agent_tools import SELF_MANAGEMENT_TOOLS, call_self_management_tool  # noqa: E402

_MODULE_FILE = Path(__file__)

TOOLS = SELF_MANAGEMENT_TOOLS + [
    {
        "name": "list_pending_images",
        "description": "Return JSON list of image paths in 00-Inbox/images/ not yet classified.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "detect_token",
        "description": (
            "Check whether a PNG image is a circular token "
            "(transparent corner/edge pixels). Returns {is_token, path}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the PNG image",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "match_token_face",
        "description": (
            "Attempt to match a token image to its source portrait via face distance. "
            "Returns {matched: bool, source_path?, score?}. "
            "Only meaningful for PNG images detected as tokens."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the token PNG image",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "classify_image",
        "description": (
            "Classify a single image via the local vision LLM "
            "(type, ancestry, class, element, environment). "
            "Returns JSON classification. Does NOT rename the file or write a draft."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the image file",
                },
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "run_batch",
        "description": (
            "Classify up to BATCH_SIZE pending images, rename them to canonical slugs, "
            "and write draft entities to 01-Processing/. "
            "Returns log output including count processed."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def call_tool(name: str, args: dict, context: dict) -> str:
    import io
    import contextlib
    import json as _json

    result = call_self_management_tool(
        name, args, context, module_file=_MODULE_FILE, task_id=TASK_ID
    )
    if result is not None:
        return result

    if name == "list_pending_images":
        state      = _load_state()
        queue      = _load_queue()
        candidates = _candidate_images(state, queue)
        return _json.dumps([str(p) for p in candidates[:20]])

    if name == "detect_token":
        p = Path(args["image_path"])
        return _json.dumps({"is_token": _is_token(p), "path": str(p)})

    if name == "match_token_face":
        p = Path(args["image_path"])
        if not p.exists():
            return _json.dumps({"error": f"Image not found: {p}"})
        state       = _load_state()
        folder_cands = _get_folder_candidates(p.parent, state)
        matched     = _try_face_match(p, folder_cands)
        if matched is None:
            return _json.dumps({"matched": False, "candidates_checked": len(folder_cands)})
        return _json.dumps({
            "matched":     True,
            "source_path": str(matched),
        })

    if name == "classify_image":
        p = Path(args["image_path"])
        if not p.exists():
            return _json.dumps({"error": f"Image not found: {p}"})
        client = LLMClient(_LLM_CFG)
        if not client.is_available():
            return _json.dumps({"error": "LLM offline (localhost:1234)"})
        step_prompts = _load_step_prompts()
        is_tk  = _is_token(p)
        try:
            clf = classify_image_full(p, client, step_prompts, is_tk)
            data = clf.model_dump()
            data["is_token"] = is_tk
            return _json.dumps(data)
        except LLMOfflineError:
            return _json.dumps({"error": "LLM offline"})
        except Exception as exc:
            return _json.dumps({"error": str(exc)})

    if name == "run_batch":
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                main()
        except SystemExit:
            pass
        return buf.getvalue().strip() or "Batch run complete"

    raise ValueError(f"Unknown tool: {name!r}")
