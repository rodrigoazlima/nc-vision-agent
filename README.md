# nc-vision-agent

Standalone extraction of the Nexus Campaigns monorepo's vision agent. Classifies RPG
images dropped into a vault's `.knowledge-base/00-Inbox/` using a local Qwen3-VL vision
model (served by [LM Studio](https://lmstudio.ai/)), renames them to a canonical slug,
and writes draft entity notes to `01-Processing/`. See `CLAUDE.md` for the full behavior
contract and `AGENT.md` for inputs/outputs/responsibilities.

## Layout

```
src/nc_vision_agent/
  tools/            classify_images.py, extract_text.py, backfill_short_drafts.py
  prompts/          the raw LLM prompt files
install.py          pre-flight check (validates .system/ is present + compatible)
run.py              entrypoint (wraps classify_images.py)
.system/            NOT checked in - the monorepo's shared library (nexus.shared),
                     copied or bind-mounted in at deploy/runtime
state/              this agent's own runtime state (gitignored)
```

## Setup

This agent depends on `nexus.shared` from the parent monorepo. It is not vendored or
published to PyPI - copy the monorepo's `system/` folder into this repo's root as
`.system/` before running anything.

```powershell
pip install -e .
python install.py     # fails fast with a clear message if .system/ is missing/incompatible
```

## Run

```powershell
python run.py                                          # classify pending images
python -m nc_vision_agent.tools.backfill_short_drafts   # one-time draft migration
python src\nc_vision_agent\tools\extract_text.py        # OCR-style text extraction
```

LM Studio must be running at `http://localhost:1234/v1`.

## Web UI

The local Web UI provides a simple upload-and-classify flow. It uses the same
`run.py` pipeline as the command-line entrypoint, so complete the setup above
and start LM Studio first.

```powershell
python webui\app.py
```

Open http://127.0.0.1:8765 in a browser, select or drop an image, then choose
**Classify image**. The upload is placed in
`.knowledge-base/00-Inbox/webui-uploads/`; the agent's classification result
and generated draft are shown in the page. The server listens only on
`127.0.0.1` and is intended for local use.

## Docker

```powershell
docker build -t nc-vision-agent .
docker run -v ${PWD}/.system:/app/.system -v ${PWD}/.knowledge-base:/app/.knowledge-base nc-vision-agent
```

The image does not bake `.system/` in - mount it (or the vault) at run time.
