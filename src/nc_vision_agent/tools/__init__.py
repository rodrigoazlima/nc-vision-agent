"""nc_vision_agent.tools

Concrete modules:
  classify_images.py       - implements IImageClassifier + BaseAgent
  extract_text.py          - OCR-style text extraction over classified images
  backfill_short_drafts.py - one-off migration for old short-body drafts
Dependencies: ILLMClient (vision_llm endpoint), IStateStore (processed-images.json)
"""
