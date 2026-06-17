"""
Pipeline configuration — one object covers all five stages.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Text LLM used for the final extraction step."""
    provider: str = "ollama"        # "ollama" | "openai" | "lmstudio"
    model: str = "mistral"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120


class VLMConfig(BaseModel):
    """
    Vision-Language Model used to describe extracted figures.
    Only Ollama is supported (local, free).

    Recommended models (pull with `ollama pull <model>`):
      llava            — general purpose, good quality
      llava-phi3       — lightweight, fast
      moondream        — very small, fastest
      minicpm-v        — good quality / speed balance
      llama3.2-vision  — Meta vision model
    """
    enabled: bool = True
    model: str = "llava"
    base_url: str = "http://localhost:11434"
    # Prompt sent to the VLM for every figure
    prompt: str = (
        "Describe this figure in full detail. "
        "If it is a chart or graph, state its type, axes, values, and trends. "
        "If it is a diagram or schema, describe all components and relationships. "
        "If it is a table rendered as an image, transcribe it. "
        "If it is a photo or illustration, describe what is shown. "
        "Be precise — your description will replace the image in a document analysis pipeline."
    )
    concurrency: int = 1    # VLM is GPU-bound; keep at 1 unless you have multiple GPUs


class DoclingConfig(BaseModel):
    """Controls the Docling parsing step."""
    images_scale: float = 2.0       # resolution multiplier for extracted figures (1–4)
    ocr_enabled: bool = False       # enable OCR for scanned PDFs (slower)

    # ── GPU / accelerator ─────────────────────────────────────────────────────
    # "auto"  → Docling picks CUDA if available, else CPU  (recommended)
    # "cuda"  → force CUDA; falls back to CPU automatically on failure
    # "mps"   → Apple Silicon GPU
    # "cpu"   → force CPU only
    accelerator: str = "auto"
    num_threads: int = 4            # CPU threads used when not on GPU


class ExtractionConfig(BaseModel):
    """Controls the LLM extraction step (runs on combined_text)."""
    schema_definition: dict[str, Any]

    system_prompt: str = (
        "You are a precise data-extraction assistant. "
        "Return ONLY valid JSON that matches the schema. No prose, no markdown fences."
    )

    user_prompt_template: str = (
        "Extract information from the following document. "
        "The document text includes figure descriptions enclosed in "
        "[FIGURE DESCRIPTION] markers — treat these as part of the content.\n\n"
        "Document:\n{document_text}\n\n"
        "Return a single JSON object matching this schema exactly:\n{schema}"
    )

    fallback_on_error: bool = True


class PipelineConfig(BaseModel):
    """Top-level config passed to all five stages."""
    docling: DoclingConfig = Field(default_factory=DoclingConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig | None = None   # None → skip ExtractStage
    max_concurrency: int = 2   # parallel documents during ingest / extract
