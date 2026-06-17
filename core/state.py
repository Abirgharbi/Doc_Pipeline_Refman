"""
Pipeline state — flows through every stage unchanged.
New pipeline shape:
  DoclingIngest → VLM → Recombine → Extract → Output
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Figure:
    """
    One image/figure extracted from a document by Docling.
    Filled in order:
      1. DoclingIngestStage  → everything except vlm_description
      2. VLMStage            → vlm_description
    """
    placeholder: str        # e.g. "[[FIGURE_0]]"  — used to splice back into text
    index: int              # order of appearance in the document (0-based)
    document_name: str
    page: int               # 1-based page number
    image_bytes: bytes      # raw PNG bytes
    image_base64: str       # base64-encoded string for VLM APIs
    caption: str = ""       # caption extracted by Docling if present
    vlm_description: str = ""  # filled by VLMStage


@dataclass
class ParsedDocument:
    """
    One document after DoclingIngestStage.
    text_with_placeholders has [[FIGURE_N]] markers where figures appear.
    combined_text is filled by RecombineStage after VLM descriptions are ready.
    """
    name: str
    text_with_placeholders: str         # text + [[FIGURE_N]] markers
    figures: list[Figure]               # all figures in document order
    combined_text: str = ""             # text with [[FIGURE_N]] replaced by VLM descriptions
    page_count: int = 0
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedData:
    """LLM-extracted structured data from one document (from combined_text)."""
    document_name: str
    data: dict[str, Any]
    raw_text: str           # the combined_text that was fed to the LLM
    error: str | None = None


@dataclass
class PipelineState:
    """Single state object that flows through all stages."""

    # ── Input ─────────────────────────────────────────────────────────────────
    query: str = ""
    # Each file: {"name": str, "content_base64": str, "mime_type": str}
    raw_files: list[dict[str, Any]] = field(default_factory=list)

    # ── After DoclingIngestStage ──────────────────────────────────────────────
    documents: list[ParsedDocument] = field(default_factory=list)

    # ── After VLMStage (figures updated in-place) ─────────────────────────────
    # (no new field — Figure.vlm_description is filled inside each document's figures list)

    # ── After RecombineStage ──────────────────────────────────────────────────
    # (ParsedDocument.combined_text is filled — documents list unchanged)

    # ── After ExtractStage ────────────────────────────────────────────────────
    extractions: list[ExtractedData] = field(default_factory=list)

    # ── After OutputStage ─────────────────────────────────────────────────────
    output: Any = None

    # ── Session / diagnostics ─────────────────────────────────────────────────
    session_id: str = ""
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
