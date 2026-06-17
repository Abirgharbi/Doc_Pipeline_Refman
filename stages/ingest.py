"""
DoclingIngestStage — parse documents with Docling, GPU-accelerated.

GPU strategy:
  1. Try Docling with the configured accelerator (default: "auto" → CUDA if available).
  2. If CUDA initialisation or conversion fails, automatically retry on CPU.
  3. The device actually used is stored in ParsedDocument.metadata["accelerator_used"].

Supported formats: PDF, DOCX, PPTX, HTML, XLSX (anything Docling supports).
Requires: pip install docling>=2.0.0
"""
from __future__ import annotations
import asyncio
import base64
import io
import os
import tempfile
import warnings
from typing import Any

from doc_pipeline.stages.base import BaseStage
from doc_pipeline.core.state import PipelineState, ParsedDocument, Figure
from doc_pipeline.core.config import PipelineConfig


class DoclingIngestStage(BaseStage):

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        sem = asyncio.Semaphore(config.max_concurrency)
        tasks = [self._parse_one(f, config, sem) for f in state.raw_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res in enumerate(results):
            if isinstance(res, Exception):
                name = state.raw_files[i].get("name", f"file_{i}")
                state.errors.append(f"{name}: {res}")
                state.documents.append(
                    ParsedDocument(name=name, text_with_placeholders="", figures=[])
                )
            else:
                state.documents.append(res)

        return state

    async def _parse_one(
        self, file_entry: dict[str, Any], config: PipelineConfig, sem: asyncio.Semaphore
    ) -> ParsedDocument:
        async with sem:
            name = file_entry.get("name", "unknown")
            mime = file_entry.get("mime_type", "")
            raw  = base64.b64decode(file_entry["content_base64"])

            # Docling is synchronous — offload to thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, _parse_with_docling, raw, name, mime, config
            )


# ── Synchronous Docling parsing (runs in executor) ────────────────────────────

def _parse_with_docling(
    content: bytes, name: str, mime: str, config: PipelineConfig
) -> ParsedDocument:
    """
    Try GPU first, fall back to CPU automatically.
    Returns a ParsedDocument with metadata["accelerator_used"] set.
    """
    requested = config.docling.accelerator.lower()

    # First attempt — with requested accelerator (cuda / auto / mps)
    try:
        return _run_docling(content, name, mime, config, accelerator=requested)

    except Exception as gpu_exc:
        # Only retry on CPU when we were on CUDA/auto and it actually failed
        if requested in ("cuda", "auto"):
            warnings.warn(
                f"[doc_pipeline] Docling GPU failed for '{name}' "
                f"({type(gpu_exc).__name__}: {gpu_exc}). "
                "Retrying on CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _run_docling(
                content, name, mime, config,
                accelerator="cpu",
                fallback_reason=f"{type(gpu_exc).__name__}: {gpu_exc}",
            )
        raise


def _run_docling(
    content: bytes,
    name: str,
    mime: str,
    config: PipelineConfig,
    accelerator: str,
    fallback_reason: str | None = None,
) -> ParsedDocument:
    """Build a Docling converter with the specified accelerator and run conversion."""
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
    except ImportError as exc:
        raise ImportError(
            "Docling is not installed. Run: pip install 'docling>=2.0.0'"
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = config.docling.images_scale
    if config.docling.ocr_enabled:
        pipeline_options.do_ocr = True

    # Apply GPU / CPU accelerator (silent no-op on older Docling that lacks this API)
    _apply_accelerator(pipeline_options, accelerator, config.docling.num_threads)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Write to temp file — Docling needs a file path
    suffix = _mime_to_suffix(mime, name)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = converter.convert(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    doc = result.document
    text_with_placeholders, figures = _build_text_and_figures(doc, name)
    page_count = getattr(doc, "num_pages", 0) or len(getattr(doc, "pages", []))

    metadata: dict[str, Any] = {"accelerator_used": accelerator}
    if fallback_reason:
        metadata["accelerator_fallback_reason"] = fallback_reason

    return ParsedDocument(
        name=name,
        text_with_placeholders=text_with_placeholders,
        figures=figures,
        page_count=page_count,
        mime_type=mime,
        metadata=metadata,
    )


def _apply_accelerator(pipeline_options: Any, accelerator: str, num_threads: int) -> None:
    """
    Set AcceleratorOptions on the pipeline.
    Silent no-op when the installed Docling version does not support this API.
    """
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
    except ImportError:
        return   # older Docling — no accelerator API, continue without it

    device_map: dict[str, Any] = {
        "auto": AcceleratorDevice.AUTO,
        "cuda": AcceleratorDevice.CUDA,
        "cpu":  AcceleratorDevice.CPU,
        "mps":  AcceleratorDevice.MPS,
    }
    device = device_map.get(accelerator, AcceleratorDevice.AUTO)

    try:
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=device,
        )
    except Exception:
        pass   # AcceleratorOptions constructor changed in some builds — skip silently


# ── Document item walker ──────────────────────────────────────────────────────

def _build_text_and_figures(doc: Any, document_name: str) -> tuple[str, list[Figure]]:
    """
    Walk docling document items in reading order.
    Figures become [[FIGURE_N]] placeholders; all other content becomes text.
    """
    try:
        from docling_core.types.doc import PictureItem, TableItem, SectionHeaderItem
    except ImportError:
        # docling_core not importable — export raw markdown as fallback
        return doc.export_to_markdown(), []

    parts: list[str] = []
    figures: list[Figure] = []
    fig_idx = 0

    for item, level in doc.iterate_items():

        # ── Figures ───────────────────────────────────────────────────────────
        if isinstance(item, PictureItem):
            placeholder = f"[[FIGURE_{fig_idx}]]"

            img_bytes = b""
            img_b64   = ""
            try:
                pil_img = _get_pil_image(item, doc)
                if pil_img:
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                    img_b64   = base64.b64encode(img_bytes).decode()
            except Exception:
                pass

            caption = ""
            try:
                caption = item.caption_text(doc) or ""
            except Exception:
                pass

            page = 0
            try:
                if item.prov:
                    page = item.prov[0].page_no
            except Exception:
                pass

            figures.append(Figure(
                placeholder=placeholder,
                index=fig_idx,
                document_name=document_name,
                page=page,
                image_bytes=img_bytes,
                image_base64=img_b64,
                caption=caption,
            ))
            parts.append(f"\n{placeholder}\n")
            if caption:
                parts.append(f"[Caption: {caption}]\n")
            fig_idx += 1

        # ── Section headers ───────────────────────────────────────────────────
        elif isinstance(item, SectionHeaderItem):
            hashes = "#" * max(1, level)
            parts.append(f"\n{hashes} {item.text}\n")

        # ── Tables ────────────────────────────────────────────────────────────
        elif isinstance(item, TableItem):
            try:
                parts.append(f"\n{item.export_to_markdown()}\n")
            except Exception:
                parts.append("\n[TABLE — could not render]\n")

        # ── Regular text ──────────────────────────────────────────────────────
        else:
            text = getattr(item, "text", "") or ""
            if text.strip():
                parts.append(text + "\n")

    return "".join(parts), figures


def _get_pil_image(item: Any, doc: Any) -> Any:
    """Try multiple Docling API shapes to get a PIL Image."""
    if hasattr(item, "get_image"):
        return item.get_image(doc)
    if hasattr(item, "image") and item.image is not None:
        if hasattr(item.image, "pil_image"):
            return item.image.pil_image
    return None


def _mime_to_suffix(mime: str, filename: str) -> str:
    if "pdf" in mime or filename.endswith(".pdf"):
        return ".pdf"
    if "wordprocessingml" in mime or filename.endswith(".docx"):
        return ".docx"
    if "presentationml" in mime or filename.endswith(".pptx"):
        return ".pptx"
    if "spreadsheetml" in mime or filename.endswith(".xlsx"):
        return ".xlsx"
    if "html" in mime:
        return ".html"
    return os.path.splitext(filename)[1] or ".pdf"
