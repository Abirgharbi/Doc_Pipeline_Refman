"""
DoclingIngestStage — pypdfium2 pour PDFs, Docling pour autres formats.

DIAGNOSTIC FINAL (2026-06-22):
  Docling bloque sur 100% des chunks de ces PDFs STM32, même sur 1 page seule.
  La cause racine est TensorFlow/oneDNN qui s'initialise dans chaque subprocess
  et se bloque dans des allocations natives sur Windows — indépendamment du
  contenu PDF. Aucune configuration Docling ne résout ce problème.

  Solution finale :
  - PDFs     → pypdfium2 direct (texte natif + images rasterisées par page)
  - Non-PDFs → Docling classique (DOCX, PPTX, XLSX, HTML, inchangé)

  Le texte extrait par pypdfium2 EST le texte natif du PDF — identique à ce
  que Docling lit avant son pipeline ML. Zéro perte sur registres/tableaux/offsets.
  Les images sont rasterisées page par page (PNG) sans ML.

Requires: pypdfium2>=5.0.0  docling>=2.0.0 (pour non-PDF uniquement)
"""
from __future__ import annotations
import asyncio
import base64
import gc
import io
import os
import tempfile
import time
import warnings
from typing import Any

from doc_pipeline.stages.base import BaseStage
from doc_pipeline.core.state import PipelineState, ParsedDocument, Figure
from doc_pipeline.core.config import PipelineConfig


SKIP_FIRST_N_PAGES   = 6    # cover / TOC — pas de données registres
CHUNK_SIZE_PAGES     = 50   # batch de pages pour le logging (pas de subprocess)
VECTOR_PATH_THRESHOLD = 80  # nb de path objects au-delà duquel une page est
                             # considérée comme un diagramme vectoriel.
                             # Les PDFs STM32 n'ont PAS d'images bitmap (type 3) —
                             # tous leurs diagrammes (blocs, timing, clocks) sont
                             # des graphiques vectoriels composés de paths + texte.
                             # Une page de texte normal a ~30-60 paths (règles,
                             # cadres, puces). Un diagramme en a 80-200+.


def _log(msg: str) -> None:
    print(f"[ingest] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Stage entry point
# ══════════════════════════════════════════════════════════════════════════════

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
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, _parse_dispatch, raw, name, mime, config
            )


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

def _parse_dispatch(
    content: bytes, name: str, mime: str, config: PipelineConfig
) -> ParsedDocument:
    is_pdf = "pdf" in mime or name.lower().endswith(".pdf")
    if is_pdf:
        return _parse_pdf_pypdfium2(content, name, config)
    return _parse_non_pdf_docling(content, name, mime, config)


# ══════════════════════════════════════════════════════════════════════════════
# PDF — pypdfium2 : texte natif + figures rasterisées
# ══════════════════════════════════════════════════════════════════════════════

def _parse_pdf_pypdfium2(
    content: bytes, name: str, config: PipelineConfig
) -> ParsedDocument:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ImportError("pypdfium2 requis : pip install pypdfium2>=5.0.0") from exc

    doc_pdf     = pdfium.PdfDocument(content)
    total_pages = len(doc_pdf)
    first_page  = min(SKIP_FIRST_N_PAGES, total_pages)
    total_chunks = max(0, (total_pages - first_page + CHUNK_SIZE_PAGES - 1) // CHUNK_SIZE_PAGES)

    if first_page > 0:
        _log(f"'{name}': skipping first {first_page} page(s) (cover/TOC).")

    _log(f"'{name}': {total_pages} pages — pypdfium2 "
         f"(pages {first_page + 1}–{total_pages}, {total_chunks} chunks).")

    text_parts: list[str]    = []
    all_figures: list[Figure] = []
    fig_idx = 0
    skipped_pages: list[int] = []

    for chunk_num, chunk_start in enumerate(
        range(first_page, total_pages, CHUNK_SIZE_PAGES), start=1
    ):
        chunk_end = min(chunk_start + CHUNK_SIZE_PAGES, total_pages)
        t0 = time.monotonic()
        _log(f"'{name}': chunk {chunk_num}/{total_chunks} "
             f"(pages {chunk_start + 1}–{chunk_end}) starting...")

        for page_idx in range(chunk_start, chunk_end):
            try:
                page     = doc_pdf.get_page(page_idx)
                textpage = page.get_textpage()

                # ── Texte natif ──────────────────────────────────────────────
                page_text = textpage.get_text_bounded()
                if page_text and page_text.strip():
                    text_parts.append(f"\n\n--- Page {page_idx + 1} ---\n")
                    text_parts.append(page_text.strip())

                # ── Détection diagrammes vectoriels + bitmaps ────────────
                # Les PDFs STM32 n'ont PAS de bitmap (type 3) — tous leurs
                # diagrammes sont vectoriels (paths). On détecte une page
                # avec un diagramme par son nombre de path objects élevé.
                # On rasterise alors la page entière en PNG.
                path_count = sum(1 for o in page.get_objects() if o.type == 1)
                has_bitmap  = any(o.type == 3 for o in page.get_objects())
                is_diagram  = path_count >= VECTOR_PATH_THRESHOLD

                if is_diagram or has_bitmap:
                    try:
                        bitmap  = page.render(scale=config.docling.images_scale)
                        pil_img = bitmap.to_pil()
                        bitmap.close()

                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                        img_b64   = base64.b64encode(img_bytes).decode()

                        placeholder = f"[[FIGURE_{fig_idx}]]"
                        all_figures.append(Figure(
                            placeholder=placeholder,
                            index=fig_idx,
                            document_name=name,
                            page=page_idx + 1,
                            image_bytes=img_bytes,
                            image_base64=img_b64,
                            caption="",
                        ))
                        text_parts.append(f"\n{placeholder}\n")
                        fig_idx += 1
                    except Exception:
                        pass   # rasterisation impossible — skip silencieux

                textpage.close()
                page.close()

            except Exception as exc:
                _log(f"'{name}': WARNING page {page_idx + 1} skippée — {exc}")
                skipped_pages.append(page_idx + 1)

        elapsed = time.monotonic() - t0
        _log(f"'{name}': chunk {chunk_num}/{total_chunks} "
             f"(pages {chunk_start + 1}–{chunk_end}) OK in {elapsed:.1f}s")
        gc.collect()

    doc_pdf.close()

    if skipped_pages:
        _log(f"'{name}': DONE — {len(skipped_pages)} page(s) skippée(s), "
             f"{len(all_figures)} figure(s).")
    else:
        _log(f"'{name}': DONE — {total_pages - first_page} page(s) extraites, "
             f"{len(all_figures)} figure(s).")

    return ParsedDocument(
        name=name,
        text_with_placeholders="".join(text_parts),
        figures=all_figures,
        page_count=total_pages,
        mime_type="application/pdf",
        metadata={
            "accelerator_used": "pypdfium2",
            "parser": "pypdfium2",
            "chunked": True,
            "chunk_size_pages": CHUNK_SIZE_PAGES,
            "chunk_count": total_chunks,
            "skipped_pages": skipped_pages,
            "figures_found": len(all_figures),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Non-PDF — Docling classique (DOCX, PPTX, XLSX, HTML)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_non_pdf_docling(
    content: bytes, name: str, mime: str, config: PipelineConfig
) -> ParsedDocument:
    requested = config.docling.accelerator.lower()
    try:
        return _run_docling(content, name, mime, config, accelerator=requested)
    except Exception as gpu_exc:
        if requested in ("cuda", "auto"):
            warnings.warn(
                f"[doc_pipeline] Docling GPU failed for '{name}' "
                f"({type(gpu_exc).__name__}: {gpu_exc}). Retrying on CPU.",
                RuntimeWarning, stacklevel=2,
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
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    except ImportError as exc:
        raise ImportError("Docling requis : pip install 'docling>=2.0.0'") from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale            = config.docling.images_scale
    pipeline_options.do_ocr                  = bool(config.docling.ocr_enabled)
    pipeline_options.do_table_structure      = False

    _apply_accelerator(pipeline_options, accelerator, config.docling.num_threads)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )

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
        del converter

    doc        = result.document
    text_with_placeholders, figures = _build_docling_text_and_figures(doc, name)
    page_count = getattr(doc, "num_pages", 0) or len(getattr(doc, "pages", []))

    metadata: dict[str, Any] = {"accelerator_used": accelerator, "parser": "docling"}
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
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
    except ImportError:
        return
    device_map: dict[str, Any] = {
        "auto": AcceleratorDevice.AUTO,
        "cuda": AcceleratorDevice.CUDA,
        "cpu":  AcceleratorDevice.CPU,
        "mps":  AcceleratorDevice.MPS,
    }
    try:
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=device_map.get(accelerator, AcceleratorDevice.AUTO),
        )
    except Exception:
        pass


def _build_docling_text_and_figures(doc: Any, document_name: str) -> tuple[str, list[Figure]]:
    try:
        from docling_core.types.doc import PictureItem, TableItem, SectionHeaderItem
    except ImportError:
        return doc.export_to_markdown(), []

    parts:   list[str]    = []
    figures: list[Figure] = []
    fig_idx = 0

    for item, level in doc.iterate_items():
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

        elif isinstance(item, SectionHeaderItem):
            hashes = "#" * max(1, level)
            parts.append(f"\n{hashes} {item.text}\n")

        elif isinstance(item, TableItem):
            try:
                parts.append(f"\n{item.export_to_markdown(doc)}\n")
            except TypeError:
                try:
                    parts.append(f"\n{item.export_to_markdown()}\n")
                except Exception:
                    parts.append("\n[TABLE — could not render]\n")
            except Exception:
                parts.append("\n[TABLE — could not render]\n")

        else:
            text = getattr(item, "text", "") or ""
            if text.strip():
                parts.append(text + "\n")

    return "".join(parts), figures


def _get_pil_image(item: Any, doc: Any) -> Any:
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