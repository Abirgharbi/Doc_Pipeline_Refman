from __future__ import annotations

import asyncio
import base64
import json
import warnings
from pathlib import Path

# Silence the noisy (and harmless) docling deprecation warning spam so the
# real progress logs from ingest.py are easy to read in the console.
warnings.filterwarnings("ignore", message=".*export_to_markdown.*")

from doc_pipeline.core.pipeline import DocumentPipeline
from doc_pipeline.core.config import (
    PipelineConfig,
    LLMConfig,
    VLMConfig,
    DoclingConfig,
    ExtractionConfig,
)
from doc_pipeline.core.state import PipelineState


REFMAN_SCHEMA = {
    "peripherals": [
        {
            "name": "string",
            "description": "string",
            "base_address": "string"
        }
    ],
    "registers": [
        {
            "name": "string",
            "offset": "string",
            "reset_value": "string",
            "description": "string",
            "fields": [
                {
                    "name": "string",
                    "bits": "string",
                    "access": "string",
                    "description": "string"
                }
            ]
        }
    ],
    "tables": [
        {
            "title": "string",
            "content": "string"
        }
    ],
    "figure_insights": "string"
}

OUTPUT_PATH = Path("stm32_refman_output.json")
PROGRESS_PATH = Path("stm32_refman_progress.json")


async def refman_renderer(state: PipelineState, config: PipelineConfig):
    results = []
    for ext in state.extractions:
        results.append({
            "document": ext.document_name,
            "data": ext.data,
            "error": ext.error,
        })
    return results


def build_refman_pipeline() -> DocumentPipeline:
    config = PipelineConfig(
        docling=DoclingConfig(
            images_scale=1.0,      # smaller figure images -> much less RAM per chunk
            ocr_enabled=False,     # refman PDFs have native text, OCR is unnecessary
            accelerator="cpu",     # avoids GPU/CPU conflicts mid-run; Docling stays on CPU
            num_threads=4,
        ),
        vlm=VLMConfig(
            model="moondream",
            base_url="http://localhost:11434",
            prompt=(
                "This image comes from an STM32 reference manual. "
                "Describe diagrams, timing diagrams, register tables, "
                "clock trees and bit fields precisely."
            ),
            concurrency=1,
        ),
        llm=LLMConfig(
            provider="ollama",
            model="mistral",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=4096,
        ),
        extraction=ExtractionConfig(
            schema_definition=REFMAN_SCHEMA,
            system_prompt=(
                "You are a JSON-only data extraction assistant for STM32 reference manuals. "
                "You MUST always respond with a single valid JSON object and nothing else. "
                "No explanations, no markdown, no prose. Only JSON. "
                "If the text contains no relevant STM32 data, respond with exactly: "
                "{\"peripherals\":[],\"registers\":[],\"tables\":[],\"figure_insights\":\"\"}"
            ),
            user_prompt_template=(
                "Extract STM32 peripherals, registers, and tables from the text below.\n"
                "Respond with ONLY a JSON object matching this schema (no other text):\n"
                "{schema}\n\n"
                "Text:\n{document_text}"
            ),
        ),
        max_concurrency=1,   # one PDF (and one chunk) at a time — no RAM doubling
    )
    return DocumentPipeline(config, renderer=refman_renderer)


def get_refman_paths() -> list[Path]:
    refman_dir = Path(__file__).parent.parent / "refman"
    paths = sorted(refman_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDF found in {refman_dir}")
    return paths


def load_one_pdf(path: Path) -> dict:
    """Load a SINGLE PDF into memory — released right after it's processed."""
    with open(path, "rb") as f:
        return {
            "name": path.name,
            "content_base64": base64.b64encode(f.read()).decode(),
            "mime_type": "application/pdf",
        }


def load_progress() -> dict:
    """Resume support: which PDFs were already fully processed in a previous run."""
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"done": []}
    return {"done": []}


def save_progress(progress: dict) -> None:
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def load_existing_results() -> list:
    """Resume support: results already written from a previous run."""
    if OUTPUT_PATH.exists():
        try:
            return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_results(results: list) -> None:
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


async def main():
    paths = get_refman_paths()
    pipeline = build_refman_pipeline()

    progress = load_progress()
    done_names = set(progress.get("done", []))
    all_results = load_existing_results()

    if done_names:
        print(f"Resuming: {len(done_names)} PDF(s) already completed previously:")
        for n in done_names:
            print(f"  - {n}")
        print()

    remaining = [p for p in paths if p.name not in done_names]
    print(f"Found {len(paths)} PDF(s) total, {len(remaining)} remaining to process.\n")

    if not remaining:
        print("Nothing left to do — all PDFs already processed.")
        print(f"Results are in {OUTPUT_PATH}")
        return

    for i, pdf_path in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] Processing {pdf_path.name} ...", flush=True)

        file_entry = load_one_pdf(pdf_path)

        try:
            state = await pipeline.run(
                [file_entry],
                query=(
                    "Extract all STM32 peripherals, registers, "
                    "bit fields, reset values and tables."
                )
            )
            if state.errors:
                print(f"  WARNING non-fatal errors: {state.errors}")

            doc_meta = state.documents[0].metadata if state.documents else {}
            skipped = doc_meta.get("skipped_chunks", [])
            if skipped:
                print(f"  NOTE: {len(skipped)} chunk(s) were skipped (timeout/crash): {skipped}")

            all_results.extend(state.output)
            n_figs = len(state.documents[0].figures) if state.documents else 0
            print(f"  DONE — {n_figs} figure(s) found.\n", flush=True)

            # Mark this PDF as fully done, persist immediately.
            done_names.add(pdf_path.name)

        except Exception as exc:
            print(f"  FAILED: {exc}\n", flush=True)
            all_results.append({"document": pdf_path.name, "data": {}, "error": str(exc)})
            # Do NOT mark as done — a future run will retry this PDF.

        # Persist after EVERY PDF (success or failure) so a crash/shutdown
        # never loses more than the current PDF's work.
        save_results(all_results)
        save_progress({"done": sorted(done_names)})

        del file_entry
        print(f"  Progress saved -> {OUTPUT_PATH} / {PROGRESS_PATH}\n", flush=True)

    print(f"All done. Final results in {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())