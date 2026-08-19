"""
ExtractStage — run LLM extraction on each document's combined_text.

combined_text is produced by RecombineStage and contains:
  - The full document text
  - Inline [FIGURE DESCRIPTION] blocks at each figure's original position

For large documents (STM32 reference manuals etc.), the combined_text far
exceeds any LLM context window. This stage therefore splits the text into
overlapping chunks of TEXT_CHUNK_CHARS characters, runs extraction on each
chunk independently, and merges the resulting lists (peripherals, registers,
tables, …) into a single ExtractedData per document.

Skipped entirely when config.extraction is None.
"""
from __future__ import annotations
import asyncio
import json
import re
from doc_pipeline.stages.base import BaseStage
from doc_pipeline.core.state import PipelineState, ParsedDocument, ExtractedData
from doc_pipeline.core.config import PipelineConfig
from doc_pipeline.providers.llm import get_llm_provider, BaseLLMProvider


# ── Tunables ──────────────────────────────────────────────────────────────────
TEXT_CHUNK_CHARS = 8_000    # characters per LLM call (~2000 tokens for English text)
CHUNK_OVERLAP    = 500      # overlap between consecutive chunks to avoid cutting
                             # a register definition mid-sentence


def _log(msg: str) -> None:
    print(f"[extract] {msg}", flush=True)


class ExtractStage(BaseStage):

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if config.extraction is None:
            state.extractions = [
                ExtractedData(
                    document_name=d.name,
                    data={},
                    raw_text=d.combined_text or d.text_with_placeholders,
                )
                for d in state.documents
            ]
            return state

        llm = get_llm_provider(config.llm)
        sem = asyncio.Semaphore(config.max_concurrency)
        tasks = [self._extract_one(doc, llm, config, sem) for doc in state.documents]
        state.extractions = await asyncio.gather(*tasks)
        return state

    async def _extract_one(
        self,
        doc: ParsedDocument,
        llm: BaseLLMProvider,
        config: PipelineConfig,
        sem: asyncio.Semaphore,
    ) -> ExtractedData:
        async with sem:
            ec = config.extraction
            source_text = doc.combined_text or doc.text_with_placeholders

            if not source_text.strip():
                return ExtractedData(
                    document_name=doc.name,
                    data={},
                    raw_text="",
                    error="empty document after ingestion",
                )

            # Split into overlapping chunks
            chunks = _split_text(source_text, TEXT_CHUNK_CHARS, CHUNK_OVERLAP)
            total_chunks = len(chunks)
            _log(f"'{doc.name}': {len(source_text):,} chars -> {total_chunks} extraction chunk(s)")

            schema_str = json.dumps(ec.schema_definition, indent=2)
            all_results: list[dict] = []

            for chunk_num, chunk_text in enumerate(chunks, start=1):
                _log(f"'{doc.name}': extracting chunk {chunk_num}/{total_chunks} "
                     f"({len(chunk_text):,} chars)...")

                prompt = ec.user_prompt_template.format(
                    document_text=chunk_text,
                    schema=schema_str,
                )

                try:
                    raw_response = await llm.complete(
                        system=ec.system_prompt, user=prompt
                    )
                    chunk_data = _parse_json(raw_response)
                    if chunk_data:
                        all_results.append(chunk_data)
                        _log(f"'{doc.name}': chunk {chunk_num}/{total_chunks} OK — "
                             f"keys: {list(chunk_data.keys())}")
                    else:
                        _log(f"'{doc.name}': chunk {chunk_num}/{total_chunks} — "
                             f"empty response from LLM (skipped)")

                except json.JSONDecodeError as exc:
                    _log(f"'{doc.name}': chunk {chunk_num}/{total_chunks} — "
                         f"JSON parse error: {exc} (skipped)")
                except Exception as exc:
                    _log(f"'{doc.name}': chunk {chunk_num}/{total_chunks} — "
                         f"LLM error: {exc} (skipped)")

            if not all_results:
                error_msg = "All extraction chunks returned empty or invalid JSON"
                _log(f"'{doc.name}': FAILED — {error_msg}")
                if ec.fallback_on_error:
                    return ExtractedData(
                        document_name=doc.name,
                        data={},
                        raw_text=source_text,
                        error=error_msg,
                    )
                raise ValueError(error_msg)

            # Merge all chunk results into one dict
            merged = _merge_results(all_results, ec.schema_definition)
            _log(f"'{doc.name}': DONE — merged {len(all_results)} chunk(s), "
                 f"final keys: { {k: len(v) if isinstance(v, list) else v for k, v in merged.items()} }")

            return ExtractedData(
                document_name=doc.name,
                data=merged,
                raw_text=source_text,
            )


# ── Text splitting ────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split `text` into chunks of at most `chunk_size` characters, with `overlap`
    characters of context carried over from the previous chunk.
    Tries to split at paragraph/newline boundaries to avoid cutting mid-sentence.
    """
    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Try to snap to a paragraph boundary within the last 20% of the chunk
        if end < length:
            snap_start = start + int(chunk_size * 0.8)
            boundary = text.rfind("\n\n", snap_start, end)
            if boundary == -1:
                boundary = text.rfind("\n", snap_start, end)
            if boundary != -1:
                end = boundary

        chunks.append(text[start:end])
        if end >= length:
            break
        start = max(end - overlap, start + 1)   # advance, always making progress

    return chunks


# ── Result merging ────────────────────────────────────────────────────────────

def _merge_results(results: list[dict], schema: dict) -> dict:
    """
    Merge multiple per-chunk extraction dicts into one.

    Rules:
      - List fields (peripherals, registers, tables, …): concatenate all items,
        de-duplicate by 'name' key where present.
      - String fields (figure_insights, summary, …): join non-empty values with
        a newline separator.
      - Missing keys in a chunk are skipped silently.
    """
    merged: dict = {}

    # Identify which top-level schema keys are lists vs strings
    list_keys = {k for k, v in schema.items() if isinstance(v, list)}
    str_keys  = {k for k, v in schema.items() if isinstance(v, str)}

    for key in list_keys:
        seen_names: set = set()
        merged[key] = []
        for result in results:
            for item in result.get(key, []):
                # De-duplicate by 'name' field if present
                item_name = item.get("name", "") if isinstance(item, dict) else str(item)
                if item_name and item_name in seen_names:
                    continue
                if item_name:
                    seen_names.add(item_name)
                merged[key].append(item)

    for key in str_keys:
        parts = [r[key] for r in results if r.get(key, "").strip()]
        merged[key] = "\n---\n".join(parts) if parts else ""

    # Pass through any keys not in the schema (LLM added extras)
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    for key in all_keys - list_keys - str_keys:
        values = [r[key] for r in results if key in r]
        merged[key] = values[-1] if values else None

    return merged


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """
    Extract a JSON object from LLM output that may contain:
    - Markdown code fences (```json ... ```)
    - Prose before/after the JSON ("Here is the JSON: {...}")
    - Empty string (model refused to answer)
    Returns {} if no valid JSON object is found.
    """
    text = text.strip()
    if not text:
        return {}

    # 1. Try to extract from markdown code fences first
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 2. Try to find a JSON object anywhere in the text (handles "Here is the JSON: {...}")
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try to find the largest valid JSON object by scanning from each {
            for match in re.finditer(r"\{", text):
                for end in range(len(text), match.start(), -1):
                    try:
                        return json.loads(text[match.start():end])
                    except json.JSONDecodeError:
                        continue

    # 3. Last resort — try the whole text as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return {}