"""
ExtractStage — run LLM extraction on each document's combined_text.

combined_text is produced by RecombineStage and contains:
  - The full document text
  - Inline [FIGURE DESCRIPTION] blocks at each figure's original position

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


class ExtractStage(BaseStage):

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if config.extraction is None:
            state.extractions = [
                ExtractedData(document_name=d.name, data={}, raw_text=d.combined_text or d.text_with_placeholders)
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
            # Prefer combined_text (with VLM descriptions); fall back to raw text
            source_text = doc.combined_text or doc.text_with_placeholders

            if not source_text.strip():
                return ExtractedData(
                    document_name=doc.name,
                    data={},
                    raw_text="",
                    error="empty document after ingestion",
                )

            schema_str = json.dumps(ec.schema_definition, indent=2)
            prompt = ec.user_prompt_template.format(
                document_text=source_text[:12_000],   # guard against very large docs
                schema=schema_str,
            )

            try:
                raw_response = await llm.complete(system=ec.system_prompt, user=prompt)
                data = _parse_json(raw_response)
                return ExtractedData(document_name=doc.name, data=data, raw_text=source_text)

            except json.JSONDecodeError as exc:
                if ec.fallback_on_error:
                    return ExtractedData(
                        document_name=doc.name,
                        data={},
                        raw_text=source_text,
                        error=f"JSON parse failed: {exc}",
                    )
                raise

            except Exception as exc:
                if ec.fallback_on_error:
                    return ExtractedData(
                        document_name=doc.name,
                        data={},
                        raw_text=source_text,
                        error=str(exc),
                    )
                raise


def _parse_json(text: str) -> dict:
    """Strip markdown code fences the LLM may have added, then parse."""
    text = text.strip()
    fenced = re.match(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)
