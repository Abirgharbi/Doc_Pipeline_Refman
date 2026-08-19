"""
DocumentPipeline — five-stage orchestrator.

Stage order (fixed):
  1. DoclingIngestStage  — parse docs, extract text + figures with [[FIGURE_N]] placeholders
  2. VLMStage            — describe each figure with a vision LLM
  3. RecombineStage      — replace placeholders with VLM descriptions → combined_text
  4. ExtractStage        — LLM extraction on combined_text → structured JSON
  5. OutputStage         — format final result (customisable via renderer= callback)

Customise
---------
  # Swap the final output format only (most common)
  pipeline = DocumentPipeline(config, renderer=my_async_renderer)

  # Replace any full stage
  pipeline = DocumentPipeline(config, output=MyOutputStage())
  pipeline.replace_stage(VLMStage, MyVLMStage())
"""
from __future__ import annotations

import uuid
from typing import Any

from doc_pipeline.core.state import PipelineState
from doc_pipeline.core.config import PipelineConfig

from doc_pipeline.stages.base import BaseStage
from doc_pipeline.stages.ingest import DoclingIngestStage
from doc_pipeline.stages.vlm import VLMStage
from doc_pipeline.stages.recombine import RecombineStage
from doc_pipeline.stages.extract import ExtractStage
from doc_pipeline.stages.output import OutputStage, RendererFn


class DocumentPipeline:

    def __init__(
        self,
        config: PipelineConfig,
        *,
        ingest: BaseStage | None = None,
        vlm: BaseStage | None = None,
        recombine: BaseStage | None = None,
        extract: BaseStage | None = None,
        output: BaseStage | None = None,
        renderer: RendererFn | None = None,   # shortcut for OutputStage only
    ) -> None:
        self.config = config
        self._stages: list[BaseStage] = [
            ingest    or DoclingIngestStage(),
            vlm       or VLMStage(),
            recombine or RecombineStage(),
            extract   or ExtractStage(),
            output    or OutputStage(renderer),
        ]

    async def run(
        self,
        files: list[dict[str, Any]],
        query: str = "",
        session_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> PipelineState:
        """
        Parameters
        ----------
        files : list of {"name": str, "content_base64": str, "mime_type": str}
        query : passed through to ExtractStage and OutputStage
        """
        state = PipelineState(
            query=query,
            raw_files=files,
            session_id=session_id or str(uuid.uuid4()),
            metadata=extra_metadata or {},
        )
        for stage in self._stages:
            state = await stage.run(state, self.config)
        return state

    def replace_stage(self, stage_type: type[BaseStage], new_stage: BaseStage) -> "DocumentPipeline":
        """Swap a stage by type. Returns self for chaining."""
        self._stages = [
            new_stage if isinstance(s, stage_type) else s
            for s in self._stages
        ]
        return self
