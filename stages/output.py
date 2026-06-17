"""
OutputStage — format the final result.

Supply a `renderer` async callable to produce any output shape you want:
  - markdown string (default)
  - dict / JSON
  - file path
  - database write confirmation
  - anything
"""
from __future__ import annotations
import json
from typing import Callable, Awaitable, Any

from doc_pipeline.stages.base import BaseStage
from doc_pipeline.core.state import PipelineState
from doc_pipeline.core.config import PipelineConfig

RendererFn = Callable[[PipelineState, PipelineConfig], Awaitable[Any]]


class OutputStage(BaseStage):

    def __init__(self, renderer: RendererFn | None = None) -> None:
        self._renderer = renderer or _default_renderer

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        state.output = await self._renderer(state, config)
        return state


async def _default_renderer(state: PipelineState, config: PipelineConfig) -> str:
    """Default: one markdown section per document showing figure count + extracted JSON."""
    lines = ["# Pipeline Output\n"]
    if state.query:
        lines.append(f"**Query:** {state.query}\n")

    for doc, ext in zip(state.documents, state.extractions):
        lines.append(f"## {doc.name}")
        lines.append(
            f"Pages: {doc.page_count}   "
            f"Figures extracted: {len(doc.figures)}   "
            f"Figures with VLM description: {sum(1 for f in doc.figures if f.vlm_description)}"
        )

        if doc.figures:
            lines.append("\n**Figures**")
            for fig in doc.figures:
                desc_preview = (fig.vlm_description[:120] + "…") if len(fig.vlm_description) > 120 else fig.vlm_description
                lines.append(
                    f"- {fig.placeholder}  page {fig.page}"
                    + (f"  caption: _{fig.caption}_" if fig.caption else "")
                    + (f"\n  VLM: {desc_preview}" if desc_preview else "")
                )

        lines.append("\n**Extracted Data**")
        if ext.error:
            lines.append(f"> Extraction error: {ext.error}\n")
        else:
            lines.append(f"```json\n{json.dumps(ext.data, indent=2, ensure_ascii=False)}\n```\n")

    return "\n".join(lines)
