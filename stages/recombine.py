"""
RecombineStage — splice VLM figure descriptions back into the document text.

For each document:
  - Takes  : text_with_placeholders  (contains [[FIGURE_N]] markers)
  - Fills  : combined_text           (placeholders replaced by VLM descriptions)

The combined_text is what the LLM extraction stage reads.
Figures appear in the text at their original position, preserving document reading order.
"""
from __future__ import annotations

from doc_pipeline.stages.base import BaseStage
from doc_pipeline.core.state import PipelineState, Figure
from doc_pipeline.core.config import PipelineConfig


_HEADER = "\n--- [FIGURE DESCRIPTION"
_FOOTER = "] ---\n"
_END    = "\n--- [END FIGURE] ---\n"


class RecombineStage(BaseStage):

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        for doc in state.documents:
            text = doc.text_with_placeholders
            for fig in doc.figures:
                text = text.replace(fig.placeholder, _render_figure(fig))
            doc.combined_text = text
        return state


def _render_figure(fig: Figure) -> str:
    """
    Format a figure description block that will appear inline in the combined text.
    Example output:

        --- [FIGURE DESCRIPTION — page 3, Figure 1] ---
        This is a bar chart showing quarterly revenue...
        --- [END FIGURE] ---
    """
    location = []
    if fig.page:
        location.append(f"page {fig.page}")
    location.append(f"Figure {fig.index + 1}")

    header = f"{_HEADER} — {', '.join(location)}{_FOOTER}"

    lines = [header]

    if fig.caption:
        lines.append(f"Caption: {fig.caption}\n")

    if fig.vlm_description:
        lines.append(fig.vlm_description)
    else:
        lines.append("[No VLM description available]")

    lines.append(_END)
    return "\n".join(lines)
