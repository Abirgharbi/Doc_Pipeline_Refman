"""
VLMStage — describe every extracted figure using a Vision-Language Model.

Uses Ollama with a vision-capable model (llava, minicpm-v, llama3.2-vision, …).
Each Figure.vlm_description is filled in-place; the documents list is not replaced.

Skipped entirely when config.vlm.enabled is False or there are no figures.
"""
from __future__ import annotations
import asyncio

from doc_pipeline.stages.base import BaseStage
from doc_pipeline.core.state import PipelineState, Figure
from doc_pipeline.core.config import PipelineConfig


class VLMStage(BaseStage):

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if not config.vlm.enabled:
            return state

        all_figures: list[Figure] = [
            fig
            for doc in state.documents
            for fig in doc.figures
            if fig.image_base64   # skip figures with no image bytes
        ]

        if not all_figures:
            return state

        sem = asyncio.Semaphore(config.vlm.concurrency)
        tasks = [self._describe(fig, config, sem) for fig in all_figures]
        await asyncio.gather(*tasks)

        return state

    async def _describe(
        self, figure: Figure, config: PipelineConfig, sem: asyncio.Semaphore
    ) -> None:
        async with sem:
            try:
                import ollama
            except ImportError as exc:
                raise ImportError(
                    "Ollama Python client not installed. Run: pip install ollama"
                ) from exc

            client = ollama.AsyncClient(host=config.vlm.base_url)

            # Build the context line shown before the prompt
            context = ""
            if figure.caption:
                context = f"Figure caption: {figure.caption}\n\n"
            if figure.page:
                context += f"(Page {figure.page} of the document)\n\n"

            try:
                response = await client.chat(
                    model=config.vlm.model,
                    messages=[
                        {
                            "role": "user",
                            "content": context + config.vlm.prompt,
                            "images": [figure.image_base64],
                        }
                    ],
                )
                figure.vlm_description = response["message"]["content"].strip()
            except Exception as exc:
                # Never crash the pipeline — store error as description
                figure.vlm_description = f"[VLM error for {figure.placeholder}: {exc}]"
                if hasattr(exc, '__class__'):
                    pass  # logged via the description text above
