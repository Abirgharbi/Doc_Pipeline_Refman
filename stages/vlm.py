"""
VLMStage — describe every extracted figure using a Vision-Language Model.

Uses Ollama with a vision-capable model (llava, minicpm-v, llama3.2-vision, …).
Each Figure.vlm_description is filled in-place; the documents list is not replaced.

Skipped entirely when config.vlm.enabled is False or there are no figures.

Reliability additions:
  - A hard timeout per figure (VLM_TIMEOUT_SECONDS) so a single slow/stuck Ollama
    call can never hang the whole pipeline indefinitely — on timeout the figure
    gets a placeholder description and processing moves on.
  - Live progress logging ("figure 47/1173 done in 8.2s") since with concurrency=1
    and hundreds/thousands of figures, the stage can otherwise look silent/frozen
    for hours with zero visibility.
"""
from __future__ import annotations
import asyncio
import time

from doc_pipeline.stages.base import BaseStage
from doc_pipeline.core.state import PipelineState, Figure
from doc_pipeline.core.config import PipelineConfig


# ── Tunables ─────────────────────────────────────────────────────────────────
VLM_TIMEOUT_SECONDS = 120   # hard timeout per figure — tune based on observed speed


def _log(msg: str) -> None:
    print(f"[vlm] {msg}", flush=True)


class VLMStage(BaseStage):

    async def run(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        if not config.vlm.enabled:
            _log("VLM disabled in config — skipping figure descriptions.")
            return state

        all_figures: list[Figure] = [
            fig
            for doc in state.documents
            for fig in doc.figures
            if fig.image_base64   # skip figures with no image bytes
        ]

        if not all_figures:
            _log("No figures with image data found — nothing to describe.")
            return state

        total = len(all_figures)
        _log(f"Starting VLM description of {total} figure(s) "
             f"(model={config.vlm.model}, concurrency={config.vlm.concurrency}, "
             f"timeout={VLM_TIMEOUT_SECONDS}s/figure)")

        sem = asyncio.Semaphore(config.vlm.concurrency)
        progress = {"done": 0, "errors": 0, "timeouts": 0}
        progress_lock = asyncio.Lock()
        t_start = time.monotonic()

        tasks = [
            self._describe(fig, config, sem, progress, progress_lock, total, t_start)
            for fig in all_figures
        ]
        await asyncio.gather(*tasks)

        elapsed = time.monotonic() - t_start
        _log(f"VLM stage DONE — {total} figure(s) in {elapsed:.1f}s "
             f"({progress['errors']} error(s), {progress['timeouts']} timeout(s))")

        return state

    async def _describe(
        self,
        figure: Figure,
        config: PipelineConfig,
        sem: asyncio.Semaphore,
        progress: dict,
        progress_lock: asyncio.Lock,
        total: int,
        t_start: float,
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

            t0 = time.monotonic()
            status = "ok"

            try:
                response = await asyncio.wait_for(
                    client.chat(
                        model=config.vlm.model,
                        messages=[
                            {
                                "role": "user",
                                "content": context + config.vlm.prompt,
                                "images": [figure.image_base64],
                            }
                        ],
                    ),
                    timeout=VLM_TIMEOUT_SECONDS,
                )
                figure.vlm_description = response["message"]["content"].strip()

            except asyncio.TimeoutError:
                figure.vlm_description = (
                    f"[VLM timeout for {figure.placeholder} "
                    f"after {VLM_TIMEOUT_SECONDS}s — description skipped]"
                )
                status = "timeout"

            except Exception as exc:
                # Never crash the pipeline — store error as description
                figure.vlm_description = f"[VLM error for {figure.placeholder}: {exc}]"
                status = "error"

            elapsed_this = time.monotonic() - t0

            async with progress_lock:
                progress["done"] += 1
                if status == "timeout":
                    progress["timeouts"] += 1
                elif status == "error":
                    progress["errors"] += 1

                done = progress["done"]
                total_elapsed = time.monotonic() - t_start
                avg = total_elapsed / done if done else 0
                remaining = total - done
                eta_sec = avg * remaining

                tag = "OK" if status == "ok" else status.upper()
                _log(
                    f"figure {done}/{total} ({figure.document_name} p.{figure.page}) "
                    f"{tag} in {elapsed_this:.1f}s — avg {avg:.1f}s/fig, "
                    f"ETA {eta_sec/60:.1f} min for remaining {remaining}"
                )