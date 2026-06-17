"""Abstract base for every pipeline stage."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doc_pipeline.core.state import PipelineState
    from doc_pipeline.core.config import PipelineConfig


class BaseStage(ABC):
    """
    A stage receives the current PipelineState, mutates it, and returns it.
    Never replace state; only add to it — prior stages' output must remain intact.
    """

    @abstractmethod
    async def run(self, state: "PipelineState", config: "PipelineConfig") -> "PipelineState":
        ...
