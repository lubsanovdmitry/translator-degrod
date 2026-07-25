from __future__ import annotations

from dataclasses import dataclass

from ..models import PipelineStageConfig


@dataclass(slots=True)
class StageContext:
    chunk_index: int
    stage_index: int
    seed: int
    config: PipelineStageConfig

