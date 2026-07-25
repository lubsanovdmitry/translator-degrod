from __future__ import annotations

import asyncio
from pathlib import Path

from semantic_telephone.checkpoints import CheckpointStore
from semantic_telephone.config import load_config
from semantic_telephone.models import StageResult
from semantic_telephone.pipeline import run_pipeline
from semantic_telephone.utils.hashing import checksum_text


def _result() -> StageResult:
    return StageResult(
        input_text="a",
        output_text="b",
        stage_type="translation_cycle",
        provider="mock",
        model="mock",
        source_language="en",
        target_language="ru",
        route=["en", "ru"],
        duration_seconds=0.1,
        attempts=1,
        warnings=[],
        error=None,
        usage=None,
        input_checksum=checksum_text("a"),
        output_checksum=checksum_text("b"),
        stage_checksum="checkpoint",
    )


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.save(1, "translation_cycle", _result())
    loaded = store.load(1, "translation_cycle", "checkpoint")
    assert loaded is not None
    assert loaded.output_text == "b"


def test_resume_reuses_completed_stages(config_file: Path) -> None:
    config = load_config(config_file)
    directory = asyncio.run(run_pipeline(config))
    events_before = (directory / "events.jsonl").read_text(encoding="utf-8")
    asyncio.run(run_pipeline(config, run_directory=directory))
    events_after = (directory / "events.jsonl").read_text(encoding="utf-8")
    assert len(events_after) > len(events_before)
    assert "checkpoint_reused" in events_after

