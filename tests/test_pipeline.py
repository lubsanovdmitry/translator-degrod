from __future__ import annotations

import asyncio
from pathlib import Path

from semantic_telephone.config import load_config
from semantic_telephone.pipeline import run_pipeline
from semantic_telephone.providers.mock import MockGenerationProvider, MockTranslationProvider
from semantic_telephone.utils.files import read_json


def test_mock_providers_are_deterministic() -> None:
    async def exercise() -> tuple[str, str, str, str]:
        translator = MockTranslationProvider()
        generator = MockGenerationProvider()
        first = await translator.translate("Лира сказала о дороге.", "ru", "de", seed=8)
        second = await translator.translate("Лира сказала о дороге.", "ru", "de", seed=8)
        generated_a = await generator.generate("<<<TEXT>>>\nТекст без точки", temperature=0.4, seed=8)
        generated_b = await generator.generate("<<<TEXT>>>\nТекст без точки", temperature=0.4, seed=8)
        return first.text, second.text, generated_a.text, generated_b.text

    first, second, generated_a, generated_b = asyncio.run(exercise())
    assert first == second
    assert generated_a == generated_b


def test_mock_pipeline_integration(config_file: Path) -> None:
    directory = asyncio.run(run_pipeline(load_config(config_file)))
    assert (directory / "final.txt").read_text(encoding="utf-8").strip()
    assert (directory / "report.md").exists()
    assert (directory / "events.jsonl").exists()
    manifest = read_json(directory / "manifest.json")
    assert manifest["state"] == "completed"
    assert manifest["deterministic"] is True
    assert len(list((directory / "chunks").glob("*/stage-*.json"))) >= 3


def test_pipeline_seed_reproduces_output(config_file: Path) -> None:
    config = load_config(config_file)
    first = asyncio.run(run_pipeline(config))
    second = asyncio.run(run_pipeline(config))
    assert (first / "final.txt").read_text(encoding="utf-8") == (
        second / "final.txt"
    ).read_text(encoding="utf-8")

