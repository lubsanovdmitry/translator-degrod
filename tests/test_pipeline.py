from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from semantic_telephone.chunking import chunk_text
from semantic_telephone.config import load_config
from semantic_telephone.models import TranslationResult
from semantic_telephone.pipeline import _chunk_context_value, run_pipeline
from semantic_telephone.providers.mock import MockGenerationProvider, MockTranslationProvider
from semantic_telephone.reporting import regenerate_report
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


def test_mock_generation_uses_context_and_extracts_repeated_variants() -> None:
    async def exercise() -> tuple[str, str, str]:
        generator = MockGenerationProvider()
        plain = await generator.generate(
            "<<<TEXT>>>\nЛира вошла. Лира села.",
            temperature=0.4,
            seed=8,
        )
        contextual = await generator.generate(
            "ПРЕДЫДУЩИЙ ПОВРЕЖДЁННЫЙ КОНТЕКСТ:\nЛира ждала.\n\n"
            "<<<TEXT>>>\nЛира вошла. Лира села.",
            temperature=0.4,
            seed=8,
        )
        memory = await generator.generate(
            "<<<TEXT>>>\nЛира вошла. Лира села.",
            temperature=0.1,
            seed=8,
            response_format="json",
        )
        return plain.text, contextual.text, memory.text

    plain, contextual, memory = asyncio.run(exercise())
    assert plain != contextual
    assert '"entity_key": "variant:лира"' in memory


def test_mock_pipeline_integration(config_file: Path) -> None:
    directory = asyncio.run(run_pipeline(load_config(config_file)))
    assert (directory / "final.txt").read_text(encoding="utf-8").strip()
    assert (directory / "report.md").exists()
    assert (directory / "events.jsonl").exists()
    manifest = read_json(directory / "manifest.json")
    assert manifest["state"] == "completed"
    assert manifest["deterministic"] is True
    assert len(list((directory / "chunks").glob("*/stage-*.json"))) >= 3
    regenerate_report(directory)
    assert "final text already declared in target language" in (
        directory / "report.md"
    ).read_text(encoding="utf-8")


def test_pipeline_seed_reproduces_output(config_file: Path) -> None:
    config = load_config(config_file)
    first = asyncio.run(run_pipeline(config))
    config.runtime.concurrency = 3
    second = asyncio.run(run_pipeline(config))
    assert (first / "final.txt").read_text(encoding="utf-8") == (
        second / "final.txt"
    ).read_text(encoding="utf-8")


def test_context_source_can_include_intermediate_generations(config_file: Path) -> None:
    config = load_config(config_file)
    config.context.source = "all_generated_text"
    value = _chunk_context_value("final", ["first", "second"], config)
    assert value == "first\n\nsecond\n\nfinal"


def test_independent_chunks_use_bounded_concurrency(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackingProvider:
        active = 0
        maximum_active = 0

        async def translate(
            self,
            text: str,
            source_language: str,
            target_language: str,
            *,
            seed: int | None = None,
        ) -> TranslationResult:
            del source_language, target_language, seed
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            try:
                await asyncio.sleep(0.01)
                return TranslationResult(
                    text=text,
                    provider="tracking",
                    model="tracking",
                    deterministic=True,
                )
            finally:
                self.active -= 1

    config = load_config(config_file)
    config.runtime.concurrency = 2
    config.pipeline = [config.pipeline[0]]
    provider = TrackingProvider()
    monkeypatch.setattr(
        "semantic_telephone.pipeline.translation_provider",
        lambda _: provider,
    )

    directory = asyncio.run(run_pipeline(config))

    assert provider.maximum_active == 2
    manifest = read_json(directory / "manifest.json")
    assert manifest["effective_chunk_concurrency"] == 2
    assert manifest["processed_chunks"] == [
        chunk.chunk_id for chunk in chunk_text(config_file.parent.joinpath("input.txt").read_text(), config.chunking)
    ]


def test_context_forces_sequential_chunk_execution(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackingProvider:
        active = 0
        maximum_active = 0

        async def translate(
            self,
            text: str,
            source_language: str,
            target_language: str,
            *,
            seed: int | None = None,
        ) -> TranslationResult:
            del source_language, target_language, seed
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            try:
                await asyncio.sleep(0.005)
                return TranslationResult(
                    text=text,
                    provider="tracking",
                    model="tracking",
                    deterministic=True,
                )
            finally:
                self.active -= 1

    config = load_config(config_file)
    config.runtime.concurrency = 3
    config.context.enabled = True
    config.pipeline = [config.pipeline[0]]
    provider = TrackingProvider()
    monkeypatch.setattr(
        "semantic_telephone.pipeline.translation_provider",
        lambda _: provider,
    )

    directory = asyncio.run(run_pipeline(config))

    assert provider.maximum_active == 1
    manifest = read_json(directory / "manifest.json")
    assert manifest["effective_chunk_concurrency"] == 1


def test_parallel_stop_policy_cancels_peer_chunks(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProvider:
        cancelled = 0

        async def translate(
            self,
            text: str,
            source_language: str,
            target_language: str,
            *,
            seed: int | None = None,
        ) -> TranslationResult:
            del source_language, target_language, seed
            if "Второй" in text:
                raise RuntimeError("planned provider failure")
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
            return TranslationResult(text=text, provider="slow", model="slow")

    config = load_config(config_file)
    config.runtime.concurrency = 2
    config.runtime.retries = 1
    config.pipeline = [config.pipeline[0]]
    provider = FailingProvider()
    monkeypatch.setattr(
        "semantic_telephone.pipeline.translation_provider",
        lambda _: provider,
    )

    with pytest.raises(RuntimeError, match="planned provider failure"):
        asyncio.run(run_pipeline(config))

    assert provider.cancelled >= 1
