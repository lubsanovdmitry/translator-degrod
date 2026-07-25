from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pytest

from semantic_telephone.config import load_config
from semantic_telephone.models import EngineRoutingConfig, ProviderConfig, TranslationResult
from semantic_telephone.pipeline import run_pipeline
from semantic_telephone.providers.router import TranslationProviderRouter
from semantic_telephone.stages.translation import translate_route


@dataclass
class FakeTranslationProvider:
    name: str
    fail: bool = False

    def supports_pair(self, source: str, target: str) -> bool:
        return source != target

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None = None,
    ) -> TranslationResult:
        if self.fail:
            raise RuntimeError("planned failure")
        return TranslationResult(
            text=f"{text}|{self.name}:{source_language}-{target_language}",
            provider=self.name,
            model=f"{self.name}-model",
            deterministic=True,
        )


def _providers() -> dict[str, FakeTranslationProvider]:
    return {
        "nllb": FakeTranslationProvider("nllb"),
        "m2m": FakeTranslationProvider("m2m100"),
        "opus": FakeTranslationProvider("opus_mt"),
    }


def test_weighted_route_is_seeded_and_alternating_avoids_repeats() -> None:
    route = ["ru", "en", "de", "tr", "ru"]
    weighted = TranslationProviderRouter(
        _providers(),
        default_provider="nllb",
        routing=EngineRoutingConfig(
            mode="weighted_random",
            weights={"nllb": 0.5, "m2m": 0.3, "opus": 0.2},
        ),
    )
    assert weighted.plan_route(route, seed=17) == weighted.plan_route(route, seed=17)

    alternating = TranslationProviderRouter(
        _providers(),
        default_provider="nllb",
        routing=EngineRoutingConfig(mode="alternating"),
    )
    selected = [candidates[0] for candidates in alternating.plan_route(route, seed=17)]
    assert all(left != right for left, right in pairwise(selected))


def test_fixed_and_heterogeneous_modes() -> None:
    route = ["ru", "en", "de", "ru"]
    fixed = TranslationProviderRouter(
        _providers(),
        default_provider="nllb",
        routing=EngineRoutingConfig(mode="fixed_engine_route", route=["opus", "m2m", "nllb"]),
    )
    assert [item[0] for item in fixed.plan_route(route, seed=1)] == [
        "opus",
        "m2m",
        "nllb",
    ]

    heterogeneous = TranslationProviderRouter(
        _providers(),
        default_provider="nllb",
        routing=EngineRoutingConfig(mode="heterogeneous"),
    )
    selected = [item[0] for item in heterogeneous.plan_route(route, seed=1)]
    assert len(set(selected)) == 3


async def test_quality_fallback_records_engine_route() -> None:
    providers = _providers()
    providers["opus"].fail = True
    router = TranslationProviderRouter(
        providers,
        default_provider="nllb",
        routing=EngineRoutingConfig(
            mode="quality_fallback",
            fallback_order=["opus", "nllb", "m2m"],
        ),
    )
    output, results = await translate_route(router, "text", ["ru", "en", "ru"], seed=4)
    assert output.endswith("|nllb:en-ru")
    assert [result.provider for result in results] == ["nllb", "nllb"]
    assert all("quality fallback" in " ".join(result.warnings) for result in results)
    assert results[0].metadata["candidate_order"] == ["opus", "nllb", "m2m"]


async def test_cache_sensitive_provider_serializes_preflight_and_entire_route() -> None:
    class CacheSensitiveProvider:
        name = "cache-sensitive"
        requires_route_serialization = True

        def __init__(self) -> None:
            self.operations: list[tuple[str, str, str]] = []

        def supports_pair(self, source: str, target: str) -> bool:
            return source != target

        async def prepare_pair(self, source: str, target: str) -> None:
            self.operations.append(("prepare", source, target))
            await asyncio.sleep(0)

        async def translate(
            self,
            text: str,
            source_language: str,
            target_language: str,
            *,
            seed: int | None = None,
        ) -> TranslationResult:
            del seed
            self.operations.append(("translate", source_language, target_language))
            await asyncio.sleep(0)
            return TranslationResult(
                text=text,
                provider=self.name,
                model=self.name,
                deterministic=True,
            )

    cache_sensitive = CacheSensitiveProvider()
    router = TranslationProviderRouter(
        {"cache": cache_sensitive},
        default_provider="cache",
        routing=EngineRoutingConfig(mode="single_engine"),
    )
    route_a = ["en", "de", "ru"]
    route_b = ["en", "tr", "ru"]

    await asyncio.gather(
        translate_route(router, "a", route_a, seed=1),
        translate_route(router, "b", route_b, seed=2),
    )

    expected_a = [
        ("prepare", "en", "de"),
        ("prepare", "de", "ru"),
        ("translate", "en", "de"),
        ("translate", "de", "ru"),
    ]
    expected_b = [
        ("prepare", "en", "tr"),
        ("prepare", "tr", "ru"),
        ("translate", "en", "tr"),
        ("translate", "tr", "ru"),
    ]
    assert cache_sensitive.operations in [expected_a + expected_b, expected_b + expected_a]


async def test_preflight_failure_preserves_provider_diagnostic() -> None:
    class BrokenPreflightProvider(FakeTranslationProvider):
        async def prepare_pair(self, source: str, target: str) -> None:
            raise RuntimeError(f"model load failed for {source}->{target}")

    router = TranslationProviderRouter(
        {"broken": BrokenPreflightProvider("broken")},
        default_provider="broken",
        routing=EngineRoutingConfig(mode="single_engine"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"broken unavailable for en->de: RuntimeError: model load failed for en->de",
    ):
        await translate_route(router, "text", ["en", "de"], seed=1)


async def test_pipeline_persists_provider_and_translation_details(config_file: Path) -> None:
    config = load_config(config_file)
    config.translation = ProviderConfig(
        default_provider="mock_a",
        providers={
            "mock_a": ProviderConfig(provider="mock"),
            "mock_b": ProviderConfig(provider="mock"),
        },
        engine_routing=EngineRoutingConfig(mode="alternating"),
    )
    run_directory = await run_pipeline(config)
    stage_path = next((run_directory / "chunks").glob("*/stage-01-*.json"))
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    assert stage["provider_route"]
    assert len(stage["translation_details"]) == len(stage["route"]) - 1
    assert all("source_language" in item for item in stage["translation_details"])
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["translation_models"]
    assert manifest["deterministic"] is True
