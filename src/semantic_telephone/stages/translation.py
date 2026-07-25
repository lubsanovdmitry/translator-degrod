from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

from ..models import TranslationResult
from ..providers.base import TranslationProvider
from ..providers.router import TranslationProviderRouter


async def translate_route(
    provider: TranslationProvider,
    text: str,
    route: list[str],
    *,
    seed: int,
    on_hop: Callable[[int, str, TranslationResult], None] | None = None,
) -> tuple[str, list[TranslationResult]]:
    if isinstance(provider, TranslationProviderRouter) and provider.serializes_routes:
        async with provider.route_lock:
            return await _translate_route_unlocked(
                provider,
                text,
                route,
                seed=seed,
                on_hop=on_hop,
            )
    return await _translate_route_unlocked(
        provider,
        text,
        route,
        seed=seed,
        on_hop=on_hop,
    )


async def _translate_route_unlocked(
    provider: TranslationProvider,
    text: str,
    route: list[str],
    *,
    seed: int,
    on_hop: Callable[[int, str, TranslationResult], None] | None,
) -> tuple[str, list[TranslationResult]]:
    current = text
    results: list[TranslationResult] = []
    transitions = list(pairwise(route))
    plan: list[list[str]] | None = None
    preflight_warnings: list[str] = []
    if isinstance(provider, TranslationProviderRouter):
        plan = provider.plan_route(route, seed=seed)
        plan, preflight_warnings = await provider.prepare_plan(route, plan)
    for index, (source, target) in enumerate(transitions):
        if isinstance(provider, TranslationProviderRouter):
            assert plan is not None
            result = await provider.translate_candidates(
                plan[index],
                current,
                source,
                target,
                seed=seed + index,
            )
        else:
            result = await provider.translate(current, source, target, seed=seed + index)
        if not result.text.strip():
            raise ValueError(f"empty translation for {source}->{target}")
        result.metadata.update(
            {
                "hop_index": index,
                "source_language": source,
                "target_language": target,
            }
        )
        if index == 0 and preflight_warnings:
            result.warnings[:0] = preflight_warnings
        if on_hop is not None:
            on_hop(index, current, result)
        current = result.text
        results.append(result)
    return current, results
