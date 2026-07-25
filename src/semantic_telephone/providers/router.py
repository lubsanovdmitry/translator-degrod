from __future__ import annotations

import asyncio
import random
from dataclasses import replace
from itertools import pairwise
from typing import Any, cast

from ..models import EngineRoutingConfig, TranslationResult
from ..runtime import BudgetExceededError


class TranslationProviderRouter:
    """Select independent MT engines for each transition of a language route."""

    def __init__(
        self,
        providers: dict[str, Any],
        *,
        default_provider: str,
        routing: EngineRoutingConfig,
    ) -> None:
        if default_provider not in providers:
            raise ValueError(f"default translation provider is unavailable: {default_provider}")
        self.providers = providers
        self.default_provider = default_provider
        self.routing = routing
        self.serializes_routes = any(
            bool(getattr(provider, "requires_route_serialization", False))
            for provider in providers.values()
        )
        self.route_lock = asyncio.Lock()

    def supports_pair(self, source_language: str, target_language: str) -> bool:
        return any(
            self._supports(provider, source_language, target_language)
            for provider in self.providers.values()
        )

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None = None,
    ) -> TranslationResult:
        if self.serializes_routes:
            async with self.route_lock:
                return await self._translate_unlocked(
                    text,
                    source_language,
                    target_language,
                    seed=seed,
                )
        return await self._translate_unlocked(
            text,
            source_language,
            target_language,
            seed=seed,
        )

    async def _translate_unlocked(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None,
    ) -> TranslationResult:
        resolved_seed = seed or 0
        route = [source_language, target_language]
        plan = self.plan_route(route, seed=resolved_seed)
        plan, warnings = await self.prepare_plan(route, plan)
        result = await self.translate_candidates(
            plan[0],
            text,
            source_language,
            target_language,
            seed=resolved_seed,
        )
        return replace(result, warnings=[*warnings, *result.warnings])

    def plan_route(self, route: list[str], *, seed: int) -> list[list[str]]:
        transitions = list(pairwise(route))
        rng = random.Random(seed)
        selected: list[str] = []
        plan: list[list[str]] = []
        architecture_counts: dict[str, int] = {}
        provider_counts = {name: 0 for name in self.providers}
        for index, (source, target) in enumerate(transitions):
            supported = [
                name
                for name, provider in self.providers.items()
                if self._supports(provider, source, target)
            ]
            if not supported:
                raise ValueError(f"no translation provider supports {source}->{target}")
            primary = self._select_primary(
                supported,
                source=source,
                target=target,
                index=index,
                previous=selected[-1] if selected else None,
                rng=rng,
                architecture_counts=architecture_counts,
                provider_counts=provider_counts,
            )
            candidates = self._candidate_order(primary, supported)
            plan.append(candidates)
            selected.append(primary)
            provider_counts[primary] += 1
            architecture = self._architecture(primary)
            architecture_counts[architecture] = architecture_counts.get(architecture, 0) + 1
        return plan

    async def prepare_plan(
        self, route: list[str], plan: list[list[str]]
    ) -> tuple[list[list[str]], list[str]]:
        """Check pair-specific/HTTP engines before translating the route."""
        prepared: list[list[str]] = []
        warnings: list[str] = []
        checked: dict[tuple[str, str, str], bool] = {}
        for (source, target), candidates in zip(pairwise(route), plan, strict=True):
            available: list[str] = []
            unavailable: list[str] = []
            for name in candidates:
                key = (name, source, target)
                if key in checked:
                    if checked[key]:
                        available.append(name)
                    continue
                provider = self.providers[name]
                try:
                    prepare = getattr(provider, "prepare_pair", None)
                    if prepare is not None:
                        await prepare(source, target)
                    elif hasattr(provider, "get_languages"):
                        await provider.get_languages()
                        if not self._supports(provider, source, target):
                            raise ValueError("language pair is not advertised by the server")
                except Exception as error:
                    if isinstance(error, BudgetExceededError):
                        raise
                    checked[key] = False
                    message = (
                        f"{name} unavailable for {source}->{target}: "
                        f"{type(error).__name__}: {error}"
                    )
                    warnings.append(message)
                    unavailable.append(message)
                    continue
                checked[key] = True
                available.append(name)
            if not available:
                details = "; ".join(unavailable) or "no candidate provider remained"
                raise RuntimeError(
                    f"all planned providers failed preflight for {source}->{target}: {details}"
                )
            prepared.append(available)
        return prepared, warnings

    async def translate_candidates(
        self,
        candidates: list[str],
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int,
    ) -> TranslationResult:
        errors: list[str] = []
        for name in candidates:
            provider = self.providers[name]
            try:
                result = await provider.translate(text, source_language, target_language, seed=seed)
            except Exception as error:
                if isinstance(error, BudgetExceededError):
                    raise
                errors.append(f"{name}: {type(error).__name__}: {error}")
                continue
            warnings = list(result.warnings)
            if errors:
                warnings.append("quality fallback used after " + "; ".join(errors))
            metadata = {
                **result.metadata,
                "engine": name,
                "provider_type": result.provider,
                "category": getattr(provider, "category", "nmt"),
                "candidate_order": candidates,
                "fallback_attempts": errors,
            }
            return cast(
                TranslationResult,
                replace(result, provider=name, warnings=warnings, metadata=metadata),
            )
        raise RuntimeError(
            f"all providers failed for {source_language}->{target_language}: " + "; ".join(errors)
        )

    def _select_primary(
        self,
        supported: list[str],
        *,
        source: str,
        target: str,
        index: int,
        previous: str | None,
        rng: random.Random,
        architecture_counts: dict[str, int],
        provider_counts: dict[str, int],
    ) -> str:
        mode = self.routing.mode
        if mode == "single_engine":
            return self.routing.provider or self.default_provider
        if mode == "fixed_engine_route":
            pair_name = self.routing.pairs.get(f"{source}-{target}")
            if pair_name:
                return pair_name
            if index < len(self.routing.route):
                return self.routing.route[index]
            return self.default_provider
        if mode == "quality_fallback":
            order = self.routing.fallback_order or list(self.providers)
            return next((name for name in order if name in supported), self.default_provider)
        candidates = supported
        avoid_repeat = mode == "alternating" or self.routing.avoid_same_engine_consecutively
        if avoid_repeat and previous in candidates and len(candidates) > 1:
            candidates = [name for name in candidates if name != previous]
        if mode == "heterogeneous":
            ranked = sorted(
                candidates,
                key=lambda name: (
                    architecture_counts.get(self._architecture(name), 0),
                    provider_counts[name],
                    name,
                ),
            )
            best_score = (
                architecture_counts.get(self._architecture(ranked[0]), 0),
                provider_counts[ranked[0]],
            )
            tied = [
                name
                for name in ranked
                if (
                    architecture_counts.get(self._architecture(name), 0),
                    provider_counts[name],
                )
                == best_score
            ]
            return rng.choice(tied)
        return self._weighted_choice(candidates, rng)

    def _weighted_choice(self, candidates: list[str], rng: random.Random) -> str:
        weights = [self.routing.weights.get(name, 1.0) for name in candidates]
        if sum(weights) <= 0:
            return rng.choice(candidates)
        return rng.choices(candidates, weights=weights, k=1)[0]

    def _candidate_order(self, primary: str, supported: list[str]) -> list[str]:
        if primary not in supported:
            raise ValueError(f"selected provider {primary} does not support this language pair")
        if self.routing.mode != "quality_fallback":
            return [primary]
        configured = self.routing.fallback_order or list(self.providers)
        return [
            primary,
            *[name for name in configured if name != primary and name in supported],
        ]

    def _supports(self, provider: Any, source: str, target: str) -> bool:
        supports = getattr(provider, "supports_pair", None)
        return bool(supports(source, target)) if supports else source != target

    def _architecture(self, name: str) -> str:
        provider = self.providers[name]
        return str(getattr(provider, "name", provider.__class__.__name__))
