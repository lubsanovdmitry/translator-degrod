from __future__ import annotations

import random

from .models import RouteConfig

LANGUAGE_GROUPS: dict[str, list[str]] = {
    "germanic": ["de", "nl", "sv", "en"],
    "romance": ["fr", "es", "it", "pt"],
    "slavic": ["pl", "uk", "hr", "cs"],
    "agglutinative": ["tr", "fi", "hu", "sw"],
    "semitic": ["ar", "he"],
    "non_latin": ["ja", "ka", "hi", "th"],
    "lower_resource": ["sw", "ka", "mt", "cy"],
    "hub": ["en"],
}
DEFAULT_LANGUAGES = sorted({value for values in LANGUAGE_GROUPS.values() for value in values})


def generate_route(
    config: RouteConfig,
    *,
    source_language: str,
    target_language: str,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    allowed = _allowed(config, source_language, target_language)
    hops = rng.randint(config.min_hops, config.max_hops)
    if config.mode == "fixed":
        route = list(config.languages) or [source_language, target_language]
    elif config.mode == "random":
        route = [source_language, *(rng.choice(allowed) for _ in range(max(0, hops - 1)))]
    elif config.mode == "stratified":
        groups = list(LANGUAGE_GROUPS.values())
        route = [source_language]
        for index in range(max(0, hops - 1)):
            candidates = [lang for lang in groups[index % len(groups)] if lang in allowed]
            route.append(rng.choice(candidates or allowed))
    elif config.mode == "hubbed":
        route = [source_language]
        for index in range(max(0, hops - 1)):
            if index % max(1, config.hub_frequency + 1) == config.hub_frequency:
                route.append(config.hub_language)
            else:
                route.append(rng.choice(allowed))
    elif config.mode == "mutating_fixed":
        route = list(config.languages) or [source_language, target_language]
        mutable = list(range(1, max(1, len(route) - 1)))
        rng.shuffle(mutable)
        for index in mutable[: config.mutations]:
            route[index] = rng.choice(allowed)
    else:
        raise ValueError(f"unknown route mode: {config.mode}")
    if not route or route[0] != source_language:
        route.insert(0, source_language)
    route = _deduplicate(route)
    if route[-1] != target_language:
        route.append(target_language)
    route = _deduplicate(route)
    return route


def _allowed(config: RouteConfig, source: str, target: str) -> list[str]:
    candidates = config.allow or config.languages or DEFAULT_LANGUAGES
    result = [value for value in candidates if value not in config.deny]
    result = [value for value in result if value not in {source, target}]
    if not result:
        excluded = set(config.deny) | {source, target}
        result = [value for value in DEFAULT_LANGUAGES if value not in excluded]
    if not result:
        raise ValueError("language allow/deny lists leave no intermediate language")
    return result


def _deduplicate(route: list[str]) -> list[str]:
    result: list[str] = []
    for language in route:
        if not result or result[-1] != language:
            result.append(language)
    return result
