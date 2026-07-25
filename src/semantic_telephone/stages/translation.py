from __future__ import annotations

from itertools import pairwise

from ..models import TranslationResult
from ..providers.base import TranslationProvider


async def translate_route(
    provider: TranslationProvider,
    text: str,
    route: list[str],
    *,
    seed: int,
) -> tuple[str, list[TranslationResult]]:
    current = text
    results: list[TranslationResult] = []
    for index, (source, target) in enumerate(pairwise(route)):
        result = await provider.translate(current, source, target, seed=seed + index)
        if not result.text.strip():
            raise ValueError(f"empty translation for {source}->{target}")
        current = result.text
        results.append(result)
    return current, results
