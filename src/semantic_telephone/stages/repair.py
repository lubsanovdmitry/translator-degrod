from __future__ import annotations

from ..models import GenerationResult
from ..providers.base import TextGenerationProvider


async def repair_text(
    provider: TextGenerationProvider,
    instruction: str,
    text: str,
    *,
    temperature: float,
    seed: int,
) -> GenerationResult:
    return await provider.generate(
        f"{instruction.strip()}\n\n<<<TEXT>>>\n{text}",
        temperature=temperature,
        seed=seed,
    )

