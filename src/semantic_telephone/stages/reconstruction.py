from __future__ import annotations

from ..models import GenerationResult
from ..providers.base import TextGenerationProvider


async def reconstruct_text(
    provider: TextGenerationProvider,
    instruction: str,
    damaged_text: str,
    *,
    temperature: float,
    seed: int,
    context: str = "",
    memory: str = "",
) -> GenerationResult:
    sections = [instruction.strip()]
    if context:
        sections.append(f"ПРЕДЫДУЩИЙ ПОВРЕЖДЁННЫЙ КОНТЕКСТ:\n{context}")
    if memory:
        sections.append(
            "НЕУВЕРЕННЫЕ АВТОМАТИЧЕСКИЕ НАБЛЮДЕНИЯ; ИХ МОЖНО ИГНОРИРОВАТЬ:\n" + memory
        )
    sections.append(f"<<<TEXT>>>\n{damaged_text}")
    return await provider.generate(
        "\n\n".join(sections), temperature=temperature, seed=seed
    )

