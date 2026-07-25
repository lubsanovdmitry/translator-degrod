from __future__ import annotations

from ..memory import MemoryStore
from ..models import GenerationResult
from ..providers.base import TextGenerationProvider


async def extract_memory(
    provider: TextGenerationProvider,
    instruction: str,
    damaged_text: str,
    *,
    temperature: float,
    seed: int,
) -> GenerationResult:
    # The signature deliberately accepts no original text.
    return await provider.generate(
        f"{instruction.strip()}\n\n<<<TEXT>>>\n{damaged_text}",
        temperature=temperature,
        seed=seed,
        response_format="json",
    )


def memory_prompt(store: MemoryStore, chunk_index: int, maximum: int, minimum: int) -> str:
    return store.prompt_items(chunk_index, maximum=maximum, minimum_count=minimum)

