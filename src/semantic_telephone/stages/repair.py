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
    max_length_ratio: float | None = None,
    allow_new_events: bool | None = None,
) -> GenerationResult:
    controls: list[str] = []
    if max_length_ratio is not None:
        controls.append(
            "Длина результата не должна превышать "
            f"{max_length_ratio:.2f} длины входа."
        )
    if allow_new_events is False:
        controls.append("Не добавляй новые события, действия, факты или реплики.")
    prompt_parts = [instruction.strip()]
    if controls:
        prompt_parts.append("ОГРАНИЧЕНИЯ РЕЖИМА:\n- " + "\n- ".join(controls))
    prompt_parts.append(f"<<<TEXT>>>\n{text}")
    return await provider.generate(
        "\n\n".join(prompt_parts),
        temperature=temperature,
        seed=seed,
    )
