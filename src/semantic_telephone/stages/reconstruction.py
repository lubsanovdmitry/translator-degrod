from __future__ import annotations

import re

from ..models import GenerationResult
from ..providers.base import TextGenerationProvider

_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_REPETITION_THRESHOLD = 8


async def reconstruct_text(
    provider: TextGenerationProvider,
    instruction: str,
    damaged_text: str,
    *,
    temperature: float,
    seed: int,
    context: str = "",
    memory: str = "",
    max_length_ratio: float | None = None,
    max_new_sentences_per_chunk: int | None = None,
    repetition_policy: str | None = None,
    allow_new_events: bool | None = None,
    allow_scene_expansion: bool | None = None,
) -> GenerationResult:
    sections = [instruction.strip()]
    controls: list[str] = []
    if max_length_ratio is not None:
        controls.append(
            "Длина результата не должна превышать "
            f"{max_length_ratio:.2f} длины входа."
        )
    if max_new_sentences_per_chunk is not None:
        controls.append(
            "Сохрани приблизительное число предложений; добавь не более "
            f"{max_new_sentences_per_chunk} коротких предложений-связок."
        )
    if repetition_policy == "preserve":
        controls.append(
            "Сохраняй повторы как повторы. Не объясняй их голосом, эхом, заклинанием, "
            "мантрой, заевшей пластинкой или нарушением реальности."
        )
    elif repetition_policy == "rationalize":
        controls.append(
            "Повторы разрешается связывать с происходящим и объяснять внутри сцены."
        )
    if allow_new_events is False:
        controls.append("Не добавляй новые события, конфликты, объекты или явления.")
    if allow_scene_expansion is False:
        controls.append("Не расширяй вход новыми сценами.")
    elif allow_scene_expansion is True:
        controls.append("Расширение существующих сцен разрешено, но не обязательно.")
    if controls:
        sections.append("ОГРАНИЧЕНИЯ РЕЖИМА:\n- " + "\n- ".join(controls))
    if context:
        sections.append(f"ПРЕДЫДУЩИЙ ПОВРЕЖДЁННЫЙ КОНТЕКСТ:\n{context}")
    if memory:
        sections.append(
            "НЕУВЕРЕННЫЕ АВТОМАТИЧЕСКИЕ НАБЛЮДЕНИЯ; ИХ МОЖНО ИГНОРИРОВАТЬ:\n" + memory
        )
    repeated_word = _long_consecutive_repetition(damaged_text)
    if repeated_word is not None:
        sections.append(
            "ТЕХНИЧЕСКОЕ ОГРАНИЧЕНИЕ:\n"
            "Во входе обнаружена длинная подряд идущая серия одного слова — это "
            "дегенерация машинного перевода, а не множество отдельных фактов. "
            f"Сократи серию «{repeated_word}» до одного-трёх повторов. Не воспроизводи "
            "никакое слово или короткую фразу более трёх раз подряд. Правило объёма "
            "80–120% применяй к очищенному смысловому содержанию, а не к длине повреждённой "
            "серии; не дополняй текст ради компенсации удалённых повторов. Обязательно "
            "закончи ответ."
        )
    sections.append(f"<<<TEXT>>>\n{damaged_text}")
    return await provider.generate(
        "\n\n".join(sections), temperature=temperature, seed=seed
    )


def _long_consecutive_repetition(text: str) -> str | None:
    previous = ""
    run_length = 0
    for match in _WORD_PATTERN.finditer(text):
        word = match.group(0)
        normalized = word.casefold()
        if normalized == previous:
            run_length += 1
        else:
            previous = normalized
            run_length = 1
        if run_length >= _REPETITION_THRESHOLD:
            return word
    return None
