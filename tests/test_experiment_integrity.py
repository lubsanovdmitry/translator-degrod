from __future__ import annotations

from pathlib import Path

from semantic_telephone.models import GenerationResult
from semantic_telephone.stages.reconstruction import reconstruct_text
from semantic_telephone.stages.repair import repair_text


def test_standard_prompts_and_configs_have_no_seeded_motifs() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        "сделай смешнее",
        "добавь абсурда",
        "добавь чёрный юмор",
        "добавь алкоголь",
        "добавь политику",
        "добавь насилие",
    ]
    files = [*root.glob("prompts/*.txt"), *root.glob("configs/*.yaml")]
    corpus = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
    assert all(phrase not in corpus for phrase in forbidden)


def test_reconstruction_stage_api_has_no_original_argument() -> None:
    from inspect import signature

    parameters = signature(reconstruct_text).parameters
    assert "original" not in parameters
    assert "damaged_text" in parameters


async def test_reconstruction_adds_guard_for_consecutive_mt_repetition() -> None:
    class CapturingProvider:
        prompt = ""

        async def generate(
            self,
            prompt: str,
            *,
            temperature: float,
            seed: int | None = None,
            response_format: str = "text",
        ) -> GenerationResult:
            del temperature, seed, response_format
            self.prompt = prompt
            return GenerationResult(text="result", provider="capture", model="capture")

    provider = CapturingProvider()
    damaged = "Начало. " + ", ".join(["шахмат"] * 20) + ". Конец."
    await reconstruct_text(provider, "instruction", damaged, temperature=0.2, seed=1)
    assert "ТЕХНИЧЕСКОЕ ОГРАНИЧЕНИЕ" in provider.prompt
    assert "одного-трёх повторов" in provider.prompt
    assert "не дополняй текст" in provider.prompt
    assert provider.prompt.endswith(f"<<<TEXT>>>\n{damaged}")


async def test_reconstruction_mode_constraints_are_added_to_prompt() -> None:
    class CapturingProvider:
        prompt = ""

        async def generate(
            self,
            prompt: str,
            *,
            temperature: float,
            seed: int | None = None,
            response_format: str = "text",
        ) -> GenerationResult:
            del temperature, seed, response_format
            self.prompt = prompt
            return GenerationResult(text="result", provider="capture", model="capture")

    provider = CapturingProvider()
    await reconstruct_text(
        provider,
        "instruction",
        "Повреждённый текст.",
        temperature=0.35,
        seed=1,
        max_length_ratio=1.15,
        max_new_sentences_per_chunk=2,
        repetition_policy="preserve",
        allow_new_events=False,
        allow_scene_expansion=False,
    )
    assert "1.15 длины входа" in provider.prompt
    assert "не более 2 коротких предложений-связок" in provider.prompt
    assert "Не объясняй их голосом, эхом, заклинанием" in provider.prompt
    assert "Не добавляй новые события, конфликты, объекты или явления" in provider.prompt
    assert "Не расширяй вход новыми сценами" in provider.prompt


async def test_grammar_repair_constraints_are_added_to_prompt() -> None:
    class CapturingProvider:
        prompt = ""

        async def generate(
            self,
            prompt: str,
            *,
            temperature: float,
            seed: int | None = None,
            response_format: str = "text",
        ) -> GenerationResult:
            del temperature, seed, response_format
            self.prompt = prompt
            return GenerationResult(text="result", provider="capture", model="capture")

    provider = CapturingProvider()
    await repair_text(
        provider,
        "instruction",
        "Повреждённый текст.",
        temperature=0.1,
        seed=1,
        max_length_ratio=1.05,
        allow_new_events=False,
    )
    assert "1.05 длины входа" in provider.prompt
    assert "Не добавляй новые события, действия, факты или реплики" in provider.prompt
