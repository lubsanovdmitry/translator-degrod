from __future__ import annotations

from typing import Protocol

from ..models import GenerationResult, TranslationResult


class TranslationProvider(Protocol):
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None = None,
    ) -> TranslationResult: ...


class TextGenerationProvider(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        seed: int | None = None,
        response_format: str = "text",
    ) -> GenerationResult: ...

