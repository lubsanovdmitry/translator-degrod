from __future__ import annotations

import asyncio

import pytest

from semantic_telephone.models import GenerationResult
from semantic_telephone.pipeline import _valid_memory_generation
from semantic_telephone.utils.retry import (
    EmptyProviderResponse,
    NonRetryableProviderError,
    with_retry,
)


def test_empty_provider_response_is_retried_and_reported() -> None:
    calls = 0

    async def empty() -> GenerationResult:
        nonlocal calls
        calls += 1
        return GenerationResult(text="", provider="test", model="test")

    with pytest.raises(EmptyProviderResponse):
        asyncio.run(with_retry(empty, retries=3, backoff_seconds=0))
    assert calls == 3


def test_transient_provider_error_is_retried() -> None:
    calls = 0

    async def eventual() -> GenerationResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary")
        return GenerationResult(text="ok", provider="test", model="test")

    result, attempts = asyncio.run(with_retry(eventual, retries=2, backoff_seconds=0))
    assert result.text == "ok"
    assert attempts == 2


def test_non_retryable_provider_error_stops_after_one_attempt() -> None:
    calls = 0

    async def invalid() -> GenerationResult:
        nonlocal calls
        calls += 1
        raise NonRetryableProviderError("same request cannot fix this")

    with pytest.raises(NonRetryableProviderError) as captured:
        asyncio.run(with_retry(invalid, retries=4, backoff_seconds=0))
    assert calls == 1
    assert captured.value.attempts == 1


def test_invalid_memory_schema_is_retried() -> None:
    calls = 0

    class MemoryGenerator:
        async def generate(
            self,
            prompt: str,
            *,
            temperature: float,
            seed: int | None = None,
            response_format: str = "text",
        ) -> GenerationResult:
            nonlocal calls
            calls += 1
            text = (
                '{"observations":[{"entity_key":"x","text":"seen"}]}'
                if calls == 1
                else '{"observations":[]}'
            )
            return GenerationResult(text=text, provider="test", model="test")

    async def exercise() -> tuple[GenerationResult, int]:
        provider = MemoryGenerator()
        return await with_retry(
            lambda: _valid_memory_generation(
                provider,
                "instruction",
                "damaged",
                temperature=0.1,
                seed=1,
            ),
            retries=2,
            backoff_seconds=0,
        )

    result, attempts = asyncio.run(exercise())
    assert result.text == '{"observations":[]}'
    assert attempts == 2
