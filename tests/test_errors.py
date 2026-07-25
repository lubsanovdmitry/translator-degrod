from __future__ import annotations

import asyncio

import pytest

from semantic_telephone.models import GenerationResult
from semantic_telephone.utils.retry import EmptyProviderResponse, with_retry


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

