from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class EmptyProviderResponse(RuntimeError):
    pass


class NonRetryableProviderError(RuntimeError):
    """A valid provider response that the same request cannot repair by retrying."""

    attempts: int = 1


def _record_attempt(error: Exception, attempt: int) -> None:
    try:
        error.__dict__["attempts"] = attempt
    except AttributeError:
        pass


async def with_retry[T](
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int,
    backoff_seconds: float,
) -> tuple[T, int]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = await operation()
            text = getattr(result, "text", None)
            if isinstance(text, str) and not text.strip():
                raise EmptyProviderResponse("provider returned an empty response")
            return result, attempt
        except Exception as error:
            last_error = error
            _record_attempt(error, attempt)
            if isinstance(error, NonRetryableProviderError):
                raise
            if attempt < retries:
                await asyncio.sleep(backoff_seconds * 2 ** (attempt - 1))
    assert last_error is not None
    raise last_error
