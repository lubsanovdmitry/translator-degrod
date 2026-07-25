from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class EmptyProviderResponse(RuntimeError):
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
        except Exception as error:  # noqa: BLE001 - provider adapters can raise vendor exceptions
            last_error = error
            if attempt < retries:
                await asyncio.sleep(backoff_seconds * 2 ** (attempt - 1))
    assert last_error is not None
    raise last_error
