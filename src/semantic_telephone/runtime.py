from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from .models import RuntimeConfig
from .utils.files import append_jsonl
from .utils.retry import NonRetryableProviderError, provider_retry_attempt


class BudgetExceededError(NonRetryableProviderError):
    """A run-level governance limit that retries and fallback may not bypass."""


class RequestController:
    def __init__(
        self,
        config: RuntimeConfig,
        events_path: Path,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self.events_path = events_path
        self.clock = clock
        self.sleeper = sleeper
        self._rate_lock = asyncio.Lock()
        self._next_start = 0.0
        self.requests = 0
        self.http_attempts = 0
        self.prompt_tokens = 0.0
        self.completion_tokens = 0.0
        self.total_tokens = 0.0
        self.cost_usd = 0.0
        self.input_characters = 0.0
        self.output_characters = 0.0
        self.segments = 0.0
        self.retries = 0
        self.enforceability_warnings: set[str] = set()
        self._started: dict[str, float] = {}
        self._restore()

    async def before_request(
        self,
        *,
        provider: str,
        operation: str,
        task: str | None = None,
        provider_type: str | None = None,
        retry_attempt: int | None = None,
    ) -> str:
        resolved_attempt = retry_attempt or provider_retry_attempt.get()
        async with self._rate_lock:
            self._raise_if_budget_prevents_request()
            rpm = self.config.requests_per_minute
            if rpm:
                now = self.clock()
                delay = max(0.0, self._next_start - now)
                if delay:
                    await self.sleeper(delay)
                    now = self.clock()
                self._next_start = max(now, self._next_start) + 60.0 / rpm
            request_id = f"request-{uuid4().hex}"
            self.requests += 1
            self.http_attempts += 1
            if resolved_attempt > 1:
                self.retries += 1
            self._started[request_id] = self.clock()
        append_jsonl(
            self.events_path,
            {
                "event": "provider_request_started",
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "provider_alias": provider,
                "provider_type": provider_type or provider,
                "operation": operation,
                "task": task,
                "retry_attempt": resolved_attempt,
                "outcome": "started",
            },
        )
        return request_id

    def request_succeeded(
        self,
        request_id: str,
        *,
        provider: str,
        operation: str,
        task: str | None = None,
        usage: dict[str, Any] | None = None,
        provider_type: str | None = None,
    ) -> None:
        normalized = self.observe_usage(usage, operation=operation)
        duration = self._request_duration(request_id)
        append_jsonl(
            self.events_path,
            {
                "event": "provider_request_completed",
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "provider_alias": provider,
                "provider_type": provider_type or provider,
                "operation": operation,
                "task": task,
                "outcome": "success",
                "duration_seconds": duration,
                "usage": normalized,
            },
        )

    def request_failed(
        self,
        request_id: str,
        *,
        provider: str,
        operation: str,
        task: str | None = None,
        error: BaseException,
        provider_type: str | None = None,
    ) -> None:
        duration = self._request_duration(request_id)
        append_jsonl(
            self.events_path,
            {
                "event": "provider_request_failed",
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "provider_alias": provider,
                "provider_type": provider_type or provider,
                "operation": operation,
                "task": task,
                "outcome": "failed",
                "duration_seconds": duration,
                "diagnostic": safe_diagnostic(error),
            },
        )

    def observe_usage(
        self, usage: dict[str, Any] | None, *, operation: str
    ) -> dict[str, float]:
        if not usage:
            if operation in {"translate", "generate", "report_generation"}:
                if self.config.budgets.max_total_tokens is not None:
                    self.enforceability_warnings.add(
                        "token budget is not fully enforceable because a provider omitted token usage"
                    )
                if self.config.budgets.max_cost_usd is not None:
                    self.enforceability_warnings.add(
                        "cost budget is not fully enforceable because a provider omitted cost usage"
                    )
            return {}
        aliases = {
            "prompt_tokens": ("prompt_tokens", "input_tokens"),
            "completion_tokens": ("completion_tokens", "output_tokens"),
            "total_tokens": ("total_tokens",),
            "cost_usd": ("cost_usd", "cost"),
            "input_characters": ("input_characters",),
            "output_characters": ("output_characters",),
            "segments": ("segments",),
        }
        normalized: dict[str, float] = {}
        for target, keys in aliases.items():
            value = next(
                (
                    usage[key]
                    for key in keys
                    if isinstance(usage.get(key), (int, float))
                    and not isinstance(usage.get(key), bool)
                ),
                None,
            )
            if value is not None:
                normalized[target] = float(value)
        if "total_tokens" not in normalized and (
            "prompt_tokens" in normalized or "completion_tokens" in normalized
        ):
            normalized["total_tokens"] = normalized.get(
                "prompt_tokens", 0.0
            ) + normalized.get("completion_tokens", 0.0)
        self.prompt_tokens += normalized.get("prompt_tokens", 0.0)
        self.completion_tokens += normalized.get("completion_tokens", 0.0)
        self.total_tokens += normalized.get("total_tokens", 0.0)
        self.cost_usd += normalized.get("cost_usd", 0.0)
        self.input_characters += normalized.get("input_characters", 0.0)
        self.output_characters += normalized.get("output_characters", 0.0)
        self.segments += normalized.get("segments", 0.0)
        if self.config.budgets.max_total_tokens is not None and "total_tokens" not in normalized:
            self.enforceability_warnings.add(
                "token budget is not fully enforceable because a provider omitted token usage"
            )
        if self.config.budgets.max_cost_usd is not None and "cost_usd" not in normalized:
            self.enforceability_warnings.add(
                "cost budget is not fully enforceable because a provider omitted cost usage"
            )
        return normalized

    def summary(self) -> dict[str, Any]:
        return {
            "http_attempts": self.http_attempts,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "input_characters": self.input_characters,
            "output_characters": self.output_characters,
            "segments": self.segments,
            "retries": self.retries,
            "enforceability_warnings": sorted(self.enforceability_warnings),
        }

    def budget_status(self) -> dict[str, Any]:
        limits = self.config.budgets
        exceeded = {
            "max_requests": (
                limits.max_requests is not None and self.requests >= limits.max_requests
            ),
            "max_total_tokens": (
                limits.max_total_tokens is not None
                and self.total_tokens >= limits.max_total_tokens
            ),
            "max_cost_usd": (
                limits.max_cost_usd is not None and self.cost_usd >= limits.max_cost_usd
            ),
        }
        return {
            "limits": {
                "max_requests": limits.max_requests,
                "max_total_tokens": limits.max_total_tokens,
                "max_cost_usd": limits.max_cost_usd,
            },
            "exceeded": exceeded,
            "warnings": sorted(self.enforceability_warnings),
        }

    def _raise_if_budget_prevents_request(self) -> None:
        limits = self.config.budgets
        if limits.max_requests is not None and self.requests >= limits.max_requests:
            raise BudgetExceededError(
                f"remote request budget exhausted: {self.requests}/{limits.max_requests}"
            )
        if (
            limits.max_total_tokens is not None
            and self.total_tokens >= limits.max_total_tokens
        ):
            raise BudgetExceededError(
                "observed token budget exhausted: "
                f"{self.total_tokens:g}/{limits.max_total_tokens}"
            )
        if limits.max_cost_usd is not None and self.cost_usd >= limits.max_cost_usd:
            raise BudgetExceededError(
                f"observed cost budget exhausted: ${self.cost_usd:g}/${limits.max_cost_usd:g}"
            )

    def _restore(self) -> None:
        if not self.events_path.exists():
            return
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("event") == "provider_request_started":
                self.requests += 1
                self.http_attempts += 1
                retry_attempt = event.get("retry_attempt")
                if isinstance(retry_attempt, int) and retry_attempt > 1:
                    self.retries += 1
            elif event.get("event") == "provider_request_completed":
                usage = event.get("usage")
                if isinstance(usage, dict):
                    self.observe_usage(usage, operation=str(event.get("operation", "")))

    def _request_duration(self, request_id: str) -> float | None:
        started = self._started.pop(request_id, None)
        return None if started is None else max(0.0, self.clock() - started)


def safe_diagnostic(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    message = re.sub(
        r"(?i)(authorization|api[_-]?key|token|secret)(\s*[:=]\s*)([^\s,;&]+)",
        r"\1\2[REDACTED]",
        message,
    )
    message = re.sub(r"(?i)bearer\s+[^\s,;&]+", "Bearer [REDACTED]", message)
    return message[:500]
