from __future__ import annotations

from typing import Any

import httpx

from ..models import GenerationResult
from ..utils.retry import NonRetryableProviderError


class GenerationResponseError(RuntimeError):
    """OpenAI-compatible response reported an upstream generation failure."""


class GenerationContentError(NonRetryableProviderError):
    """The provider completed the request without usable final text."""


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_tokens: int | None = None,
        parameters: dict[str, Any] | None = None,
        provider_name: str = "openai_compatible",
        headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.parameters = parameters or {}
        self.provider_name = provider_name
        self.headers = headers or {}
        self._client = client

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        seed: int | None = None,
        response_format: str = "text",
    ) -> GenerationResult:
        body: dict[str, object] = {
            **self.parameters,
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if seed is not None:
            body["seed"] = seed
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}", **self.headers}
        if self._client is not None:
            response = await self._client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=body
            )
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=body
                )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise GenerationContentError("generation response body is not an object")
        if data.get("error") is not None:
            raise GenerationResponseError(_provider_error_message(data["error"]))
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise GenerationContentError(
                "generation response has no choices[0].message object"
            ) from error
        if not isinstance(choice, dict) or not isinstance(message, dict):
            raise GenerationContentError("generation response has no choices[0].message object")
        response_error = choice.get("error", data.get("error"))
        if response_error is not None:
            raise GenerationResponseError(_provider_error_message(response_error))
        finish_reason = choice.get("finish_reason")
        text = _extract_text_content(message.get("content"))
        if finish_reason not in {None, "stop"}:
            raise GenerationContentError(
                _content_error_message(
                    data,
                    message,
                    finish_reason=finish_reason,
                    max_tokens=self.max_tokens,
                )
            )
        if text is None or not text.strip():
            raise GenerationContentError(
                _content_error_message(
                    data,
                    message,
                    finish_reason=finish_reason,
                    max_tokens=self.max_tokens,
                )
            )
        usage_raw = data.get("usage")
        usage = usage_raw if isinstance(usage_raw, dict) else None
        response_model = data.get("model")
        model = response_model if isinstance(response_model, str) else self.model
        warnings = []
        if model != self.model:
            warnings.append(f"configured model {self.model} resolved to {model}")
        return GenerationResult(
            text=text,
            provider=self.provider_name,
            model=model,
            response_id=data.get("id"),
            usage=usage,
            warnings=warnings,
            deterministic=False,
        )


def _extract_text_content(content: object) -> str | None:
    if isinstance(content, str):
        return content
    blocks = content if isinstance(content, list) else [content]
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts) if parts else None


def _content_error_message(
    data: dict[str, Any],
    message: dict[str, Any],
    *,
    finish_reason: object,
    max_tokens: int | None,
) -> str:
    usage = data.get("usage")
    completion_tokens: object = None
    reasoning_tokens: object = None
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning_tokens = details.get("reasoning_tokens")
    content = message.get("content")
    details = [
        "generation returned no complete final text",
        f"finish_reason={finish_reason!r}",
        f"content_type={type(content).__name__}",
        f"reasoning_returned={bool(message.get('reasoning') or message.get('reasoning_details'))}",
    ]
    if max_tokens is not None:
        details.append(f"max_tokens={max_tokens}")
    if completion_tokens is not None:
        details.append(f"completion_tokens={completion_tokens}")
    if reasoning_tokens is not None:
        details.append(f"reasoning_tokens={reasoning_tokens}")
    if finish_reason in {"length", "max_tokens"}:
        if isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0:
            details.append(
                "increase the task max_tokens or lower/disable reasoning in generation parameters"
            )
        else:
            details.append(
                "the model exhausted its output budget; inspect the input for repetition "
                "or increase the task max_tokens"
            )
    elif message.get("refusal"):
        details.append("the model returned a refusal")
    elif message.get("tool_calls"):
        details.append("the model returned tool calls instead of text")
    return "; ".join(details)


def _provider_error_message(value: object) -> str:
    if not isinstance(value, dict):
        return "provider returned an embedded generation error"
    metadata = value.get("metadata")
    error_type = metadata.get("error_type") if isinstance(metadata, dict) else None
    fields = [
        f"code={value.get('code')!r}",
        f"error_type={error_type!r}",
    ]
    message = value.get("message")
    if isinstance(message, str):
        fields.append(f"message={message[:300]}")
    return "provider returned an embedded generation error: " + "; ".join(fields)


class OpenRouterGenerationProvider(OpenAICompatibleProvider):
    """OpenRouter's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 60.0,
        max_tokens: int | None = None,
        parameters: dict[str, Any] | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
            parameters=parameters,
            provider_name="openrouter",
            headers=headers,
            client=client,
        )
