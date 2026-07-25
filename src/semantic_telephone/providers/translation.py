from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..models import TranslationResult
from ..runtime import RequestController


class LibreTranslateProvider:
    """Client for LibreTranslate-compatible local or hosted endpoints."""

    category = "self_hosted_nmt"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        retries: int = 3,
        retry_backoff_seconds: float = 0.5,
        client: httpx.AsyncClient | None = None,
        request_controller: RequestController | None = None,
        provider_alias: str = "libretranslate",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = client
        self._languages: list[dict[str, Any]] | None = None
        self.server_version: str | None = None
        self.request_controller = request_controller
        self.provider_alias = provider_alias

    def supports_pair(self, source_language: str, target_language: str) -> bool:
        if self._languages is None:
            return source_language != target_language
        supported = {
            item.get("code")
            for item in self._languages
            if isinstance(item, dict) and isinstance(item.get("code"), str)
        }
        return source_language in supported and target_language in supported

    async def get_languages(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if self._languages is not None and not refresh:
            return self._languages
        response = await self._request("GET", "/languages", task="preflight")
        data = response.json()
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise TypeError("LibreTranslate /languages response must be a list")
        self._languages = data
        self._capture_version(response)
        return data

    async def check_availability(self) -> bool:
        try:
            await self.get_languages(refresh=True)
        except (httpx.HTTPError, TypeError, ValueError):
            return False
        return True

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None = None,
    ) -> TranslationResult:
        await self.get_languages()
        if not self.supports_pair(source_language, target_language):
            raise ValueError(
                f"LibreTranslate does not support {source_language}->{target_language}"
            )
        payload: dict[str, object] = {
            "q": text,
            "source": source_language,
            "target": target_language,
            "format": "text",
        }
        if self.api_key:
            payload["api_key"] = self.api_key
        response = await self._request(
            "POST",
            "/translate",
            json=payload,
            task=f"{source_language}->{target_language}",
        )
        self._capture_version(response)
        data = response.json()
        translated = data.get("translatedText")
        if not isinstance(translated, str):
            raise TypeError("translation response has no translatedText string")
        warnings = [] if seed is None else ["provider does not support seed"]
        usage: dict[str, int | float] = {
            "input_characters": len(text),
            "output_characters": len(translated),
        }
        if self.request_controller:
            self.request_controller.observe_usage(usage, operation="translate")
        return TranslationResult(
            text=translated,
            provider="libretranslate",
            model="configured-endpoint",
            usage=usage,
            warnings=warnings,
            deterministic=False,
            metadata={
                "category": self.category,
                "server_version": self.server_version,
                "base_url": self.base_url,
            },
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        task: str | None = None,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            request_id = (
                await self.request_controller.before_request(
                    provider=self.provider_alias,
                    provider_type="libretranslate",
                    operation=path.removeprefix("/") or method.lower(),
                    task=task,
                    retry_attempt=attempt + 1,
                )
                if self.request_controller
                else None
            )
            try:
                if self._client is not None:
                    response = await self._client.request(
                        method, f"{self.base_url}{path}", json=json
                    )
                else:
                    async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                        response = await client.request(method, f"{self.base_url}{path}", json=json)
                response.raise_for_status()
                if self.request_controller and request_id:
                    self.request_controller.request_succeeded(
                        request_id,
                        provider=self.provider_alias,
                        provider_type="libretranslate",
                        operation=path.removeprefix("/") or method.lower(),
                        task=task,
                    )
                return response
            except (httpx.HTTPError, TimeoutError) as error:
                if self.request_controller and request_id:
                    self.request_controller.request_failed(
                        request_id,
                        provider=self.provider_alias,
                        provider_type="libretranslate",
                        operation=path.removeprefix("/") or method.lower(),
                        task=task,
                        error=error,
                    )
                last_error = error
                if attempt + 1 < self.retries:
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
        assert last_error is not None
        raise last_error

    def _capture_version(self, response: httpx.Response) -> None:
        self.server_version = (
            response.headers.get("x-libretranslate-version")
            or response.headers.get("x-api-version")
            or self.server_version
        )
