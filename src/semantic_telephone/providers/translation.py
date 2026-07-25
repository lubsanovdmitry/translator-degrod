from __future__ import annotations

import httpx

from ..models import TranslationResult


class LibreTranslateProvider:
    """Client for LibreTranslate-compatible local or hosted endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None = None,
    ) -> TranslationResult:
        payload: dict[str, object] = {
            "q": text,
            "source": source_language,
            "target": target_language,
            "format": "text",
        }
        if self.api_key:
            payload["api_key"] = self.api_key
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/translate", json=payload)
            response.raise_for_status()
        data = response.json()
        translated = data.get("translatedText")
        if not isinstance(translated, str):
            raise TypeError("translation response has no translatedText string")
        warnings = [] if seed is None else ["provider does not support seed"]
        return TranslationResult(
            text=translated,
            provider="libretranslate",
            model="configured-endpoint",
            usage={"input_characters": len(text), "output_characters": len(translated)},
            warnings=warnings,
            deterministic=False,
        )
