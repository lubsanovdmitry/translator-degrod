from __future__ import annotations

import httpx

from ..models import GenerationResult


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        seed: int | None = None,
        response_format: str = "text",
    ) -> GenerationResult:
        body: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=body
            )
            response.raise_for_status()
            data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("invalid OpenAI-compatible response") from error
        if not isinstance(text, str):
            raise TypeError("generation content is not text")
        usage_raw = data.get("usage")
        usage = usage_raw if isinstance(usage_raw, dict) else None
        return GenerationResult(
            text=text,
            provider="openai_compatible",
            model=self.model,
            response_id=data.get("id"),
            usage=usage,
            deterministic=False,
        )
