from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from dotenv import load_dotenv

from ..models import ProviderConfig, TranslationResult
from .base import TextGenerationProvider, TranslationProvider
from .local_mt import M2M100TranslationProvider, NllbTranslationProvider
from .mock import MockGenerationProvider, MockTranslationProvider
from .openai_compatible import OpenAICompatibleProvider, OpenRouterGenerationProvider
from .opus_mt import OpusMtTranslationProvider
from .router import TranslationProviderRouter
from .translation import LibreTranslateProvider


def translation_provider(config: ProviderConfig) -> TranslationProvider:
    load_dotenv()
    if config.providers:
        providers = {
            name: _single_translation_provider(provider)
            for name, provider in config.providers.items()
            if provider.enabled
        }
        return TranslationProviderRouter(
            providers,
            default_provider=config.default_provider or "",
            routing=config.engine_routing,
        )
    return _single_translation_provider(config)


def _single_translation_provider(config: ProviderConfig) -> TranslationProvider:
    if config.provider == "mock":
        return MockTranslationProvider()
    if config.provider in {"local", "libretranslate"}:
        base_url = (
            config.base_url or os.getenv("LIBRETRANSLATE_BASE_URL") or "http://localhost:5000"
        )
        key_env = config.api_key_env or "LIBRETRANSLATE_API_KEY"
        key = os.getenv(key_env)
        return LibreTranslateProvider(
            base_url=base_url,
            api_key=key,
            timeout_seconds=config.timeout_seconds,
            retries=int(config.options.get("retries", 3)),
            retry_backoff_seconds=float(config.options.get("retry_backoff_seconds", 0.5)),
        )
    if config.provider in {"nllb", "m2m100"}:
        provider_class = (
            NllbTranslationProvider if config.provider == "nllb" else M2M100TranslationProvider
        )
        return provider_class(
            model=None if config.model == "mock" else config.model,
            revision=config.revision,
            device=str(config.options.get("device", "auto")),
            dtype=str(config.options.get("dtype", "auto")),
            max_input_tokens=int(config.options.get("max_input_tokens", 450)),
            decoding=_dictionary(config.options.get("decoding"), "decoding"),
        )
    if config.provider in {"opus", "opus_mt"}:
        return OpusMtTranslationProvider(
            pairs={
                str(key): str(value)
                for key, value in _dictionary(config.options.get("pairs"), "opus_mt.pairs").items()
            },
            revisions={
                str(key): str(value)
                for key, value in _dictionary(
                    config.options.get("revisions"), "opus_mt.revisions"
                ).items()
            },
            allow_downloads=bool(config.options.get("allow_downloads", False)),
            configured_pairs_only=bool(config.options.get("configured_pairs_only", False)),
            fallback_hub_language=_optional_string(
                config.options.get("fallback_hub_language", "en")
            ),
            max_loaded_models=int(config.options.get("max_loaded_models", 2)),
            device=str(config.options.get("device", "auto")),
            dtype=str(config.options.get("dtype", "auto")),
            max_input_tokens=int(config.options.get("max_input_tokens", 450)),
            decoding=_dictionary(config.options.get("decoding"), "decoding"),
        )
    if config.provider in {
        "google_cloud",
        "deepl",
        "azure_translator",
        "yandex_translate",
        "commercial_nmt",
    }:
        return CommercialTranslationProviderStub(config.provider)
    raise ValueError(f"unknown translation provider: {config.provider}")


def generation_provider(
    config: ProviderConfig, *, task: str | None = None
) -> TextGenerationProvider:
    load_dotenv()
    selected = _task_provider_config(config, task)
    if selected.provider == "mock":
        return MockGenerationProvider()
    if selected.provider == "openai_compatible":
        if not selected.base_url or not selected.api_key_env:
            raise ValueError("generation.base_url and generation.api_key_env are required")
        key = os.getenv(selected.api_key_env)
        if not key:
            raise ValueError(f"environment variable is not set: {selected.api_key_env}")
        return OpenAICompatibleProvider(
            base_url=selected.base_url,
            api_key=key,
            model=selected.model,
            timeout_seconds=selected.timeout_seconds,
            max_tokens=_optional_int(selected.options.get("max_tokens")),
            parameters=_generation_parameters(selected.options),
        )
    if selected.provider == "openrouter":
        key_env = selected.api_key_env or "OPENROUTER_API_KEY"
        key = os.getenv(key_env)
        if not key:
            raise ValueError(f"environment variable is not set: {key_env}")
        model = (
            selected.model
            if selected.model and selected.model != "mock"
            else os.getenv("OPENROUTER_MODEL")
        )
        if not model:
            raise ValueError(f"OpenRouter model is not configured for task {task or 'default'}")
        return OpenRouterGenerationProvider(
            base_url=(
                selected.base_url
                or os.getenv("OPENROUTER_BASE_URL")
                or "https://openrouter.ai/api/v1"
            ),
            api_key=key,
            model=model,
            timeout_seconds=selected.timeout_seconds,
            max_tokens=_optional_int(selected.options.get("max_tokens")),
            parameters=_generation_parameters(selected.options),
            site_url=_optional_string(selected.options.get("site_url")),
            app_name=_optional_string(selected.options.get("app_name")),
        )
    raise ValueError(f"unknown generation provider: {selected.provider}")


class CommercialTranslationProviderStub:
    """Interface placeholder; commercial MT is always a distinct experimental category."""

    category = "commercial_nmt"

    def __init__(self, provider_name: str) -> None:
        self.name = provider_name

    def supports_pair(self, source_language: str, target_language: str) -> bool:
        return source_language != target_language

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None = None,
    ) -> TranslationResult:
        del text, source_language, target_language, seed
        raise NotImplementedError(
            f"{self.name} is an interface stub; configure an implemented commercial client"
        )


def _task_provider_config(config: ProviderConfig, task: str | None) -> ProviderConfig:
    if task is None or task not in config.tasks:
        return config
    selected = config.tasks[task]
    return replace(
        selected,
        base_url=selected.base_url or config.base_url,
        model=selected.model if selected.model != "mock" else config.model,
        api_key_env=selected.api_key_env or config.api_key_env,
        timeout_seconds=(
            selected.timeout_seconds if selected.timeout_seconds != 60.0 else config.timeout_seconds
        ),
        options={**config.options, **selected.options},
    )


def _dictionary(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (str, int, float)):
        raise TypeError("integer option must be a string or number")
    return int(value)


def _generation_parameters(options: dict[str, Any]) -> dict[str, Any]:
    value = options.get("parameters", options.get("additional_parameters", {}))
    explicit = _dictionary(value, "generation.parameters")
    reserved = {
        "additional_parameters",
        "app_name",
        "max_tokens",
        "parameters",
        "site_url",
        "temperature",
    }
    return {
        **{key: item for key, item in options.items() if key not in reserved},
        **explicit,
    }
