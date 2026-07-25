from __future__ import annotations

import os

from dotenv import load_dotenv

from ..models import ProviderConfig
from .base import TextGenerationProvider, TranslationProvider
from .mock import MockGenerationProvider, MockTranslationProvider
from .openai_compatible import OpenAICompatibleProvider
from .translation import LibreTranslateProvider


def translation_provider(config: ProviderConfig) -> TranslationProvider:
    load_dotenv()
    if config.provider == "mock":
        return MockTranslationProvider()
    if config.provider in {"local", "libretranslate"}:
        if not config.base_url:
            raise ValueError("translation.base_url is required for LibreTranslate")
        key = os.getenv(config.api_key_env) if config.api_key_env else None
        return LibreTranslateProvider(
            base_url=config.base_url, api_key=key, timeout_seconds=config.timeout_seconds
        )
    raise ValueError(f"unknown translation provider: {config.provider}")


def generation_provider(config: ProviderConfig) -> TextGenerationProvider:
    load_dotenv()
    if config.provider == "mock":
        return MockGenerationProvider()
    if config.provider == "openai_compatible":
        if not config.base_url or not config.api_key_env:
            raise ValueError("generation.base_url and generation.api_key_env are required")
        key = os.getenv(config.api_key_env)
        if not key:
            raise ValueError(f"environment variable is not set: {config.api_key_env}")
        return OpenAICompatibleProvider(
            base_url=config.base_url,
            api_key=key,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
        )
    raise ValueError(f"unknown generation provider: {config.provider}")

