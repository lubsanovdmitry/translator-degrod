from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from semantic_telephone.config import config_from_resolved, load_config
from semantic_telephone.models import ProviderConfig
from semantic_telephone.providers.factory import generation_provider
from semantic_telephone.providers.local_mt import NllbTranslationProvider
from semantic_telephone.providers.openai_compatible import (
    GenerationContentError,
    OpenRouterGenerationProvider,
)
from semantic_telephone.providers.opus_mt import OpusMtTranslationProvider
from semantic_telephone.providers.translation import LibreTranslateProvider
from semantic_telephone.utils.retry import with_retry


def test_all_mt_profiles_validate_and_roundtrip() -> None:
    for name in (
        "nllb_only",
        "m2m100_only",
        "pairwise_opus",
        "mixed_local",
        "raw_translation",
        "grammar_repair",
        "conservative_reconstruction",
        "aggressive_reconstruction",
        "mixed_with_libretranslate",
        "commercial_baseline",
    ):
        config = load_config(f"configs/{name}.yaml")
        restored = config_from_resolved(config.to_dict())
        assert restored.translation.default_provider == config.translation.default_provider
        assert restored.translation.engine_routing.mode == (config.translation.engine_routing.mode)
        assert restored.translation.providers.keys() == config.translation.providers.keys()


async def test_libretranslate_languages_translation_and_version() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/languages":
            return httpx.Response(
                200,
                json=[{"code": "ru"}, {"code": "en"}],
                headers={"x-libretranslate-version": "1.6.0"},
            )
        payload = json.loads(request.content)
        assert "api_key" not in payload
        return httpx.Response(200, json={"translatedText": "hello"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = LibreTranslateProvider(base_url="http://libre.test", client=client, retries=2)
        result = await provider.translate("привет", "ru", "en", seed=1)
        assert result.text == "hello"
        assert result.metadata["server_version"] == "1.6.0"
        await provider.get_languages()
    assert [request.url.path for request in requests] == ["/languages", "/translate"]


def test_openrouter_task_config_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/default")
    config = ProviderConfig(
        provider="openrouter",
        tasks={
            "reconstruction": ProviderConfig(
                provider="openrouter",
                model="vendor/reconstructor",
                options={"max_tokens": 321},
            )
        },
    )
    provider = generation_provider(config, task="reconstruction")
    assert isinstance(provider, OpenRouterGenerationProvider)
    assert provider.model == "vendor/reconstructor"
    assert provider.max_tokens == 321
    assert provider.base_url == "https://openrouter.ai/api/v1"


async def test_openrouter_request_contains_task_parameters() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["model"] == "vendor/model"
        assert body["max_tokens"] == 321
        assert body["top_p"] == 0.8
        assert body["reasoning"] == {"enabled": False}
        assert body["seed"] == 7
        return httpx.Response(
            200,
            json={
                "id": "generation-1",
                "choices": [{"message": {"content": "result"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterGenerationProvider(
            api_key="secret",
            model="vendor/model",
            max_tokens=321,
            parameters={"top_p": 0.8, "reasoning": {"enabled": False}},
            client=client,
        )
        result = await provider.generate("prompt", temperature=0.4, seed=7)
    assert result.provider == "openrouter"
    assert result.response_id == "generation-1"
    assert result.text == "result"


async def test_openrouter_extracts_text_content_blocks() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "generation-blocks",
                "model": "resolved/model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [
                                {"type": "text", "text": "first"},
                                {"type": "output_text", "text": "second"},
                            ]
                        },
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterGenerationProvider(
            api_key="secret", model="configured/model", client=client
        )
        result = await provider.generate("prompt", temperature=0.4)
    assert result.text == "first\nsecond"
    assert result.model == "resolved/model"
    assert "resolved to" in result.warnings[0]


async def test_openrouter_null_content_reports_reasoning_exhaustion_without_retry() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "id": "generation-reasoning-only",
                "model": "qwen/qwen3.5-9b",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": None, "reasoning": "internal reasoning"},
                    }
                ],
                "usage": {
                    "completion_tokens": 1600,
                    "completion_tokens_details": {"reasoning_tokens": 1600},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenRouterGenerationProvider(
            api_key="secret",
            model="qwen/qwen3.5-9b",
            max_tokens=1600,
            client=client,
        )
        with pytest.raises(GenerationContentError, match="reasoning_tokens=1600"):
            await with_retry(
                lambda: provider.generate("prompt", temperature=0.4),
                retries=4,
                backoff_seconds=0,
            )
    assert calls == 1


def test_real_profiles_disable_generation_reasoning() -> None:
    for name in (
        "nllb_only",
        "m2m100_only",
        "pairwise_opus",
        "mixed_local",
        "grammar_repair",
        "conservative_reconstruction",
        "aggressive_reconstruction",
        "mixed_with_libretranslate",
    ):
        config = load_config(f"configs/{name}.yaml")
        assert config.generation.options["parameters"]["reasoning"] == {"enabled": False}


class _TokenizingStub:
    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        tokens = list(range(len(text.split())))
        if add_special_tokens:
            tokens = [99, *tokens, 100]
        return {"input_ids": tokens}

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return 2

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return " ".join(f"t{token_id}" for token_id in token_ids)


def test_local_mt_splits_oversized_sentences_below_safe_limit() -> None:
    provider = NllbTranslationProvider(max_input_tokens=6)
    provider._tokenizer = _TokenizingStub()
    segments = provider._split_oversized("one two three four five six seven eight nine ten", "\n\n")
    assert len(segments) == 3
    assert segments[-1][1] == "\n\n"
    assert all(len(text.split()) <= 4 for text, _ in segments)


def test_local_mt_default_decoding_is_bounded_and_avoids_repeated_ngrams() -> None:
    provider = NllbTranslationProvider()
    assert provider.decoding["max_new_tokens"] == 512
    assert provider.decoding["no_repeat_ngram_size"] == 3


def test_local_mt_does_not_inject_transformers_generation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class FakeModel:
        config = SimpleNamespace(_commit_hash="cached-revision")

        def to(self, device: str) -> FakeModel:
            assert device == "cpu"
            return self

        def eval(self) -> None:
            return None

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def fake_model_from_pretrained(*args: object, **kwargs: object) -> FakeModel:
        assert "generation_config" not in kwargs
        assert "config" not in kwargs
        return FakeModel()

    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    provider = NllbTranslationProvider(device="cpu", local_files_only=True)
    provider._load("ru")
    assert provider._resolved_revision == "cached-revision"


def test_opus_can_download_only_explicitly_configured_pairs() -> None:
    provider = OpusMtTranslationProvider(
        pairs={"ru-en": "Helsinki-NLP/opus-mt-ru-en"},
        allow_downloads=True,
        configured_pairs_only=True,
    )
    assert provider.supports_pair("ru", "en") is True
    assert provider.supports_pair("ru", "de") is False
    assert provider.requires_route_serialization is True


def test_opus_does_not_offer_hub_route_with_a_missing_leg() -> None:
    provider = OpusMtTranslationProvider(
        pairs={
            "de-en": "Helsinki-NLP/opus-mt-de-en",
            "tr-en": "Helsinki-NLP/opus-mt-tr-en",
        },
        allow_downloads=True,
        configured_pairs_only=True,
        fallback_hub_language="en",
    )

    assert provider.supports_pair("de", "tr") is False


def test_real_profiles_do_not_reference_missing_opus_en_tr_model() -> None:
    for name in (
        "mixed_local",
        "raw_translation",
        "grammar_repair",
        "conservative_reconstruction",
        "aggressive_reconstruction",
    ):
        config = load_config(f"configs/{name}.yaml")
        pairs = config.translation.providers["opus"].options["pairs"]
        assert "en-tr" not in pairs
