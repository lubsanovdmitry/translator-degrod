from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from ..models import TranslationResult
from .local_mt import _TransformerSeq2SeqProvider


class _MarianPairProvider(_TransformerSeq2SeqProvider):
    name = "opus_mt"

    def __init__(self, source: str, target: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.target = target

    def supports_pair(self, source_language: str, target_language: str) -> bool:
        return source_language == self.source and target_language == self.target

    def _forced_bos_token_id(self, target_language: str) -> None:
        del target_language


class OpusMtTranslationProvider:
    """Pair-specific Helsinki-NLP Marian models with optional English hub routing."""

    name = "opus_mt"
    category = "local_nmt"
    # The bounded pair-model cache must remain stable from route preflight
    # through the last route hop. The router uses this marker to prevent
    # another chunk's preflight from evicting models reserved by this route.
    requires_route_serialization = True

    def __init__(
        self,
        *,
        pairs: dict[str, str] | None = None,
        revisions: dict[str, str] | None = None,
        allow_downloads: bool = False,
        configured_pairs_only: bool = False,
        fallback_hub_language: str | None = "en",
        max_loaded_models: int = 2,
        device: str = "auto",
        dtype: str = "auto",
        max_input_tokens: int = 450,
        decoding: dict[str, Any] | None = None,
    ) -> None:
        self.pairs = pairs or {}
        self.revisions = revisions or {}
        self.allow_downloads = allow_downloads
        self.configured_pairs_only = configured_pairs_only
        self.fallback_hub_language = fallback_hub_language
        self.max_loaded_models = max(1, max_loaded_models)
        self.device = device
        self.dtype = dtype
        self.max_input_tokens = max_input_tokens
        self.decoding = decoding
        self._cache: OrderedDict[str, _MarianPairProvider] = OrderedDict()
        self.models_manifest: dict[str, dict[str, str | None]] = {}
        # Cache mutation and eviction must not race across parallel chunks.
        self._operation_lock = asyncio.Lock()

    def _key(self, source: str, target: str) -> str:
        return f"{source}-{target}"

    def _model_for(self, source: str, target: str) -> str | None:
        key = self._key(source, target)
        if key in self.pairs:
            return self.pairs[key]
        if self.allow_downloads and not self.configured_pairs_only:
            return f"Helsinki-NLP/opus-mt-{source}-{target}"
        return None

    def supports_pair(self, source_language: str, target_language: str) -> bool:
        if self._model_for(source_language, target_language):
            return True
        hub = self.fallback_hub_language
        return bool(
            hub
            and source_language != hub
            and target_language != hub
            and self._model_for(source_language, hub)
            and self._model_for(hub, target_language)
        )

    def pair_models(self, source_language: str, target_language: str) -> list[tuple[str, str]]:
        direct = self._model_for(source_language, target_language)
        if direct:
            return [(source_language, target_language)]
        hub = self.fallback_hub_language
        if (
            hub
            and source_language != hub
            and target_language != hub
            and self._model_for(source_language, hub)
            and self._model_for(hub, target_language)
        ):
            return [(source_language, hub), (hub, target_language)]
        raise ValueError(
            f"OPUS-MT has no configured model for {source_language}->{target_language}"
        )

    async def prepare_pair(self, source_language: str, target_language: str) -> None:
        """Resolve and load every required model before the first translated segment."""
        async with self._operation_lock:
            for source, target in self.pair_models(source_language, target_language):
                await asyncio.to_thread(self._get_pair_provider, source, target, True)

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None = None,
    ) -> TranslationResult:
        async with self._operation_lock:
            return await self._translate_locked(
                text,
                source_language,
                target_language,
                seed=seed,
            )

    async def _translate_locked(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None,
    ) -> TranslationResult:
        legs = self.pair_models(source_language, target_language)
        current = text
        results: list[TranslationResult] = []
        for index, (source, target) in enumerate(legs):
            pair_provider = self._get_pair_provider(source, target)
            result = await pair_provider.translate(
                current, source, target, seed=None if seed is None else seed + index
            )
            current = result.text
            self.models_manifest[self._key(source, target)] = {
                "model": pair_provider.model_name,
                "revision": pair_provider._resolved_revision,
            }
            results.append(result)
        models = [result.model for result in results]
        warnings = [warning for result in results for warning in result.warnings]
        if len(legs) > 1:
            warnings.append(
                f"OPUS-MT used hub language {self.fallback_hub_language} "
                f"for {source_language}->{target_language}"
            )
        return TranslationResult(
            text=current,
            provider=self.name,
            model=" + ".join(models),
            usage=_sum_usage(result.usage for result in results),
            warnings=warnings,
            deterministic=all(result.deterministic is True for result in results),
            metadata={
                "category": self.category,
                "legs": [
                    {
                        "source_language": source,
                        "target_language": target,
                        "model": result.model,
                        **result.metadata,
                    }
                    for (source, target), result in zip(legs, results, strict=True)
                ],
            },
        )

    def _get_pair_provider(
        self, source: str, target: str, load: bool = False
    ) -> _MarianPairProvider:
        key = self._key(source, target)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            if load:
                cached._load(source)
            return cached
        model = self._model_for(source, target)
        if not model:
            raise ValueError(f"OPUS-MT model is not configured for {key}")
        provider = _MarianPairProvider(
            source,
            target,
            model=model,
            revision=self.revisions.get(key),
            device=self.device,
            dtype=self.dtype,
            max_input_tokens=self.max_input_tokens,
            decoding=self.decoding,
            local_files_only=not self.allow_downloads,
        )
        if load:
            provider._load(source)
            self.models_manifest[key] = {
                "model": provider.model_name,
                "revision": provider._resolved_revision,
            }
        self._cache[key] = provider
        self._evict_if_needed()
        return provider

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_loaded_models:
            _, provider = self._cache.popitem(last=False)
            if provider._model is not None:
                provider._model.to("cpu")
                if provider._torch is not None and provider._torch.cuda.is_available():
                    provider._torch.cuda.empty_cache()


def _sum_usage(
    values: Any,
) -> dict[str, int | float] | None:
    total: dict[str, int | float] = {}
    for usage in values:
        if not usage:
            continue
        for key, value in usage.items():
            total[key] = total.get(key, 0) + value
    return total or None
