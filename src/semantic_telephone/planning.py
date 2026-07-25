from __future__ import annotations

import importlib.util
import os
import random
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from .chunking import chunk_text
from .config import resolve_input_path
from .models import ProviderConfig, RunConfig, StageType
from .pipeline import load_prompts
from .providers.local_mt import M2M100_LANGUAGE_CODES, NLLB_LANGUAGE_CODES
from .providers.opus_mt import OpusMtTranslationProvider
from .providers.router import TranslationProviderRouter
from .routes import generate_route
from .runtime import safe_diagnostic


def create_plan(config: RunConfig) -> dict[str, Any]:
    load_prompts(config)
    source = resolve_input_path(config).read_text(encoding=config.input_encoding)
    chunks = chunk_text(source, config.chunking)
    translator = _offline_translation_provider(config.translation)
    planned_routes: list[dict[str, Any]] = []
    generation_calls = 0
    translation_hops = 0
    for chunk in chunks:
        stage_number = 0
        for stage in config.pipeline:
            for repetition in range(stage.repeat):
                stage_number += 1
                seed = (
                    stage.seed
                    if stage.seed is not None
                    else config.seed
                    + chunk.index * 10_000
                    + stage_number * 101
                    + repetition
                    + stage.seed_offset
                )
                if not stage.enabled or random.Random(seed).random() >= stage.probability:
                    continue
                if stage.type is StageType.TRANSLATION_CYCLE:
                    route_config = replace(
                        config.route,
                        min_hops=int(
                            stage.parameters.get("hops", {}).get(
                                "min", config.route.min_hops
                            )
                        ),
                        max_hops=int(
                            stage.parameters.get("hops", {}).get(
                                "max", config.route.max_hops
                            )
                        ),
                    )
                    route = generate_route(
                        route_config,
                        source_language=config.source_language,
                        target_language=config.target_language,
                        seed=seed,
                    )
                    engine_plan: list[list[str]] = []
                    if isinstance(translator, TranslationProviderRouter):
                        engine_plan = translator.plan_route(route, seed=seed)
                    else:
                        engine_plan = [
                            [getattr(translator, "name", config.translation.provider)]
                            for _ in pairwise(route)
                        ]
                    translation_hops += max(0, len(route) - 1)
                    planned_routes.append(
                        {
                            "chunk": chunk.index + 1,
                            "stage": stage_number,
                            "seed": seed,
                            "languages": route,
                            "engines": engine_plan,
                        }
                    )
                elif stage.type in {
                    StageType.CONSERVATIVE_REPAIR,
                    StageType.RECONSTRUCTION,
                    StageType.CONTEXTUAL_RECONSTRUCTION,
                    StageType.MEMORY_EXTRACTION,
                }:
                    generation_calls += 1
    report_config = config.generation.tasks.get("report_generation", config.generation)
    report_generation = report_config.provider != "mock"
    if report_generation:
        generation_calls += 1
    providers = (
        config.translation.providers
        if config.translation.providers
        else {config.translation.provider: config.translation}
    )
    resources = [
        {
            "alias": name,
            "type": provider.provider,
            "model": provider.model,
            "revision": provider.revision,
            "may_download": provider.provider in {"nllb", "m2m100"}
            or bool(provider.options.get("allow_downloads", False)),
            "configured_pairs": sorted(
                str(key)
                for key in (
                    provider.options.get("pairs", {})
                    if isinstance(provider.options.get("pairs"), dict)
                    else {}
                )
            ),
        }
        for name, provider in providers.items()
    ]
    generation_configs = {
        "generation": config.generation,
        **{
            f"generation.{task}": provider
            for task, provider in config.generation.tasks.items()
        },
    }
    resources.extend(
        {
            "alias": name,
            "type": provider.provider,
            "model": provider.model,
            "revision": provider.revision,
            "may_download": False,
            "configured_pairs": [],
        }
        for name, provider in generation_configs.items()
        if provider.provider != "mock"
    )
    if config.metrics.semantic.enabled:
        resources.append(
            {
                "alias": "metrics.semantic",
                "type": config.metrics.semantic.provider,
                "model": config.metrics.semantic.model,
                "revision": config.metrics.semantic.revision,
                "may_download": config.metrics.semantic.allow_downloads,
                "configured_pairs": [],
            }
        )
    remote_preflight_calls = sum(
        1
        for provider in providers.values()
        if provider.provider in {"local", "libretranslate"}
    )
    remote_services = sorted(
        {
            provider.provider
            for provider in providers.values()
            if provider.provider in {"local", "libretranslate"}
        }
        | {
            provider.provider
            for provider in generation_configs.values()
            if provider.provider in {"openrouter", "openai_compatible"}
        }
    )
    return {
        "schema_version": 1,
        "run": config.name,
        "seed": config.seed,
        "chunks": len(chunks),
        "effective_concurrency": (
            1 if config.context.enabled or config.memory.enabled else config.runtime.concurrency
        ),
        "planned_routes": planned_routes,
        "resources": resources,
        "remote_services": remote_services,
        "request_bounds": {
            "translation_hops": translation_hops,
            "generation_calls": generation_calls,
            "remote_preflight_calls": remote_preflight_calls,
            "provider_calls": translation_hops + generation_calls,
            "maximum_attempts": (
                translation_hops + generation_calls + remote_preflight_calls
            )
            * config.runtime.retries,
        },
        "budgets": asdict(config.runtime.budgets),
        "semantic_metrics": asdict(config.metrics.semantic),
        "warnings": (
            ["matrix concurrency multiplies per-run request rates"]
            if config.runtime.concurrency > 1 and remote_services
            else []
        ),
    }


class _PlannedProvider:
    def __init__(self, name: str, languages: set[str] | None = None) -> None:
        self.name = name
        self.languages = languages

    def supports_pair(self, source: str, target: str) -> bool:
        return source != target and (
            self.languages is None
            or (source in self.languages and target in self.languages)
        )


def _offline_translation_provider(config: ProviderConfig) -> Any:
    if config.providers:
        providers = {
            name: _offline_single_provider(provider)
            for name, provider in config.providers.items()
            if provider.enabled
        }
        return TranslationProviderRouter(
            providers,
            default_provider=config.default_provider or "",
            routing=config.engine_routing,
        )
    return _offline_single_provider(config)


def _offline_single_provider(config: ProviderConfig) -> Any:
    if config.provider == "nllb":
        return _PlannedProvider("nllb", set(NLLB_LANGUAGE_CODES))
    if config.provider == "m2m100":
        return _PlannedProvider("m2m100", set(M2M100_LANGUAGE_CODES))
    if config.provider in {"opus", "opus_mt"}:
        pairs = config.options.get("pairs", {})
        revisions = config.options.get("revisions", {})
        return OpusMtTranslationProvider(
            pairs=(
                {str(key): str(value) for key, value in pairs.items()}
                if isinstance(pairs, dict)
                else {}
            ),
            revisions=(
                {str(key): str(value) for key, value in revisions.items()}
                if isinstance(revisions, dict)
                else {}
            ),
            allow_downloads=bool(config.options.get("allow_downloads", False)),
            configured_pairs_only=bool(
                config.options.get("configured_pairs_only", False)
            ),
            fallback_hub_language=(
                str(config.options.get("fallback_hub_language", "en"))
                if config.options.get("fallback_hub_language", "en") is not None
                else None
            ),
        )
    return _PlannedProvider(config.provider)


async def doctor_config(
    config: RunConfig,
    *,
    allow_downloads: bool = False,
) -> dict[str, Any]:
    dotenv_path = (
        Path(config.config_path).parent / ".env"
        if config.config_path is not None
        else None
    )
    load_dotenv(dotenv_path=dotenv_path if dotenv_path and dotenv_path.exists() else None)
    checks: list[dict[str, str]] = []
    try:
        load_prompts(config)
        checks.append({"name": "prompts", "status": "pass", "detail": "all prompts resolve"})
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        checks.append(
            {
                "name": "prompts",
                "status": "fail",
                "detail": safe_diagnostic(error),
            }
        )
    try:
        plan = create_plan(config)
        checks.append(
            {
                "name": "route and pair support",
                "status": "pass",
                "detail": (
                    f"{len(plan['planned_routes'])} routes; "
                    f"{plan['request_bounds']['translation_hops']} translation hops"
                ),
            }
        )
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        checks.append(
            {
                "name": "route and pair support",
                "status": "fail",
                "detail": safe_diagnostic(error),
            }
        )
    provider_types = {
        provider.provider
        for provider in (
            config.translation.providers.values()
            if config.translation.providers
            else [config.translation]
        )
    }
    if provider_types & {"nllb", "m2m100", "opus", "opus_mt"}:
        missing = [
            name
            for name in ("torch", "transformers")
            if importlib.util.find_spec(name) is None
        ]
        checks.append(
            {
                "name": "local-mt dependencies",
                "status": "fail" if missing else "pass",
                "detail": "missing: " + ", ".join(missing) if missing else "available",
            }
        )
        if not missing:
            try:
                import torch

                requested_devices = {
                    str(provider.options.get("device", "auto"))
                    for provider in (
                        config.translation.providers.values()
                        if config.translation.providers
                        else [config.translation]
                    )
                    if provider.provider in {"nllb", "m2m100", "opus", "opus_mt"}
                }
                cuda = bool(torch.cuda.is_available())
                impossible = "cuda" in requested_devices and not cuda
                checks.append(
                    {
                        "name": "compute device",
                        "status": "fail" if impossible else "pass",
                        "detail": (
                            f"requested={','.join(sorted(requested_devices))}; "
                            f"cuda_available={cuda}"
                        ),
                    }
                )
            except Exception as error:  # noqa: BLE001 - diagnostic boundary
                checks.append(
                    {
                        "name": "compute device",
                        "status": "fail",
                        "detail": safe_diagnostic(error),
                    }
                )
    if config.metrics.semantic.enabled:
        available = importlib.util.find_spec("sentence_transformers") is not None
        checks.append(
            {
                "name": "semantic metrics dependency",
                "status": "pass" if available else "fail",
                "detail": "available" if available else "sentence-transformers is not installed",
            }
        )
    checks.extend(await _remote_checks(config))
    checks.extend(_cache_checks(config, allow_downloads=allow_downloads))
    return {
        "schema_version": 1,
        "run": config.name,
        "allow_downloads": allow_downloads,
        "checks": checks,
        "ok": not any(item["status"] == "fail" for item in checks),
    }


async def _remote_checks(config: RunConfig) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    providers = (
        config.translation.providers.values()
        if config.translation.providers
        else [config.translation]
    )
    for provider in providers:
        if provider.provider not in {"local", "libretranslate"}:
            continue
        base_url = (
            provider.base_url
            or os.getenv("LIBRETRANSLATE_BASE_URL")
            or "http://localhost:5000"
        ).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=provider.timeout_seconds) as client:
                response = await client.get(f"{base_url}/languages")
                response.raise_for_status()
            checks.append(
                {"name": "LibreTranslate", "status": "pass", "detail": base_url}
            )
        except Exception as error:  # noqa: BLE001 - diagnostic boundary
            checks.append(
                {
                    "name": "LibreTranslate",
                    "status": "fail",
                    "detail": f"{base_url}: {safe_diagnostic(error)}",
                }
            )
    generation_configs = [config.generation, *config.generation.tasks.values()]
    seen: set[tuple[str, str]] = set()
    for provider in generation_configs:
        if provider.provider not in {"openrouter", "openai_compatible"}:
            continue
        key_env = provider.api_key_env or (
            "OPENROUTER_API_KEY" if provider.provider == "openrouter" else ""
        )
        key = os.getenv(key_env) if key_env else None
        base_url = (
            provider.base_url
            or (
                os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
                if provider.provider == "openrouter"
                else ""
            )
        ).rstrip("/")
        identity = (provider.provider, base_url)
        if identity in seen:
            continue
        seen.add(identity)
        if not key or not base_url:
            checks.append(
                {
                    "name": provider.provider,
                    "status": "fail",
                    "detail": f"missing endpoint or credential environment variable {key_env}",
                }
            )
            continue
        try:
            async with httpx.AsyncClient(timeout=provider.timeout_seconds) as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
                response.raise_for_status()
            checks.append(
                {"name": provider.provider, "status": "pass", "detail": base_url}
            )
        except Exception as error:  # noqa: BLE001 - diagnostic boundary
            checks.append(
                {
                    "name": provider.provider,
                    "status": "fail",
                    "detail": f"{base_url}: {safe_diagnostic(error)}",
                }
            )
    return checks


def _cache_checks(config: RunConfig, *, allow_downloads: bool) -> list[dict[str, str]]:
    repositories: dict[str, str | None] = {}
    providers = (
        config.translation.providers.values()
        if config.translation.providers
        else [config.translation]
    )
    for provider in providers:
        if provider.provider in {"nllb", "m2m100"}:
            repositories[provider.model] = provider.revision
        if provider.provider in {"opus", "opus_mt"}:
            pairs = provider.options.get("pairs", {})
            revisions = provider.options.get("revisions", {})
            if isinstance(pairs, dict):
                for pair, model in pairs.items():
                    revision = revisions.get(pair) if isinstance(revisions, dict) else None
                    repositories[str(model)] = str(revision) if revision else None
    semantic = config.metrics.semantic
    if semantic.enabled:
        repositories[semantic.model] = semantic.revision
    if not repositories:
        return []
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return [
            {
                "name": "Hugging Face cache",
                "status": "fail",
                "detail": "huggingface-hub is unavailable",
            }
        ]
    checks: list[dict[str, str]] = []
    for model, revision in repositories.items():
        try:
            snapshot_download(
                repo_id=model,
                revision=revision,
                local_files_only=not allow_downloads,
            )
        except Exception as error:  # noqa: BLE001 - cache/download diagnostic boundary
            checks.append(
                {
                    "name": model,
                    "status": "fail",
                    "detail": (
                        safe_diagnostic(error)
                        if allow_downloads
                        else "configured revision is not cached; use --allow-downloads"
                    ),
                }
            )
            continue
        checks.append(
            {
                "name": model,
                "status": "pass",
                "detail": (
                    f"available revision {revision or 'configured default'}"
                    + (" (downloads allowed)" if allow_downloads else " (cached)")
                ),
            }
        )
    return checks
