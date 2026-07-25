from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from .models import (
    BudgetConfig,
    ChunkingConfig,
    ContextConfig,
    EngineRoutingConfig,
    MemoryConfig,
    MetricsConfig,
    PipelineStageConfig,
    ProviderConfig,
    RouteConfig,
    RunConfig,
    RuntimeConfig,
    SemanticMetricConfig,
    StageType,
)


class ConfigError(ValueError):
    pass


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    return value


def _positive(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ConfigError(f"'{name}' must be greater than zero")


def _positive_integer(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{name}' must be a positive integer")


def _construct[T](factory: type[T], raw: dict[str, Any], name: str) -> T:
    try:
        return factory(**raw)
    except TypeError as error:
        raise ConfigError(f"invalid '{name}': {error}") from error


def _provider_config(value: Any, name: str, *, inherited_provider: str = "mock") -> ProviderConfig:
    raw = _mapping(value, name).copy()
    provider = str(raw.pop("provider", raw.pop("type", inherited_provider)))
    nested_providers = {
        str(key): _provider_config(item, f"{name}.providers.{key}")
        for key, item in _mapping(raw.pop("providers", {}), f"{name}.providers").items()
    }
    tasks = {
        str(key): _provider_config(item, f"{name}.tasks.{key}", inherited_provider=provider)
        for key, item in _mapping(raw.pop("tasks", {}), f"{name}.tasks").items()
    }
    routing_raw = _mapping(raw.pop("engine_routing", {}), f"{name}.engine_routing")
    try:
        routing = EngineRoutingConfig(**routing_raw)
    except TypeError as error:
        raise ConfigError(f"invalid '{name}.engine_routing': {error}") from error
    known_names = {
        "enabled",
        "base_url",
        "model",
        "revision",
        "api_key_env",
        "timeout_seconds",
        "options",
        "default_provider",
    }
    known = {key: raw.pop(key) for key in list(raw) if key in known_names}
    explicit_options = _mapping(known.pop("options", {}), f"{name}.options")
    options = {**explicit_options, **raw}
    try:
        return ProviderConfig(
            provider=provider,
            providers=nested_providers,
            tasks=tasks,
            engine_routing=routing,
            options=options,
            **known,
        )
    except TypeError as error:
        raise ConfigError(f"invalid provider configuration '{name}': {error}") from error


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"configuration file not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML: {error}") from error
    root = _mapping(raw, "root")
    run = _mapping(root.get("run"), "run")
    input_section = _mapping(root.get("input"), "input")
    if not run.get("name"):
        raise ConfigError("'run.name' is required")
    if not input_section.get("path"):
        raise ConfigError("'input.path' is required")

    chunking_raw = _mapping(root.get("chunking"), "chunking")
    chunking = _construct(ChunkingConfig, chunking_raw, "chunking")

    translation_raw = _mapping(root.get("translation"), "translation").copy()
    route_keys = {
        "route_mode",
        "languages",
        "allow",
        "deny",
        "min_hops",
        "max_hops",
        "hub_language",
        "hub_frequency",
        "mutations",
        "per",
        "return_to_target_after_cycle",
    }
    route_raw = {
        key: translation_raw.pop(key) for key in list(translation_raw) if key in route_keys
    }
    if "route_mode" in route_raw:
        route_raw["mode"] = route_raw.pop("route_mode")
    languages = route_raw.get("languages")
    if isinstance(languages, dict):
        route_raw["allow"] = languages.get("allow", [])
        route_raw["deny"] = languages.get("deny", [])
        route_raw["languages"] = []
    translation = _provider_config(translation_raw, "translation")
    route = _construct(RouteConfig, route_raw, "translation route")

    generation_raw = _mapping(root.get("generation"), "generation").copy()
    temperature = generation_raw.pop("temperature", {})
    generation = _provider_config(generation_raw, "generation")
    temperatures = {
        "conservative_repair": 0.3,
        "reconstruction": 0.7,
        "contextual_reconstruction": 0.7,
        "memory_extraction": 0.1,
        "report_generation": 0.2,
        **_mapping(temperature, "generation.temperature"),
    }
    if "repair" in temperatures:
        temperatures["conservative_repair"] = temperatures["repair"]

    stage_items = root.get("pipeline", [])
    if not isinstance(stage_items, list) or not stage_items:
        raise ConfigError("'pipeline' must be a non-empty list")
    pipeline: list[PipelineStageConfig] = []
    for index, item in enumerate(stage_items):
        stage_raw = _mapping(item, f"pipeline[{index}]").copy()
        try:
            stage_type = StageType(stage_raw.pop("type"))
        except (KeyError, ValueError) as error:
            valid = ", ".join(stage.value for stage in StageType)
            raise ConfigError(
                f"invalid pipeline stage at index {index}; expected one of {valid}"
            ) from error
        known = {"enabled", "probability", "repeat", "seed", "seed_offset"}
        parameters = {key: stage_raw.pop(key) for key in list(stage_raw) if key not in known}
        pipeline.append(PipelineStageConfig(type=stage_type, parameters=parameters, **stage_raw))

    runtime_raw = _mapping(root.get("runtime"), "runtime").copy()
    if "resume" in runtime_raw:
        warnings.warn(
            "'runtime.resume' is deprecated and will be removed in v1.0; "
            "use the explicit resume command",
            FutureWarning,
            stacklevel=2,
        )
    budgets = _construct(
        BudgetConfig,
        _mapping(runtime_raw.pop("budgets", {}), "runtime.budgets"),
        "runtime.budgets",
    )
    runtime = _construct(RuntimeConfig, {**runtime_raw, "budgets": budgets}, "runtime")
    metrics_raw = _mapping(root.get("metrics"), "metrics").copy()
    semantic = _construct(
        SemanticMetricConfig,
        _mapping(metrics_raw.pop("semantic", {}), "metrics.semantic"),
        "metrics.semantic",
    )
    if metrics_raw:
        raise ConfigError("invalid 'metrics': unexpected fields " + ", ".join(sorted(metrics_raw)))
    config = RunConfig(
        name=str(run["name"]),
        seed=int(run.get("seed", 0)),
        source_language=str(run.get("source_language", "auto")),
        target_language=str(run.get("target_language", run.get("source_language", "en"))),
        input_path=str(input_section["path"]),
        input_encoding=str(input_section.get("encoding", "utf-8")),
        output_root=str(run.get("output_root", "runs")),
        chunking=chunking,
        translation=translation,
        generation=generation,
        route=route,
        context=_construct(ContextConfig, _mapping(root.get("context"), "context"), "context"),
        memory=_construct(MemoryConfig, _mapping(root.get("memory"), "memory"), "memory"),
        runtime=runtime,
        metrics=MetricsConfig(semantic=semantic),
        pipeline=pipeline,
        temperatures=temperatures,
        custom_prompts={
            str(key): str(value) for key, value in _mapping(root.get("prompts"), "prompts").items()
        },
        config_path=str(config_path),
    )
    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    strategies = {"paragraph", "target_chars", "target_tokens", "sentence_window"}
    if config.chunking.strategy not in strategies:
        raise ConfigError(f"unknown chunking strategy: {config.chunking.strategy}")
    route_modes = {"fixed", "random", "stratified", "hubbed", "mutating_fixed"}
    if config.route.mode not in route_modes:
        raise ConfigError(f"unknown route mode: {config.route.mode}")
    _positive_integer(config.chunking.max_chars, "chunking.max_chars")
    _positive_integer(config.chunking.target_chars, "chunking.target_chars")
    _positive_integer(config.chunking.target_tokens, "chunking.target_tokens")
    _positive_integer(config.chunking.sentence_window, "chunking.sentence_window")
    if (
        isinstance(config.chunking.paragraph_overlap, bool)
        or not isinstance(config.chunking.paragraph_overlap, int)
        or config.chunking.paragraph_overlap < 0
    ):
        raise ConfigError("'chunking.paragraph_overlap' must be a non-negative integer")
    _positive_integer(config.runtime.retries, "runtime.retries")
    rpm = config.runtime.requests_per_minute
    if rpm is not None and (
        isinstance(rpm, bool) or not isinstance(rpm, int) or rpm < 0
    ):
        raise ConfigError("'runtime.requests_per_minute' must be null or a non-negative integer")
    if (
        isinstance(config.runtime.concurrency, bool)
        or not isinstance(config.runtime.concurrency, int)
        or config.runtime.concurrency <= 0
    ):
        raise ConfigError("'runtime.concurrency' must be a positive integer")
    if config.context.enabled:
        _positive(config.context.previous_chunks, "context.previous_chunks")
        _positive(config.context.max_chars, "context.max_chars")
    if config.context.truncation not in {"head", "tail"}:
        raise ConfigError("'context.truncation' must be head or tail")
    context_sources = {
        "final_generated_text",
        "intermediate_generations",
        "all_generated_text",
    }
    if config.context.source not in context_sources:
        raise ConfigError(
            "'context.source' must be final_generated_text, "
            "intermediate_generations, or all_generated_text"
        )
    _positive(config.memory.half_life_chunks, "memory.half_life_chunks")
    _positive_integer(
        config.memory.maximum_items_in_prompt, "memory.maximum_items_in_prompt"
    )
    _positive_integer(config.memory.minimum_count, "memory.minimum_count")
    _positive_integer(config.route.min_hops, "translation.min_hops")
    _positive_integer(config.route.max_hops, "translation.max_hops")
    if config.route.min_hops > config.route.max_hops:
        raise ConfigError("'translation.min_hops' cannot exceed 'translation.max_hops'")
    if config.runtime.failure_policy not in {"stop", "skip", "fallback"}:
        raise ConfigError("'runtime.failure_policy' must be stop, skip, or fallback")
    if config.runtime.failure_policy == "fallback" and config.runtime.fallback_provider != "mock":
        raise ConfigError("the built-in fallback_provider currently supports only 'mock'")
    for name, value in (
        ("max_requests", config.runtime.budgets.max_requests),
        ("max_total_tokens", config.runtime.budgets.max_total_tokens),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ConfigError(f"'runtime.budgets.{name}' must be null or a non-negative integer")
    cost = config.runtime.budgets.max_cost_usd
    if cost is not None and (
        isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0
    ):
        raise ConfigError(
            "'runtime.budgets.max_cost_usd' must be null or a non-negative number"
        )
    semantic = config.metrics.semantic
    if semantic.provider != "sentence_transformers":
        raise ConfigError("'metrics.semantic.provider' must be sentence_transformers")
    _positive_integer(semantic.batch_size, "metrics.semantic.batch_size")
    _validate_provider(config.translation, "translation", translation=True)
    _validate_provider(config.generation, "generation", translation=False)
    routing_modes = {
        "single_engine",
        "fixed_engine_route",
        "weighted_random",
        "alternating",
        "quality_fallback",
        "heterogeneous",
    }
    if config.translation.providers:
        routing = config.translation.engine_routing
        if routing.mode not in routing_modes:
            raise ConfigError(f"unknown engine routing mode: {routing.mode}")
        enabled = {
            name for name, provider in config.translation.providers.items() if provider.enabled
        }
        default_provider = config.translation.default_provider
        if not default_provider:
            raise ConfigError("'translation.default_provider' is required with multiple providers")
        if default_provider not in enabled:
            raise ConfigError(
                "'translation.default_provider' must name an enabled configured provider"
            )
        referenced = (
            set(routing.route)
            | set(routing.pairs.values())
            | set(routing.weights)
            | set(routing.fallback_order)
        )
        if routing.provider:
            referenced.add(routing.provider)
        unknown = referenced - enabled
        if unknown:
            raise ConfigError(
                "engine routing references unknown or disabled providers: "
                + ", ".join(sorted(unknown))
            )
        if routing.mode == "weighted_random":
            if not routing.weights or sum(routing.weights.values()) <= 0:
                raise ConfigError("'translation.engine_routing.weights' must have positive total")
            if any(weight < 0 for weight in routing.weights.values()):
                raise ConfigError("engine routing weights cannot be negative")
    for stage in config.pipeline:
        if not 0 <= stage.probability <= 1:
            raise ConfigError(f"stage probability must be 0..1: {stage.type}")
        _positive_integer(stage.repeat, f"repeat for {stage.type}")
        if "max_length_ratio" in stage.parameters:
            _positive(
                float(stage.parameters["max_length_ratio"]),
                f"max_length_ratio for {stage.type}",
            )
        if "max_new_sentences_per_chunk" in stage.parameters:
            value = stage.parameters["max_new_sentences_per_chunk"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigError(
                    f"'max_new_sentences_per_chunk' for {stage.type} "
                    "must be a non-negative integer"
                )
        repetition_policy = stage.parameters.get("repetition_policy")
        if repetition_policy not in {None, "preserve", "rationalize"}:
            raise ConfigError(
                f"'repetition_policy' for {stage.type} must be preserve or rationalize"
            )
        for name in ("allow_new_events", "allow_scene_expansion"):
            if name in stage.parameters and not isinstance(stage.parameters[name], bool):
                raise ConfigError(f"'{name}' for {stage.type} must be a boolean")
    if not config.memory.enabled and any(
        stage.type is StageType.MEMORY_EXTRACTION and stage.enabled for stage in config.pipeline
    ):
        raise ConfigError("memory_extraction stage requires memory.enabled: true")
    guaranteed_translation = False
    for stage in config.pipeline:
        if stage.type is StageType.MEMORY_EXTRACTION and stage.enabled:
            if not guaranteed_translation:
                raise ConfigError(
                    "memory_extraction requires an earlier enabled translation_cycle "
                    "with probability: 1"
                )
        elif stage.type is StageType.TRANSLATION_CYCLE and stage.enabled and stage.probability == 1:
            guaranteed_translation = True


def _validate_provider(config: ProviderConfig, name: str, *, translation: bool) -> None:
    _positive(config.timeout_seconds, f"{name}.timeout_seconds")
    supported = (
        {
            "mock",
            "local",
            "libretranslate",
            "nllb",
            "m2m100",
            "opus",
            "opus_mt",
            "google_cloud",
            "deepl",
            "azure_translator",
            "yandex_translate",
            "commercial_nmt",
        }
        if translation
        else {"mock", "openrouter", "openai_compatible"}
    )
    if not config.providers and config.provider not in supported:
        raise ConfigError(f"unknown {name} provider: {config.provider}")
    for key in ("max_input_tokens", "max_loaded_models", "retries", "max_tokens"):
        if key in config.options:
            _positive_integer(config.options[key], f"{name}.{key}")
    if "retry_backoff_seconds" in config.options:
        value = config.options["retry_backoff_seconds"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
        ):
            raise ConfigError(f"'{name}.retry_backoff_seconds' must be non-negative")
    for key in ("allow_downloads", "configured_pairs_only", "local_files_only"):
        if key in config.options and not isinstance(config.options[key], bool):
            raise ConfigError(f"'{name}.{key}' must be a boolean")
    decoding = config.options.get("decoding")
    if decoding is not None:
        if not isinstance(decoding, dict):
            raise ConfigError(f"'{name}.decoding' must be a mapping")
        for key in ("num_beams", "max_new_tokens"):
            if key in decoding:
                _positive_integer(decoding[key], f"{name}.decoding.{key}")
        if "no_repeat_ngram_size" in decoding:
            value = decoding["no_repeat_ngram_size"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(
                    f"'{name}.decoding.no_repeat_ngram_size' must be non-negative"
                )
    for alias, provider in config.providers.items():
        _validate_provider(provider, f"{name}.providers.{alias}", translation=translation)
    for task, provider in config.tasks.items():
        _validate_provider(provider, f"{name}.tasks.{task}", translation=translation)


def resolve_input_path(config: RunConfig) -> Path:
    path = Path(config.input_path)
    if path.is_absolute() or config.config_path is None:
        return path
    return Path(config.config_path).parent / path


def resolve_output_root(config: RunConfig) -> Path:
    path = Path(config.output_root)
    if path.is_absolute() or config.config_path is None:
        return path
    parent = Path(config.config_path).parent
    base = parent.parent if parent.name == "configs" else parent
    return base / path


def config_from_resolved(raw: dict[str, Any]) -> RunConfig:
    """Rehydrate the internal resolved representation stored in the manifest."""
    stages = [
        PipelineStageConfig(
            type=StageType(item["type"]),
            enabled=bool(item.get("enabled", True)),
            parameters=dict(item.get("parameters", {})),
            probability=float(item.get("probability", 1.0)),
            repeat=int(item.get("repeat", 1)),
            seed=item.get("seed"),
            seed_offset=int(item.get("seed_offset", 0)),
        )
        for item in raw.get("pipeline", [])
    ]
    return RunConfig(
        name=str(raw["name"]),
        seed=int(raw["seed"]),
        source_language=str(raw["source_language"]),
        target_language=str(raw["target_language"]),
        input_path=str(raw["input_path"]),
        input_encoding=str(raw.get("input_encoding", "utf-8")),
        output_root=str(raw.get("output_root", "runs")),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        translation=_provider_config(raw.get("translation", {}), "translation"),
        generation=_provider_config(raw.get("generation", {}), "generation"),
        route=RouteConfig(**raw.get("route", {})),
        context=ContextConfig(**raw.get("context", {})),
        memory=MemoryConfig(**raw.get("memory", {})),
        runtime=_runtime_from_resolved(raw.get("runtime", {})),
        metrics=_metrics_from_resolved(raw.get("metrics", {})),
        pipeline=stages,
        temperatures=dict(raw.get("temperatures", {})),
        custom_prompts=dict(raw.get("custom_prompts", {})),
        config_path=raw.get("config_path"),
    )


def _runtime_from_resolved(raw: Any) -> RuntimeConfig:
    value = _mapping(raw, "runtime").copy()
    budgets = _construct(
        BudgetConfig,
        _mapping(value.pop("budgets", {}), "runtime.budgets"),
        "runtime.budgets",
    )
    return _construct(RuntimeConfig, {**value, "budgets": budgets}, "runtime")


def _metrics_from_resolved(raw: Any) -> MetricsConfig:
    value = _mapping(raw, "metrics").copy()
    semantic = _construct(
        SemanticMetricConfig,
        _mapping(value.pop("semantic", {}), "metrics.semantic"),
        "metrics.semantic",
    )
    if value:
        raise ConfigError("invalid 'metrics': unexpected fields " + ", ".join(sorted(value)))
    return MetricsConfig(semantic=semantic)
