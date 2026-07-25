from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (
    ChunkingConfig,
    ContextConfig,
    EngineRoutingConfig,
    MemoryConfig,
    PipelineStageConfig,
    ProviderConfig,
    RouteConfig,
    RunConfig,
    RuntimeConfig,
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


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ConfigError(f"'{name}' must be greater than zero")


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
    chunking = ChunkingConfig(**chunking_raw)

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
    route = RouteConfig(**route_raw)

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
        context=ContextConfig(**_mapping(root.get("context"), "context")),
        memory=MemoryConfig(**_mapping(root.get("memory"), "memory")),
        runtime=RuntimeConfig(**_mapping(root.get("runtime"), "runtime")),
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
    _positive(config.chunking.max_chars, "chunking.max_chars")
    _positive(config.runtime.retries, "runtime.retries")
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
    _positive(config.memory.maximum_items_in_prompt, "memory.maximum_items_in_prompt")
    _positive(config.memory.minimum_count, "memory.minimum_count")
    if config.route.min_hops > config.route.max_hops:
        raise ConfigError("'translation.min_hops' cannot exceed 'translation.max_hops'")
    if config.runtime.failure_policy not in {"stop", "skip", "fallback"}:
        raise ConfigError("'runtime.failure_policy' must be stop, skip, or fallback")
    if config.runtime.failure_policy == "fallback" and config.runtime.fallback_provider != "mock":
        raise ConfigError("the built-in fallback_provider currently supports only 'mock'")
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
        _positive(stage.repeat, f"repeat for {stage.type}")
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
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        pipeline=stages,
        temperatures=dict(raw.get("temperatures", {})),
        custom_prompts=dict(raw.get("custom_prompts", {})),
        config_path=raw.get("config_path"),
    )
