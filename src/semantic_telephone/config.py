from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (
    ChunkingConfig,
    ContextConfig,
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

    translation_raw = _mapping(root.get("translation"), "translation")
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
    route_raw = {key: translation_raw.pop(key) for key in list(translation_raw) if key in route_keys}
    if "route_mode" in route_raw:
        route_raw["mode"] = route_raw.pop("route_mode")
    languages = route_raw.get("languages")
    if isinstance(languages, dict):
        route_raw["allow"] = languages.get("allow", [])
        route_raw["deny"] = languages.get("deny", [])
        route_raw["languages"] = []
    translation = ProviderConfig(**translation_raw)
    route = RouteConfig(**route_raw)

    generation_raw = _mapping(root.get("generation"), "generation")
    temperature = generation_raw.pop("temperature", {})
    generation = ProviderConfig(**generation_raw)
    temperatures = {
        "repair": 0.3,
        "reconstruction": 0.7,
        "memory_extraction": 0.1,
        **_mapping(temperature, "generation.temperature"),
    }

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
            raise ConfigError(f"invalid pipeline stage at index {index}; expected one of {valid}") from error
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
            str(key): str(value)
            for key, value in _mapping(root.get("prompts"), "prompts").items()
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
    _positive(config.runtime.concurrency, "runtime.concurrency")
    if config.route.min_hops > config.route.max_hops:
        raise ConfigError("'translation.min_hops' cannot exceed 'translation.max_hops'")
    if config.runtime.failure_policy not in {"stop", "skip", "fallback"}:
        raise ConfigError("'runtime.failure_policy' must be stop, skip, or fallback")
    if (
        config.runtime.failure_policy == "fallback"
        and config.runtime.fallback_provider != "mock"
    ):
        raise ConfigError("the built-in fallback_provider currently supports only 'mock'")
    for stage in config.pipeline:
        if not 0 <= stage.probability <= 1:
            raise ConfigError(f"stage probability must be 0..1: {stage.type}")
        _positive(stage.repeat, f"repeat for {stage.type}")
    if not config.memory.enabled and any(
        stage.type is StageType.MEMORY_EXTRACTION and stage.enabled for stage in config.pipeline
    ):
        raise ConfigError("memory_extraction stage requires memory.enabled: true")


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
        translation=ProviderConfig(**raw.get("translation", {})),
        generation=ProviderConfig(**raw.get("generation", {})),
        route=RouteConfig(**raw.get("route", {})),
        context=ContextConfig(**raw.get("context", {})),
        memory=MemoryConfig(**raw.get("memory", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        pipeline=stages,
        temperatures=dict(raw.get("temperatures", {})),
        custom_prompts=dict(raw.get("custom_prompts", {})),
        config_path=raw.get("config_path"),
    )
