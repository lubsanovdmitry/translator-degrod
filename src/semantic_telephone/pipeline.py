from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import platform
import random
import re
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from . import __version__
from .checkpoints import CheckpointStore
from .chunking import assemble_chunks, chunk_text
from .config import resolve_input_path, resolve_output_root
from .memory import MemoryStore, validate_memory_payload
from .metrics import calculate_metrics, prepare_semantic_metrics
from .models import (
    Chunk,
    GenerationResult,
    PipelineStageConfig,
    RunConfig,
    RunManifest,
    StageResult,
    StageType,
    TranslationResult,
)
from .providers.factory import generation_provider, translation_provider
from .providers.mock import MockGenerationProvider, MockTranslationProvider
from .reporting import create_report
from .resources import prompt_text
from .routes import generate_route
from .runtime import BudgetExceededError, RequestController, safe_diagnostic
from .stages.context import rolling_context
from .stages.memory import extract_memory, memory_prompt
from .stages.reconstruction import reconstruct_text
from .stages.repair import repair_text
from .stages.translation import translate_route
from .utils.files import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    read_json,
)
from .utils.hashing import checksum_data, checksum_text
from .utils.retry import with_retry

logger = logging.getLogger("semantic_telephone")

PROMPT_FILES = {
    StageType.CONSERVATIVE_REPAIR: "grammar_repair.txt",
    StageType.RECONSTRUCTION: "conservative_reconstruction.txt",
    StageType.CONTEXTUAL_RECONSTRUCTION: "contextual_reconstruction.txt",
    StageType.MEMORY_EXTRACTION: "memory_extraction.txt",
}


@dataclass(slots=True)
class _ChunkResult:
    chunk_id: str
    index: int
    final_text: str
    context_value: str
    routes: list[list[str]]
    warnings: list[str]


def load_prompts(config: RunConfig) -> tuple[dict[StageType, str], dict[str, str]]:
    prompts: dict[StageType, str] = {}
    checksums: dict[str, str] = {}
    for stage_type, filename in PROMPT_FILES.items():
        custom = config.custom_prompts.get(stage_type.value)
        if custom:
            if custom.startswith("builtin:"):
                text = prompt_text(custom.split(":", 1)[1])
            else:
                path = Path(custom)
                if not path.is_absolute() and config.config_path:
                    path = Path(config.config_path).parent / path
                text = path.read_text(encoding="utf-8")
        else:
            text = prompt_text(Path(filename).stem)
        prompts[stage_type] = text
        checksums[stage_type.value] = checksum_text(text)
    report_custom = config.custom_prompts.get("report_generation")
    if report_custom and report_custom.startswith("builtin:"):
        report_prompt = prompt_text(report_custom.split(":", 1)[1])
    elif report_custom:
        report_path = Path(report_custom)
        if not report_path.is_absolute() and config.config_path:
            report_path = Path(config.config_path).parent / report_path
        report_prompt = report_path.read_text(encoding="utf-8")
    else:
        report_prompt = prompt_text("report_generation")
    checksums["report_generation"] = checksum_text(report_prompt)
    return prompts, checksums


def _new_run_directory(config: RunConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = resolve_output_root(config) / f"{timestamp}-{_slug(config.name)}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}-{suffix}")
        suffix += 1
    return candidate


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "run"


def _manifest_paths(run_directory: Path) -> dict[str, str]:
    return {
        "run_directory": str(run_directory.resolve()),
        "source": "source.txt",
        "final": "final.txt",
        "events": "events.jsonl",
        "metrics": "metrics.json",
        "report": "report.md",
    }


async def run_pipeline(config: RunConfig, *, run_directory: Path | None = None) -> Path:
    run_directory = run_directory.resolve() if run_directory else _new_run_directory(config)
    run_directory.mkdir(parents=True, exist_ok=True)
    source_path = resolve_input_path(config)
    source = (
        (run_directory / "source.txt").read_text(encoding=config.input_encoding)
        if (run_directory / "source.txt").exists()
        else source_path.read_text(encoding=config.input_encoding)
    )
    if not source.strip():
        raise ValueError("input text is empty")

    manifest_path = run_directory / "manifest.json"
    existing_manifest = read_json(manifest_path) if manifest_path.exists() else None
    if (
        existing_manifest is not None
        and int(existing_manifest.get("artifact_schema_version", 1)) < 2
        and config.memory.enabled
    ):
        raise RuntimeError(
            "cannot safely resume a schema-v1 run with memory enabled; "
            "start a new run so memory payloads can be checkpointed"
        )
    prompts, prompt_checksums = load_prompts(config)
    semantic_metrics = (
        await asyncio.to_thread(prepare_semantic_metrics, config.metrics.semantic)
        if config.metrics.semantic.enabled
        else None
    )
    atomic_write_text(run_directory / "source.txt", source)
    atomic_write_yaml(run_directory / "resolved_config.yaml", config.to_dict())
    if existing_manifest is not None:
        manifest = existing_manifest
        manifest["state"] = "running"
    else:
        created = RunManifest.create(
            config,
            version=__version__,
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "pid": str(os.getpid()),
            },
            paths=_manifest_paths(run_directory),
            prompt_checksums=prompt_checksums,
        )
        manifest = created.to_dict()
    atomic_write_json(manifest_path, manifest)

    request_controller = RequestController(config.runtime, run_directory / "events.jsonl")
    translator = (
        translation_provider(config.translation, request_controller=request_controller)
        if "request_controller" in inspect.signature(translation_provider).parameters
        else translation_provider(config.translation)
    )
    generators: dict[str, Any] = {}
    generation_stage_types = {
        StageType.CONSERVATIVE_REPAIR,
        StageType.RECONSTRUCTION,
        StageType.CONTEXTUAL_RECONSTRUCTION,
        StageType.MEMORY_EXTRACTION,
    }
    for stage_type in {
        stage.type for stage in config.pipeline if stage.enabled
    } & generation_stage_types:
        generators[stage_type.value] = generation_provider(
            config.generation,
            task=stage_type.value,
            request_controller=request_controller,
        )
    chunks = chunk_text(source, config.chunking)
    # Rebuild memory causally from stage checkpoints. Loading the final state here
    # would expose later chunks to earlier reconstruction stages during resume.
    memory = MemoryStore(
        run_directory / "memory",
        half_life=config.memory.half_life_chunks,
        load_items=False,
    )
    previous_contexts: list[str] = []
    final_values: list[str] = []
    all_routes: list[list[str]] = []
    warnings: list[str] = []
    chunk_concurrency = _effective_chunk_concurrency(config)
    manifest["effective_chunk_concurrency"] = chunk_concurrency
    routes_serialized = bool(getattr(translator, "serializes_routes", False))
    manifest["translation_routes_serialized"] = routes_serialized
    if routes_serialized:
        logger.info("serializing translation routes to protect a bounded pair-model cache")
    if chunk_concurrency < config.runtime.concurrency:
        dependency = "rolling context" if config.context.enabled else "memory"
        warnings.append(
            f"runtime.concurrency reduced to 1 because {dependency} requires chunk order"
        )
        logger.info(
            "using sequential chunk execution because %s is enabled",
            dependency,
        )
    atomic_write_json(manifest_path, manifest)
    try:
        if chunk_concurrency == 1:
            for chunk in chunks:
                result = await _run_chunk(
                    chunk=chunk,
                    config=config,
                    run_directory=run_directory,
                    previous_contexts=previous_contexts,
                    prompts=prompts,
                    prompt_checksums=prompt_checksums,
                    translator=translator,
                    generators=generators,
                    memory=memory,
                )
                previous_contexts.append(result.context_value)
                _record_chunk_result(
                    result,
                    final_values=final_values,
                    all_routes=all_routes,
                    warnings=warnings,
                    manifest=manifest,
                )
                atomic_write_json(manifest_path, manifest)
                logger.info("completed chunk %d/%d", chunk.index + 1, len(chunks))
        else:
            chunk_results = await _run_chunks_concurrently(
                chunks=chunks,
                concurrency=chunk_concurrency,
                config=config,
                run_directory=run_directory,
                prompts=prompts,
                prompt_checksums=prompt_checksums,
                translator=translator,
                generators=generators,
                memory=memory,
            )
            for result in chunk_results:
                _record_chunk_result(
                    result,
                    final_values=final_values,
                    all_routes=all_routes,
                    warnings=warnings,
                    manifest=manifest,
                )
            atomic_write_json(manifest_path, manifest)
            logger.info("completed %d chunks with concurrency %d", len(chunks), chunk_concurrency)
        final = assemble_chunks(final_values)
        atomic_write_text(run_directory / "final.txt", final)
        metrics = calculate_metrics(source, final)
        if semantic_metrics is not None:
            metrics["semantic"] = await asyncio.to_thread(
                semantic_metrics.calculate,
                source,
                final,
                run_directory,
            )
        atomic_write_json(run_directory / "metrics.json", metrics)
        manifest["metrics"] = metrics
        warnings.extend(
            str(item)
            for item in request_controller.summary().get("enforceability_warnings", [])
        )
        report_path = create_report(
            run_directory,
            config=config.to_dict(),
            metrics=metrics,
            routes=all_routes,
            warnings=sorted(set(warnings)),
        )
        report_generation = await _append_generated_report_summary(
            report_path,
            config=config,
            metrics=metrics,
            routes=all_routes,
            request_controller=request_controller,
        )
        manifest["translation_models"] = _translation_models_manifest(run_directory)
        if report_generation is not None:
            manifest["report_generation"] = {
                "provider": report_generation.provider,
                "model": report_generation.model,
                "response_id": report_generation.response_id,
                "usage": report_generation.usage,
            }
        manifest["request_summary"] = request_controller.summary()
        manifest["budget_status"] = request_controller.budget_status()
        manifest["state"] = "completed"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        atomic_write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["state"] = (
            "interrupted"
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            else ("budget_exceeded" if isinstance(error, BudgetExceededError) else "failed")
        )
        manifest["last_error"] = safe_diagnostic(error)
        manifest["translation_models"] = _translation_models_manifest(run_directory)
        manifest["request_summary"] = request_controller.summary()
        manifest["budget_status"] = request_controller.budget_status()
        atomic_write_json(manifest_path, manifest)
        raise
    return run_directory


def _effective_chunk_concurrency(config: RunConfig) -> int:
    if config.context.enabled or config.memory.enabled:
        return 1
    return config.runtime.concurrency


async def _run_chunks_concurrently(
    *,
    chunks: list[Chunk],
    concurrency: int,
    config: RunConfig,
    run_directory: Path,
    prompts: dict[StageType, str],
    prompt_checksums: dict[str, str],
    translator: Any,
    generators: dict[str, Any],
    memory: MemoryStore,
) -> list[_ChunkResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def execute(chunk: Chunk) -> _ChunkResult:
        async with semaphore:
            result = await _run_chunk(
                chunk=chunk,
                config=config,
                run_directory=run_directory,
                previous_contexts=[],
                prompts=prompts,
                prompt_checksums=prompt_checksums,
                translator=translator,
                generators=generators,
                memory=memory,
            )
            logger.info("finished chunk %d/%d", chunk.index + 1, len(chunks))
            return result

    tasks = [
        asyncio.create_task(execute(chunk), name=f"chunk-{chunk.index + 1}")
        for chunk in chunks
    ]
    try:
        # gather returns in input order even when chunks finish out of order.
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_chunk(
    *,
    chunk: Chunk,
    config: RunConfig,
    run_directory: Path,
    previous_contexts: list[str],
    prompts: dict[StageType, str],
    prompt_checksums: dict[str, str],
    translator: Any,
    generators: dict[str, Any],
    memory: MemoryStore,
) -> _ChunkResult:
    chunk_directory = run_directory / "chunks" / f"{chunk.index + 1:04d}"
    atomic_write_text(chunk_directory / "source.txt", chunk.source_text)
    atomic_write_text(chunk_directory / "context.txt", chunk.context_prefix)
    store = CheckpointStore(chunk_directory)
    current = chunk.source_text
    stage_number = 0
    translated = False
    stage_outputs: list[str] = []
    routes: list[list[str]] = []
    warnings: list[str] = []
    for stage in config.pipeline:
        for repetition in range(stage.repeat):
            stage_number += 1
            current, route, stage_warnings, execution_succeeded = await _run_stage(
                config=config,
                stage=stage,
                repetition=repetition,
                stage_number=stage_number,
                chunk_index=chunk.index,
                input_text=current,
                previous_contexts=previous_contexts,
                input_is_damaged=translated,
                prompts=prompts,
                prompt_checksums=prompt_checksums,
                translator=translator,
                generator=generators.get(stage.type.value),
                memory=memory,
                store=store,
                events_path=run_directory / "events.jsonl",
            )
            if route:
                routes.append(route)
            warnings.extend(stage_warnings)
            stage_outputs.append(current)
            if stage.type is StageType.TRANSLATION_CYCLE and execution_succeeded:
                translated = True
    atomic_write_text(chunk_directory / "final.txt", current)
    return _ChunkResult(
        chunk_id=chunk.chunk_id,
        index=chunk.index,
        final_text=current,
        context_value=_chunk_context_value(current, stage_outputs, config),
        routes=routes,
        warnings=warnings,
    )


def _record_chunk_result(
    result: _ChunkResult,
    *,
    final_values: list[str],
    all_routes: list[list[str]],
    warnings: list[str],
    manifest: dict[str, Any],
) -> None:
    final_values.append(result.final_text)
    all_routes.extend(result.routes)
    warnings.extend(result.warnings)
    processed = manifest.setdefault("processed_chunks", [])
    if result.chunk_id not in processed:
        processed.append(result.chunk_id)


async def _run_stage(
    *,
    config: RunConfig,
    stage: PipelineStageConfig,
    repetition: int,
    stage_number: int,
    chunk_index: int,
    input_text: str,
    previous_contexts: list[str],
    input_is_damaged: bool,
    prompts: dict[StageType, str],
    prompt_checksums: dict[str, str],
    translator: Any,
    generator: Any,
    memory: MemoryStore,
    store: CheckpointStore,
    events_path: Path,
) -> tuple[str, list[str], list[str], bool]:
    seed = (
        stage.seed
        if stage.seed is not None
        else (
            config.seed + chunk_index * 10_000 + stage_number * 101 + repetition + stage.seed_offset
        )
    )
    rng = random.Random(seed)
    applied = stage.enabled and rng.random() < stage.probability
    context = (
        rolling_context(previous_contexts, config.context)
        if stage.type is StageType.CONTEXTUAL_RECONSTRUCTION
        else ""
    )
    observations = (
        memory_prompt(
            memory,
            chunk_index,
            config.memory.maximum_items_in_prompt,
            config.memory.minimum_count,
        )
        if config.memory.enabled
        and stage.type in {StageType.RECONSTRUCTION, StageType.CONTEXTUAL_RECONSTRUCTION}
        else ""
    )
    route: list[str] = []
    if applied and stage.type is StageType.TRANSLATION_CYCLE:
        route_config = replace(
            config.route,
            min_hops=int(stage.parameters.get("hops", {}).get("min", config.route.min_hops)),
            max_hops=int(stage.parameters.get("hops", {}).get("max", config.route.max_hops)),
        )
        route = generate_route(
            route_config,
            source_language=config.source_language,
            target_language=config.target_language,
            seed=seed,
        )
    if applied and stage.type is StageType.MEMORY_EXTRACTION and not input_is_damaged:
        raise ValueError(
            "memory_extraction refused original input; a successful translation_cycle "
            "must run first"
        )
    stage_checksum = checksum_data(
        {
            "input": checksum_text(input_text),
            "stage": asdict(stage),
            "seed": seed,
            "applied": applied,
            "route": route,
            "context": checksum_text(context),
            "memory": checksum_text(observations),
            "translation": asdict(config.translation),
            "generation": asdict(config.generation),
            "prompt_checksum": prompt_checksums.get(stage.type.value),
        }
    )
    cached = store.load(stage_number, stage.type.value, stage_checksum)
    if cached:
        if stage.type is StageType.MEMORY_EXTRACTION:
            if not cached.memory_payload or not cached.memory_event_id:
                raise RuntimeError(
                    "memory checkpoint lacks schema-v2 side-effect data; start a new run"
                )
            memory.ingest_json(
                cached.memory_payload,
                chunk_index,
                provenance="damaged",
                event_id=cached.memory_event_id,
            )
        append_jsonl(
            events_path,
            {
                "event": "checkpoint_reused",
                "chunk": chunk_index,
                "stage": stage_number,
                "stage_type": stage.type.value,
                "stage_checksum": stage_checksum,
            },
        )
        logger.info("chunk %d stage %d %s (checkpoint)", chunk_index + 1, stage_number, stage.type)
        return (
            cached.output_text,
            cached.route,
            cached.warnings,
            cached.execution_succeeded,
        )

    started = perf_counter()
    attempts = 1
    provider_name = "none"
    model = "none"
    usage: dict[str, int | float] | None = None
    response_id: str | None = None
    stage_warnings: list[str] = []
    provider_route: list[str] = []
    translation_details: list[dict[str, Any]] = []
    memory_payload: str | None = None
    memory_event_id: str | None = None
    output = input_text
    prompt_checksum = prompt_checksums.get(stage.type.value)
    error_message: str | None = None
    fatal_error: Exception | None = None
    execution_succeeded = False
    logger.info("chunk %d stage %d %s", chunk_index + 1, stage_number, stage.type)
    if route:
        logger.info("route: %s", " -> ".join(route))
    try:
        if not applied:
            stage_warnings.append("stage skipped by enabled/probability decision")
        elif stage.type is StageType.TRANSLATION_CYCLE:

            hop_results: list[TranslationResult] = []

            def record_hop(
                hop_index: int,
                hop_input: str,
                hop_result: TranslationResult,
            ) -> None:
                nonlocal output, provider_name, model, response_id, usage
                output_path, output_checksum = store.save_hop_output(
                    stage_number,
                    stage.type.value,
                    hop_index,
                    hop_result.text,
                )
                translation_details.append(
                    {
                        "provider": hop_result.provider,
                        "model": hop_result.model,
                        "response_id": hop_result.response_id,
                        "usage": hop_result.usage,
                        "warnings": hop_result.warnings,
                        "deterministic": hop_result.deterministic,
                        "input_checksum": checksum_text(hop_input),
                        "output_checksum": output_checksum,
                        "output_path": output_path,
                        "seed": seed + hop_index,
                        **hop_result.metadata,
                    }
                )
                hop_results.append(hop_result)
                provider_route.append(hop_result.provider)
                output = hop_result.text
                provider_name = hop_result.provider
                model = hop_result.model
                response_id = hop_result.response_id
                usage = _sum_usage(result.usage for result in hop_results)

            async def operation() -> tuple[str, list[TranslationResult]]:
                nonlocal output, provider_name, model, response_id, usage
                translation_details.clear()
                provider_route.clear()
                hop_results.clear()
                output = input_text
                provider_name = "none"
                model = "none"
                response_id = None
                usage = None
                return await translate_route(
                    translator,
                    input_text,
                    route,
                    seed=seed,
                    on_hop=record_hop,
                )

            (output, hop_results), attempts = await with_retry(
                operation,
                retries=config.runtime.retries,
                backoff_seconds=config.runtime.retry_backoff_seconds,
            )
            if hop_results:
                stage_warnings.extend(
                    warning for result in hop_results for warning in result.warnings
                )
            execution_succeeded = True
        elif stage.type is StageType.FINAL_TRANSLATION:
            # Translation cycles already return to target; translate only when explicitly configured.
            source = str(stage.parameters.get("source_language", config.target_language))
            if source == config.target_language:
                stage_warnings.append("final text already declared in target language")
                execution_succeeded = True
            else:
                result, attempts = await with_retry(
                    lambda: translator.translate(
                        input_text, source, config.target_language, seed=seed
                    ),
                    retries=config.runtime.retries,
                    backoff_seconds=config.runtime.retry_backoff_seconds,
                )
                output = result.text
                provider_name, model = result.provider, result.model
                provider_route = [result.provider]
                output_path, output_checksum = store.save_hop_output(
                    stage_number,
                    stage.type.value,
                    0,
                    result.text,
                )
                translation_details = [
                    {
                        "provider": result.provider,
                        "model": result.model,
                        "response_id": result.response_id,
                        "usage": result.usage,
                        "warnings": result.warnings,
                        "deterministic": result.deterministic,
                        "input_checksum": checksum_text(input_text),
                        "output_checksum": output_checksum,
                        "output_path": output_path,
                        "source_language": source,
                        "target_language": config.target_language,
                        "seed": seed,
                        **result.metadata,
                    }
                ]
                response_id, usage = result.response_id, result.usage
                stage_warnings.extend(result.warnings)
                execution_succeeded = True
        elif stage.type is StageType.CONSERVATIVE_REPAIR:
            result, attempts = await with_retry(
                lambda: repair_text(
                    generator,
                    prompts[stage.type],
                    input_text,
                    temperature=_temperature(config, stage.type.value, 0.3),
                    seed=seed,
                    max_length_ratio=_optional_float(
                        stage.parameters.get("max_length_ratio")
                    ),
                    allow_new_events=_optional_bool(
                        stage.parameters.get("allow_new_events")
                    ),
                ),
                retries=config.runtime.retries,
                backoff_seconds=config.runtime.retry_backoff_seconds,
            )
            output, provider_name, model, response_id, usage = _generation_fields(result)
            stage_warnings.extend(result.warnings)
            execution_succeeded = True
        elif stage.type in {StageType.RECONSTRUCTION, StageType.CONTEXTUAL_RECONSTRUCTION}:
            result, attempts = await with_retry(
                lambda: reconstruct_text(
                    generator,
                    prompts[stage.type],
                    input_text,
                    temperature=_temperature(config, stage.type.value, 0.7),
                    seed=seed,
                    context=context,
                    memory=observations,
                    max_length_ratio=_optional_float(
                        stage.parameters.get("max_length_ratio")
                    ),
                    max_new_sentences_per_chunk=_optional_int(
                        stage.parameters.get("max_new_sentences_per_chunk")
                    ),
                    repetition_policy=_optional_string(
                        stage.parameters.get("repetition_policy")
                    ),
                    allow_new_events=_optional_bool(
                        stage.parameters.get("allow_new_events")
                    ),
                    allow_scene_expansion=_optional_bool(
                        stage.parameters.get("allow_scene_expansion")
                    ),
                ),
                retries=config.runtime.retries,
                backoff_seconds=config.runtime.retry_backoff_seconds,
            )
            output, provider_name, model, response_id, usage = _generation_fields(result)
            stage_warnings.extend(result.warnings)
            execution_succeeded = True
        elif stage.type is StageType.MEMORY_EXTRACTION:
            result, attempts = await with_retry(
                lambda: _valid_memory_generation(
                    generator,
                    prompts[stage.type],
                    input_text,
                    temperature=_temperature(config, stage.type.value, 0.1),
                    seed=seed,
                ),
                retries=config.runtime.retries,
                backoff_seconds=config.runtime.retry_backoff_seconds,
            )
            memory_payload = result.text
            memory_event_id = "memory-" + checksum_data(
                {
                    "run": store.directory.parents[1].name,
                    "chunk": chunk_index,
                    "stage": stage_number,
                    "stage_checksum": stage_checksum,
                }
            )[:24]
            provider_name, model = result.provider, result.model
            response_id, usage = result.response_id, result.usage
            stage_warnings.extend(result.warnings)
            execution_succeeded = True
        else:
            raise ValueError(f"unsupported stage type: {stage.type}")
    except Exception as error:  # noqa: BLE001 - provider boundary records vendor exceptions
        attempts = int(getattr(error, "attempts", attempts))
        error_message = safe_diagnostic(error)
        if isinstance(error, BudgetExceededError):
            stage_warnings.append(error_message)
            output = input_text
            fatal_error = error
        elif config.runtime.failure_policy == "skip":
            stage_warnings.append(error_message)
            output = input_text
        elif config.runtime.failure_policy == "fallback":
            stage_warnings.append(
                f"primary provider failed; used {config.runtime.fallback_provider}: {error_message}"
            )
            fallback = await _run_mock_fallback(
                stage=stage,
                input_text=input_text,
                route=route,
                prompts=prompts,
                config=config,
                seed=seed,
                context=context,
                observations=observations,
            )
            output, provider_name, model, response_id, usage = _generation_fields(fallback)
            if stage.type is StageType.MEMORY_EXTRACTION:
                memory_payload = fallback.text
                memory_event_id = "memory-" + checksum_data(
                    {
                        "run": store.directory.parents[1].name,
                        "chunk": chunk_index,
                        "stage": stage_number,
                        "stage_checksum": stage_checksum,
                    }
                )[:24]
            attempts += 1
            execution_succeeded = True
        else:
            stage_warnings.append(error_message)
            output = input_text
            fatal_error = error
    max_output_ratio = float(
        stage.parameters.get(
            "max_length_ratio",
            stage.parameters.get("max_output_ratio", 3.0),
        )
    )
    if len(output) > max(1, len(input_text)) * max_output_ratio:
        stage_warnings.append(
            f"output length exceeds configured ratio {max_output_ratio:.2f}; preserved for inspection"
        )
    max_new_sentences = stage.parameters.get("max_new_sentences_per_chunk")
    if max_new_sentences is not None:
        sentence_growth = _sentence_count(output) - _sentence_count(input_text)
        if sentence_growth > int(max_new_sentences):
            stage_warnings.append(
                "output adds "
                f"{sentence_growth} sentences; configured maximum is {int(max_new_sentences)}"
            )
    result_record = StageResult(
        input_text=input_text,
        output_text=output,
        stage_type=stage.type.value,
        provider=provider_name,
        model=model,
        source_language=config.source_language if route else None,
        target_language=config.target_language if route else None,
        route=route,
        duration_seconds=perf_counter() - started,
        attempts=attempts,
        warnings=stage_warnings,
        error=error_message,
        usage=usage,
        input_checksum=checksum_text(input_text),
        output_checksum=checksum_text(output),
        stage_checksum=stage_checksum,
        applied=applied,
        response_id=response_id,
        prompt_checksum=prompt_checksum,
        execution_succeeded=execution_succeeded,
        checkpoint_reusable=fatal_error is None,
        provider_route=provider_route,
        translation_details=translation_details,
        memory_payload=memory_payload,
        memory_event_id=memory_event_id,
    )
    metadata_path, _ = store.save(stage_number, stage.type.value, result_record)
    if (
        stage.type is StageType.MEMORY_EXTRACTION
        and execution_succeeded
        and memory_payload
        and memory_event_id
    ):
        memory.ingest_json(
            memory_payload,
            chunk_index,
            provenance="damaged",
            event_id=memory_event_id,
        )
    append_jsonl(
        events_path,
        {
            "event": "stage_completed",
            "timestamp": datetime.now(UTC).isoformat(),
            "chunk": chunk_index,
            "stage": stage_number,
            "stage_type": stage.type.value,
            "applied": applied,
            "seed": seed,
            "route": route,
            "provider_route": provider_route,
            "translations": translation_details,
            "attempts": attempts,
            "checkpoint": str(metadata_path),
            "warnings": stage_warnings,
            "error": error_message,
        },
    )
    if fatal_error is not None:
        raise fatal_error
    return output, route, stage_warnings, execution_succeeded


def _chunk_context_value(
    final_text: str,
    stage_outputs: list[str],
    config: RunConfig,
) -> str:
    if config.context.source == "final_generated_text" and not config.context.include_intermediate:
        return final_text
    values = stage_outputs[:-1]
    if config.context.source == "all_generated_text" or config.context.include_intermediate:
        values = [*stage_outputs, final_text]
    unique: list[str] = []
    for value in values:
        if value and (not unique or unique[-1] != value):
            unique.append(value)
    return "\n\n".join(unique)


async def _run_mock_fallback(
    *,
    stage: PipelineStageConfig,
    input_text: str,
    route: list[str],
    prompts: dict[StageType, str],
    config: RunConfig,
    seed: int,
    context: str,
    observations: str,
) -> GenerationResult:
    mock_generator = MockGenerationProvider()
    if stage.type is StageType.TRANSLATION_CYCLE:
        text, _ = await translate_route(MockTranslationProvider(), input_text, route, seed=seed)
        return GenerationResult(
            text=text,
            provider="mock-fallback",
            model="mock-translation",
            deterministic=True,
        )
    if stage.type is StageType.CONSERVATIVE_REPAIR:
        return await repair_text(
            mock_generator,
            prompts[stage.type],
            input_text,
            temperature=_temperature(config, stage.type.value, 0.3),
            seed=seed,
            max_length_ratio=_optional_float(stage.parameters.get("max_length_ratio")),
            allow_new_events=_optional_bool(stage.parameters.get("allow_new_events")),
        )
    if stage.type in {StageType.RECONSTRUCTION, StageType.CONTEXTUAL_RECONSTRUCTION}:
        return await reconstruct_text(
            mock_generator,
            prompts[stage.type],
            input_text,
            temperature=_temperature(config, stage.type.value, 0.7),
            seed=seed,
            context=context,
            memory=observations,
            max_length_ratio=_optional_float(stage.parameters.get("max_length_ratio")),
            max_new_sentences_per_chunk=_optional_int(
                stage.parameters.get("max_new_sentences_per_chunk")
            ),
            repetition_policy=_optional_string(stage.parameters.get("repetition_policy")),
            allow_new_events=_optional_bool(stage.parameters.get("allow_new_events")),
            allow_scene_expansion=_optional_bool(
                stage.parameters.get("allow_scene_expansion")
            ),
        )
    if stage.type is StageType.MEMORY_EXTRACTION:
        return await _valid_memory_generation(
            mock_generator,
            prompts[stage.type],
            input_text,
            temperature=_temperature(config, stage.type.value, 0.1),
            seed=seed,
        )
    return GenerationResult(
        text=input_text,
        provider="mock-fallback",
        model="no-op",
        deterministic=True,
    )


async def _valid_memory_generation(
    generator: Any,
    instruction: str,
    input_text: str,
    *,
    temperature: float,
    seed: int,
) -> GenerationResult:
    result = await extract_memory(
        generator, instruction, input_text, temperature=temperature, seed=seed
    )
    validate_memory_payload(result.text)
    return result


def _generation_fields(
    result: GenerationResult,
) -> tuple[str, str, str, str | None, dict[str, int | float] | None]:
    return result.text, result.provider, result.model, result.response_id, result.usage


def _sum_usage(values: Any) -> dict[str, int | float] | None:
    total: dict[str, int | float] = {}
    for value in values:
        if not value:
            continue
        for key, amount in value.items():
            if isinstance(amount, (int, float)):
                total[key] = total.get(key, 0) + amount
    return total or None


def _temperature(config: RunConfig, task: str, default: float) -> float:
    task_config = config.generation.tasks.get(task)
    if task_config is not None and "temperature" in task_config.options:
        return float(task_config.options["temperature"])
    aliases = {
        StageType.CONSERVATIVE_REPAIR.value: "repair",
        StageType.CONTEXTUAL_RECONSTRUCTION.value: "reconstruction",
    }
    value = config.temperatures.get(task)
    if value is None and task in aliases:
        value = config.temperatures.get(aliases[task])
    return float(default if value is None else value)


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?…]+(?:\s+|$)", text) if part.strip()])


async def _append_generated_report_summary(
    report_path: Path,
    *,
    config: RunConfig,
    metrics: dict[str, Any],
    routes: list[list[str]],
    request_controller: RequestController,
) -> GenerationResult | None:
    generator = generation_provider(
        config.generation,
        task="report_generation",
        request_controller=request_controller,
    )
    if isinstance(generator, MockGenerationProvider):
        return None
    custom = config.custom_prompts.get("report_generation")
    if custom and custom.startswith("builtin:"):
        instruction = prompt_text(custom.split(":", 1)[1])
    elif custom:
        path = Path(custom)
        if not path.is_absolute() and config.config_path:
            path = Path(config.config_path).parent / path
        instruction = path.read_text(encoding="utf-8")
    else:
        instruction = prompt_text("report_generation")
    prompt = f"{instruction}\n\n<<<TEXT>>>\n" + json.dumps(
        {
            "run": config.name,
            "seed": config.seed,
            "metrics": metrics,
            "routes": routes[:50],
        },
        ensure_ascii=False,
        indent=2,
    )
    result, _ = await with_retry(
        lambda: generator.generate(
            prompt,
            temperature=_temperature(config, "report_generation", 0.2),
            seed=config.seed + 999_983,
        ),
        retries=config.runtime.retries,
        backoff_seconds=config.runtime.retry_backoff_seconds,
    )
    existing = report_path.read_text(encoding="utf-8").rstrip()
    atomic_write_text(
        report_path,
        f"{existing}\n\n## LLM-generated diagnostic summary\n\n{result.text.strip()}\n",
    )
    return result


def _translation_models_manifest(run_directory: Path) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for stage_path in sorted((run_directory / "chunks").glob("*/stage-*.json")):
        stage = read_json(stage_path)
        for detail in stage.get("translation_details", []):
            if not isinstance(detail, dict):
                continue
            item = {
                key: detail.get(key)
                for key in (
                    "engine",
                    "provider",
                    "provider_type",
                    "category",
                    "model",
                    "revision",
                    "source_language",
                    "target_language",
                    "legs",
                    "server_version",
                )
                if detail.get(key) is not None
            }
            encoded = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if item and encoded not in seen:
                seen.add(encoded)
                models.append(item)
    return models
