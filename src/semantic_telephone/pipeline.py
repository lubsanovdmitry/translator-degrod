from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import random
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from . import __version__
from .checkpoints import CheckpointStore
from .chunking import assemble_chunks, chunk_text
from .config import resolve_input_path, resolve_output_root
from .memory import InvalidMemoryResponse, MemoryStore
from .metrics import calculate_metrics
from .models import (
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
from .routes import generate_route
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


def _prompt_root() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts"


def load_prompts(config: RunConfig) -> tuple[dict[StageType, str], dict[str, str]]:
    prompts: dict[StageType, str] = {}
    checksums: dict[str, str] = {}
    for stage_type, filename in PROMPT_FILES.items():
        custom = config.custom_prompts.get(stage_type.value)
        if custom:
            path = Path(custom)
            if not path.is_absolute() and config.config_path:
                path = Path(config.config_path).parent / path
        else:
            path = _prompt_root() / filename
        text = path.read_text(encoding="utf-8")
        prompts[stage_type] = text
        checksums[stage_type.value] = checksum_text(text)
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

    prompts, prompt_checksums = load_prompts(config)
    atomic_write_text(run_directory / "source.txt", source)
    atomic_write_yaml(run_directory / "resolved_config.yaml", config.to_dict())
    manifest_path = run_directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
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

    translator = translation_provider(config.translation)
    generator = generation_provider(config.generation)
    chunks = chunk_text(source, config.chunking)
    memory = MemoryStore(
        run_directory / "memory", half_life=config.memory.half_life_chunks
    )
    previous_finals: list[str] = []
    final_values: list[str] = []
    all_routes: list[list[str]] = []
    warnings: list[str] = []
    try:
        for chunk in chunks:
            chunk_directory = run_directory / "chunks" / f"{chunk.index + 1:04d}"
            atomic_write_text(chunk_directory / "source.txt", chunk.source_text)
            atomic_write_text(chunk_directory / "context.txt", chunk.context_prefix)
            store = CheckpointStore(chunk_directory)
            current = chunk.source_text
            stage_number = 0
            for stage in config.pipeline:
                for repetition in range(stage.repeat):
                    stage_number += 1
                    current, route, stage_warnings = await _run_stage(
                        config=config,
                        stage=stage,
                        repetition=repetition,
                        stage_number=stage_number,
                        chunk_index=chunk.index,
                        input_text=current,
                        previous_finals=previous_finals,
                        prompts=prompts,
                        prompt_checksums=prompt_checksums,
                        translator=translator,
                        generator=generator,
                        memory=memory,
                        store=store,
                        events_path=run_directory / "events.jsonl",
                    )
                    if route:
                        all_routes.append(route)
                    warnings.extend(stage_warnings)
            atomic_write_text(chunk_directory / "final.txt", current)
            final_values.append(current)
            previous_finals.append(current)
            processed = manifest.setdefault("processed_chunks", [])
            if chunk.chunk_id not in processed:
                processed.append(chunk.chunk_id)
            atomic_write_json(manifest_path, manifest)
            logger.info("completed chunk %d/%d", chunk.index + 1, len(chunks))
        final = assemble_chunks(final_values)
        atomic_write_text(run_directory / "final.txt", final)
        metrics = calculate_metrics(source, final)
        atomic_write_json(run_directory / "metrics.json", metrics)
        manifest["metrics"] = metrics
        manifest["state"] = "completed"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        atomic_write_json(manifest_path, manifest)
        create_report(
            run_directory,
            config=config.to_dict(),
            metrics=metrics,
            routes=all_routes,
            warnings=sorted(set(warnings)),
        )
    except BaseException as error:
        manifest["state"] = "interrupted" if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
        manifest["last_error"] = f"{type(error).__name__}: {error}"
        atomic_write_json(manifest_path, manifest)
        raise
    return run_directory


async def _run_stage(
    *,
    config: RunConfig,
    stage: PipelineStageConfig,
    repetition: int,
    stage_number: int,
    chunk_index: int,
    input_text: str,
    previous_finals: list[str],
    prompts: dict[StageType, str],
    prompt_checksums: dict[str, str],
    translator: Any,
    generator: Any,
    memory: MemoryStore,
    store: CheckpointStore,
    events_path: Path,
) -> tuple[str, list[str], list[str]]:
    seed = stage.seed if stage.seed is not None else (
        config.seed + chunk_index * 10_000 + stage_number * 101 + repetition + stage.seed_offset
    )
    rng = random.Random(seed)
    applied = stage.enabled and rng.random() < stage.probability
    context = (
        rolling_context(previous_finals, config.context)
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
        return cached.output_text, cached.route, cached.warnings

    started = perf_counter()
    attempts = 1
    provider_name = "none"
    model = "none"
    usage: dict[str, int | float] | None = None
    response_id: str | None = None
    stage_warnings: list[str] = []
    output = input_text
    prompt_checksum = prompt_checksums.get(stage.type.value)
    error_message: str | None = None
    fatal_error: Exception | None = None
    logger.info("chunk %d stage %d %s", chunk_index + 1, stage_number, stage.type)
    if route:
        logger.info("route: %s", " -> ".join(route))
    try:
        if not applied:
            stage_warnings.append("stage skipped by enabled/probability decision")
        elif stage.type is StageType.TRANSLATION_CYCLE:
            async def operation() -> tuple[str, list[TranslationResult]]:
                return await translate_route(translator, input_text, route, seed=seed)

            (output, hop_results), attempts = await with_retry(
                operation,
                retries=config.runtime.retries,
                backoff_seconds=config.runtime.retry_backoff_seconds,
            )
            if hop_results:
                provider_name = hop_results[-1].provider
                model = hop_results[-1].model
                response_id = hop_results[-1].response_id
                stage_warnings.extend(
                    warning for result in hop_results for warning in result.warnings
                )
                usage = _sum_usage(result.usage for result in hop_results)
        elif stage.type is StageType.FINAL_TRANSLATION:
            # Translation cycles already return to target; translate only when explicitly configured.
            source = str(stage.parameters.get("source_language", config.target_language))
            if source == config.target_language:
                stage_warnings.append("final text already declared in target language")
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
                response_id, usage = result.response_id, result.usage
                stage_warnings.extend(result.warnings)
        elif stage.type is StageType.CONSERVATIVE_REPAIR:
            result, attempts = await with_retry(
                lambda: repair_text(
                    generator,
                    prompts[stage.type],
                    input_text,
                    temperature=config.temperatures["repair"],
                    seed=seed,
                ),
                retries=config.runtime.retries,
                backoff_seconds=config.runtime.retry_backoff_seconds,
            )
            output, provider_name, model, response_id, usage = _generation_fields(result)
            stage_warnings.extend(result.warnings)
        elif stage.type in {StageType.RECONSTRUCTION, StageType.CONTEXTUAL_RECONSTRUCTION}:
            result, attempts = await with_retry(
                lambda: reconstruct_text(
                    generator,
                    prompts[stage.type],
                    input_text,
                    temperature=config.temperatures["reconstruction"],
                    seed=seed,
                    context=context,
                    memory=observations,
                ),
                retries=config.runtime.retries,
                backoff_seconds=config.runtime.retry_backoff_seconds,
            )
            output, provider_name, model, response_id, usage = _generation_fields(result)
            stage_warnings.extend(result.warnings)
        elif stage.type is StageType.MEMORY_EXTRACTION:
            result, attempts = await with_retry(
                lambda: _valid_memory_generation(
                    generator,
                    prompts[stage.type],
                    input_text,
                    temperature=config.temperatures["memory_extraction"],
                    seed=seed,
                ),
                retries=config.runtime.retries,
                backoff_seconds=config.runtime.retry_backoff_seconds,
            )
            memory.ingest_json(result.text, chunk_index, provenance="damaged")
            provider_name, model = result.provider, result.model
            response_id, usage = result.response_id, result.usage
            stage_warnings.extend(result.warnings)
        else:
            raise ValueError(f"unsupported stage type: {stage.type}")
    except Exception as error:  # noqa: BLE001 - provider boundary records vendor exceptions
        error_message = f"{type(error).__name__}: {error}"
        if config.runtime.failure_policy == "skip":
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
                memory.ingest_json(fallback.text, chunk_index, provenance="damaged")
            attempts += 1
        else:
            stage_warnings.append(error_message)
            output = input_text
            fatal_error = error
    max_output_ratio = float(stage.parameters.get("max_output_ratio", 3.0))
    if len(output) > max(1, len(input_text)) * max_output_ratio:
        stage_warnings.append(
            f"output length exceeds configured ratio {max_output_ratio:.2f}; preserved for inspection"
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
    )
    metadata_path, _ = store.save(stage_number, stage.type.value, result_record)
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
            "attempts": attempts,
            "checkpoint": str(metadata_path),
            "warnings": stage_warnings,
            "error": error_message,
        },
    )
    if fatal_error is not None:
        raise fatal_error
    return output, route, stage_warnings


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
            temperature=config.temperatures["repair"],
            seed=seed,
        )
    if stage.type in {StageType.RECONSTRUCTION, StageType.CONTEXTUAL_RECONSTRUCTION}:
        return await reconstruct_text(
            mock_generator,
            prompts[stage.type],
            input_text,
            temperature=config.temperatures["reconstruction"],
            seed=seed,
            context=context,
            memory=observations,
        )
    if stage.type is StageType.MEMORY_EXTRACTION:
        return await _valid_memory_generation(
            mock_generator,
            prompts[stage.type],
            input_text,
            temperature=config.temperatures["memory_extraction"],
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
    try:
        value = json.loads(result.text)
    except json.JSONDecodeError as error:
        raise InvalidMemoryResponse(f"invalid JSON: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("observations"), list):
        raise InvalidMemoryResponse("response must contain an observations list")
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
