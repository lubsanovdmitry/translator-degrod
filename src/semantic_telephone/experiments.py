from __future__ import annotations

import asyncio
import csv
import statistics
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError, load_config
from .models import RunConfig
from .pipeline import run_pipeline
from .utils.files import atomic_write_json, atomic_write_text, read_json


async def run_matrix(path: Path) -> Path:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read matrix configuration: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("matrix configuration must be a mapping")
    experiment = raw.get("experiment", {})
    variants = raw.get("variants")
    if not isinstance(experiment, dict) or not isinstance(variants, list) or not variants:
        raise ConfigError("matrix requires experiment and a non-empty variants list")
    name = str(experiment.get("name", "experiment"))
    seeds_raw = experiment.get("seeds", [])
    repetitions = int(experiment.get("repetitions", 1))
    seeds = [int(seed) for seed in seeds_raw] if seeds_raw else list(range(repetitions))
    concurrency = experiment.get("concurrency", 1)
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ConfigError("'experiment.concurrency' must be a positive integer")
    failure_policy = str(experiment.get("failure_policy", "stop"))
    if failure_policy not in {"stop", "continue"}:
        raise ConfigError("'experiment.failure_policy' must be stop or continue")
    base = path.parent / "runs" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-matrix-{name}"
    )
    root = base
    suffix = 1
    while root.exists():
        root = Path(f"{base}-{suffix}")
        suffix += 1
    root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, int, int, RunConfig, Path]] = []
    variant_names: list[str] = []
    for variant in variants:
        if not isinstance(variant, dict) or "config" not in variant:
            raise ConfigError("each matrix variant requires config")
        config_path = Path(str(variant["config"]))
        if not config_path.is_absolute():
            config_path = path.parent / config_path
        base_config = load_config(config_path)
        variant_name = str(variant.get("name", base_config.name))
        if variant_name in variant_names:
            raise ConfigError(f"matrix variant names must be unique: {variant_name}")
        variant_names.append(variant_name)
        for repetition, seed in enumerate(seeds):
            config = replace(
                base_config,
                name=f"{variant_name}-{repetition + 1}",
                seed=seed,
                output_root=str(root / "runs"),
            )
            job_number = len(jobs) + 1
            run_directory = (
                root
                / "runs"
                / f"{job_number:04d}-{_slug(variant_name)}-{repetition + 1}"
            )
            jobs.append((variant_name, repetition + 1, seed, config, run_directory))
    baseline = str(experiment.get("baseline", variant_names[0] if variant_names else ""))
    if baseline not in variant_names:
        raise ConfigError("'experiment.baseline' must name a configured variant")
    rows = await _run_matrix_jobs(
        jobs,
        concurrency=concurrency,
        failure_policy=failure_policy,
    )
    aggregates = _aggregate_rows(rows)
    paired = _paired_deltas(rows, baseline=baseline)
    atomic_write_json(
        root / "summary.json",
        {
            "experiment": experiment,
            "baseline": baseline,
            "runs": rows,
            "aggregates": aggregates,
            "paired_deltas": paired,
        },
    )
    _write_csv(root / "summary.csv", rows)
    _write_csv(root / "aggregates.csv", aggregates)
    report_lines = [
        f"# Experiment matrix: {name}",
        "",
        "> These comparisons are diagnostic and do not establish artistic quality or a",
        "> scientifically proven effect.",
        "",
        f"- Baseline: `{baseline}`",
        f"- Failure policy: `{failure_policy}`",
        "",
        "| Variant | Rep | Seed | Status | Length ratio | Token overlap | HTTP | Final |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "completed":
            relative = Path(str(row["final"])).relative_to(root)
            report_lines.append(
                f"| {row['variant']} | {row['repetition']} | {row['seed']} | completed | "
                f"{float(row['length_ratio']):.3f} | {float(row['token_overlap']):.3f} | "
                f"{row['http_attempts']} | [text]({relative.as_posix()}) |"
            )
        else:
            report_lines.append(
                f"| {row['variant']} | {row['repetition']} | {row['seed']} | failed | "
                f"— | — | — | {row.get('error', 'unknown error')} |"
            )
    report_lines.extend(["", "## Aggregate statistics", ""])
    for aggregate in aggregates:
        report_lines.append(
            f"- {aggregate['variant']} / {aggregate['metric']}: "
            f"mean={float(aggregate['mean']):.4f}, "
            f"stdev={float(aggregate['stdev']):.4f}, "
            f"median={float(aggregate['median']):.4f}, "
            f"min={float(aggregate['minimum']):.4f}, "
            f"max={float(aggregate['maximum']):.4f}, "
            f"n={aggregate['n']}, failures={aggregate['failures']}"
        )
    report_lines.extend(["", "## Paired deltas from baseline", ""])
    for delta in paired:
        mean = delta.get("mean_delta")
        mean_text = f"{float(mean):.4f}" if isinstance(mean, (int, float)) else "unavailable"
        report_lines.append(
            f"- {delta['variant']} / {delta['metric']}: "
            f"mean delta={mean_text}, n={delta['n']}, "
            f"missing pairs={delta['missing_pairs']}, failures={delta['failures']}"
        )
    report_lines.extend(["", "## Provider and model breakdown", ""])
    for row in rows:
        if row["status"] == "completed":
            report_lines.append(
                f"- {row['variant']} (seed {row['seed']}): "
                f"translation=`{row['translation_models'] or 'not reported'}`; "
                f"report=`{row['report_model'] or 'none'}`"
            )
    atomic_write_text(root / "report.md", "\n".join(report_lines) + "\n")
    return root


async def _run_matrix_jobs(
    jobs: list[tuple[str, int, int, RunConfig, Path]],
    *,
    concurrency: int,
    failure_policy: str = "stop",
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def execute(job: tuple[str, int, int, RunConfig, Path]) -> dict[str, Any]:
        variant_name, repetition, seed, config, run_directory = job
        async with semaphore:
            try:
                run_dir = await run_pipeline(config, run_directory=run_directory)
            except Exception as error:
                if failure_policy == "stop":
                    raise
                return {
                    "variant": variant_name,
                    "repetition": repetition,
                    "seed": seed,
                    "status": "failed",
                    "run_directory": str(run_directory),
                    "error": f"{type(error).__name__}: {error}",
                }
            metrics = read_json(run_dir / "metrics.json")
            usage = _usage_summary(run_dir)
            models = _model_summary(run_dir)
            semantic = metrics.get("semantic", {})
            return {
                "variant": variant_name,
                "repetition": repetition,
                "seed": seed,
                "status": "completed",
                "run_directory": str(run_dir),
                "final": str(run_dir / "final.txt"),
                "length_ratio": metrics["structural"]["length_ratio"],
                "token_overlap": metrics["lexical"]["token_overlap"],
                "character_4gram_similarity": metrics["lexical"][
                    "character_4gram_similarity"
                ],
                "semantic_similarity": (
                    semantic.get("overall_similarity")
                    if isinstance(semantic, dict) and semantic.get("available") is True
                    else None
                ),
                **models,
                **usage,
            }

    tasks = [
        asyncio.create_task(execute(job), name=f"matrix-run-{index + 1}")
        for index, job in enumerate(jobs)
    ]
    try:
        # Keep summary rows in configuration order, regardless of completion order.
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "run"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _usage_summary(run_directory: Path) -> dict[str, int | float | bool]:
    translation_hops = 0
    generation_requests = 0
    stage_retries = 0
    for path in (run_directory / "chunks").glob("*/stage-*.json"):
        stage = read_json(path)
        details = stage.get("translation_details")
        if isinstance(details, list):
            translation_hops += len(details)
        if stage.get("stage_type") in {
            "conservative_repair",
            "reconstruction",
            "contextual_reconstruction",
            "memory_extraction",
        } and stage.get("applied", True):
            generation_requests += 1
        attempts = stage.get("attempts")
        if isinstance(attempts, int):
            stage_retries += max(0, attempts - 1)
    manifest = read_json(run_directory / "manifest.json")
    if manifest.get("report_generation"):
        generation_requests += 1
    request_summary = manifest.get("request_summary")
    approximate = not isinstance(request_summary, dict)
    request_summary = request_summary if isinstance(request_summary, dict) else {}
    retries = request_summary.get("retries")
    if not isinstance(retries, (int, float)) or isinstance(retries, bool):
        retries = stage_retries
        approximate = True
    started = manifest.get("started_at")
    completed = manifest.get("completed_at")
    duration = 0.0
    if isinstance(started, str) and isinstance(completed, str):
        duration = (
            datetime.fromisoformat(completed) - datetime.fromisoformat(started)
        ).total_seconds()
    return {
        "http_attempts": int(request_summary.get("http_attempts", 0)),
        "translation_hops": translation_hops,
        "generation_requests": generation_requests,
        "prompt_tokens": float(request_summary.get("prompt_tokens", 0.0)),
        "completion_tokens": float(request_summary.get("completion_tokens", 0.0)),
        "total_tokens": float(request_summary.get("total_tokens", 0.0)),
        "cost_usd": float(request_summary.get("cost_usd", 0.0)),
        "input_characters": float(request_summary.get("input_characters", 0.0)),
        "output_characters": float(request_summary.get("output_characters", 0.0)),
        "segments": float(request_summary.get("segments", 0.0)),
        "retries": int(retries),
        "duration_seconds": duration,
        "usage_approximate": approximate,
    }


def _model_summary(run_directory: Path) -> dict[str, str]:
    manifest = read_json(run_directory / "manifest.json")
    labels: list[str] = []
    for item in manifest.get("translation_models", []):
        if not isinstance(item, dict):
            continue
        provider = item.get("engine") or item.get("provider") or item.get("provider_type")
        model = item.get("model")
        revision = item.get("revision")
        label = "/".join(str(value) for value in (provider, model) if value)
        if revision:
            label = f"{label}@{revision}" if label else str(revision)
        if label and label not in labels:
            labels.append(label)
    report_generation = manifest.get("report_generation")
    report_label = ""
    if isinstance(report_generation, dict):
        provider = report_generation.get("provider")
        model = report_generation.get("model")
        report_label = "/".join(str(value) for value in (provider, model) if value)
    return {
        "translation_models": "; ".join(labels),
        "report_model": report_label,
    }


_AGGREGATE_METRICS = (
    "length_ratio",
    "token_overlap",
    "character_4gram_similarity",
    "semantic_similarity",
    "http_attempts",
    "translation_hops",
    "generation_requests",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "input_characters",
    "output_characters",
    "segments",
    "retries",
    "duration_seconds",
)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    result: list[dict[str, Any]] = []
    for variant in variants:
        successful = [
            row for row in rows if row["variant"] == variant and row.get("status") == "completed"
        ]
        failures = sum(
            1 for row in rows if row["variant"] == variant and row.get("status") == "failed"
        )
        for metric in _AGGREGATE_METRICS:
            values = [
                float(row[metric])
                for row in successful
                if isinstance(row.get(metric), (int, float))
                and not isinstance(row.get(metric), bool)
            ]
            if not values:
                continue
            result.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n": len(values),
                    "failures": failures,
                    "mean": statistics.fmean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )
    return result


def _paired_deltas(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    baseline_rows = {
        int(row["seed"]): row
        for row in completed
        if row["variant"] == baseline
    }
    variants = list(
        dict.fromkeys(str(row["variant"]) for row in rows if row["variant"] != baseline)
    )
    result: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = {
            int(row["seed"]): row
            for row in completed
            if row["variant"] == variant
        }
        common = sorted(set(baseline_rows) & set(variant_rows))
        expected = {
            int(row["seed"])
            for row in rows
            if row["variant"] in {baseline, variant}
        }
        failures = sum(
            1
            for row in rows
            if row["variant"] in {baseline, variant} and row.get("status") == "failed"
        )
        for metric in _AGGREGATE_METRICS:
            deltas = [
                float(variant_rows[key][metric]) - float(baseline_rows[key][metric])
                for key in common
                if isinstance(variant_rows[key].get(metric), (int, float))
                and isinstance(baseline_rows[key].get(metric), (int, float))
            ]
            result.append(
                {
                    "variant": variant,
                    "baseline": baseline,
                    "metric": metric,
                    "n": len(deltas),
                    "missing_pairs": len(expected) - len(deltas),
                    "failures": failures,
                    "mean_delta": statistics.fmean(deltas) if deltas else None,
                    "minimum_delta": min(deltas) if deltas else None,
                    "maximum_delta": max(deltas) if deltas else None,
                }
            )
    return result


def run_matrix_sync(path: Path) -> Path:
    return asyncio.run(run_matrix(path.resolve()))
