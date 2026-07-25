from __future__ import annotations

import asyncio
import csv
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
    for variant in variants:
        if not isinstance(variant, dict) or "config" not in variant:
            raise ConfigError("each matrix variant requires config")
        config_path = Path(str(variant["config"]))
        if not config_path.is_absolute():
            config_path = path.parent / config_path
        base_config = load_config(config_path)
        variant_name = str(variant.get("name", base_config.name))
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
    rows = await _run_matrix_jobs(jobs, concurrency=concurrency)
    atomic_write_json(root / "summary.json", {"experiment": experiment, "runs": rows})
    _write_csv(root / "summary.csv", rows)
    report_lines = [
        f"# Experiment matrix: {name}",
        "",
        "> These comparisons are diagnostic and do not establish artistic quality or a",
        "> scientifically proven effect.",
        "",
        "| Variant | Rep | Seed | Length ratio | Token overlap | Requests | Final |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        relative = Path(row["final"]).relative_to(root)
        report_lines.append(
            f"| {row['variant']} | {row['repetition']} | {row['seed']} | "
            f"{row['length_ratio']:.3f} | {row['token_overlap']:.3f} | {row['requests']} | "
            f"[text]({relative.as_posix()}) |"
        )
    atomic_write_text(root / "report.md", "\n".join(report_lines) + "\n")
    return root


async def _run_matrix_jobs(
    jobs: list[tuple[str, int, int, RunConfig, Path]],
    *,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def execute(job: tuple[str, int, int, RunConfig, Path]) -> dict[str, Any]:
        variant_name, repetition, seed, config, run_directory = job
        async with semaphore:
            run_dir = await run_pipeline(config, run_directory=run_directory)
            metrics = read_json(run_dir / "metrics.json")
            usage = _usage_summary(run_dir)
            return {
                "variant": variant_name,
                "repetition": repetition,
                "seed": seed,
                "run_directory": str(run_dir),
                "final": str(run_dir / "final.txt"),
                "length_ratio": metrics["structural"]["length_ratio"],
                "token_overlap": metrics["lexical"]["token_overlap"],
                "character_4gram_similarity": metrics["lexical"][
                    "character_4gram_similarity"
                ],
                "requests": usage["requests"],
                "usage_total": usage["usage_total"],
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _usage_summary(run_directory: Path) -> dict[str, int | float]:
    requests = 0
    usage_total = 0.0
    for path in (run_directory / "chunks").glob("*/stage-*.json"):
        stage = read_json(path)
        if stage.get("provider") not in {None, "none"} and stage.get("applied", True):
            requests += 1
        usage = stage.get("usage")
        if isinstance(usage, dict):
            usage_total += sum(
                float(value) for value in usage.values() if isinstance(value, (int, float))
            )
    return {"requests": requests, "usage_total": usage_total}


def run_matrix_sync(path: Path) -> Path:
    return asyncio.run(run_matrix(path.resolve()))
