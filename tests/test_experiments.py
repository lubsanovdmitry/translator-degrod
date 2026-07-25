from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from semantic_telephone.config import ConfigError
from semantic_telephone.experiments import run_matrix
from semantic_telephone.models import RunConfig
from semantic_telephone.pipeline import run_pipeline as real_run_pipeline
from semantic_telephone.utils.files import read_json


def test_matrix_runs_are_bounded_and_rows_stay_ordered(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix_path = config_file.parent / "matrix.yaml"
    matrix_path.write_text(
        f"""experiment:
  name: parallel-test
  seeds: [9]
  concurrency: 2
variants:
  - name: slow-first
    config: {config_file}
  - name: fast-second
    config: {config_file}
""",
        encoding="utf-8",
    )
    active = 0
    maximum_active = 0

    async def tracked_run_pipeline(
        config: RunConfig,
        *,
        run_directory: Path | None = None,
    ) -> Path:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            delay = 0.03 if config.name.startswith("slow-first") else 0
            await asyncio.sleep(delay)
            return await real_run_pipeline(config, run_directory=run_directory)
        finally:
            active -= 1

    monkeypatch.setattr(
        "semantic_telephone.experiments.run_pipeline",
        tracked_run_pipeline,
    )

    directory = asyncio.run(run_matrix(matrix_path))
    rows = read_json(directory / "summary.json")["runs"]

    assert maximum_active == 2
    assert [row["variant"] for row in rows] == ["slow-first", "fast-second"]
    assert rows[0]["run_directory"] != rows[1]["run_directory"]


def test_matrix_concurrency_must_be_positive_integer(
    config_file: Path,
) -> None:
    matrix_path = config_file.parent / "invalid-matrix.yaml"
    matrix_path.write_text(
        f"""experiment:
  name: invalid
  concurrency: 0
variants:
  - config: {config_file}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="experiment.concurrency"):
        asyncio.run(run_matrix(matrix_path))
