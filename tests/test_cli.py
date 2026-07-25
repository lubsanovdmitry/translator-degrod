from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from semantic_telephone.cli import app
from semantic_telephone.models import RunConfig

runner = CliRunner()


def test_validate_command(config_file: Path) -> None:
    result = runner.invoke(app, ["validate", str(config_file)])
    assert result.exit_code == 0
    assert "Valid configuration" in result.stdout


def test_init_does_not_overwrite(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("keep", encoding="utf-8")
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert input_path.read_text(encoding="utf-8") == "keep"
    assert (tmp_path / "semantic-telephone.yaml").exists()


def test_invalid_yaml_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("run: []\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 2
    assert "Configuration error" in result.output


def test_run_provider_runtime_error_has_clear_message(
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(config: RunConfig) -> Path:
        del config
        raise RuntimeError("provider preflight failed")

    monkeypatch.setattr("semantic_telephone.cli.run_pipeline", fail)
    result = runner.invoke(app, ["run", str(config_file)])

    assert result.exit_code == 1
    assert "Run failed: provider preflight failed" in result.output
