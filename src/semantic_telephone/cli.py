from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer

from .config import ConfigError, config_from_resolved, load_config
from .experiments import run_matrix_sync
from .memory import MemoryStore
from .pipeline import run_pipeline
from .reporting import regenerate_report
from .utils.files import atomic_write_text, read_json

app = typer.Typer(help="Reproducible recursive translation experiments.", no_args_is_help=True)
memory_app = typer.Typer(help="Inspect or clear experimental inferred memory.")
app.add_typer(memory_app, name="memory")


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        force=True,
    )


@app.command()
def init(directory: Annotated[Path, typer.Argument()] = Path(".")) -> None:
    """Create a minimal mock-only starter workspace without overwriting files."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        ".env": (
            "# Optional API credentials. Mock providers need none.\n"
            "OPENROUTER_API_KEY=\n"
            "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n"
            "OPENROUTER_MODEL=\n"
            "LIBRETRANSLATE_BASE_URL=http://localhost:5000\n"
            "LIBRETRANSLATE_API_KEY=\n"
        ),
        "input.txt": "",
        "semantic-telephone.yaml": _starter_config(),
        "SEMANTIC_TELEPHONE.md": (
            "# Mock smoke test only\n\n"
            "This starter checks the CLI and artifact pipeline; it does not perform "
            "machine translation. For a real run, use the repository's "
            "`configs/nllb_only.yaml` or `configs/mixed_local.yaml` profile.\n\n"
            "Put text in `input.txt`, then run:\n\n"
            "```bash\nsemantic-telephone validate semantic-telephone.yaml\n"
            "semantic-telephone run semantic-telephone.yaml\n```\n"
        ),
    }
    for name, content in files.items():
        path = directory / name
        if not path.exists():
            atomic_write_text(path, content)
            typer.echo(f"created {path}")
        else:
            typer.echo(f"kept existing {path}")


@app.command()
def validate(config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Validate YAML and provider-independent invariants."""
    try:
        loaded = load_config(config)
    except ConfigError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(f"Valid configuration: {loaded.name} ({len(loaded.pipeline)} stages)")


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Start a new run."""
    _configure_logging(verbose)
    try:
        directory = asyncio.run(run_pipeline(load_config(config)))
    except (ConfigError, ValueError, OSError, RuntimeError) as error:
        typer.echo(f"Run failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Completed: {directory}")


@app.command()
def resume(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Continue from valid per-stage checkpoints."""
    _configure_logging(verbose)
    manifest = read_json(run_directory / "manifest.json")
    config = config_from_resolved(manifest["resolved_config"])
    try:
        directory = asyncio.run(run_pipeline(config, run_directory=run_directory))
    except (ValueError, OSError, RuntimeError) as error:
        typer.echo(f"Resume failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Completed: {directory}")


@app.command()
def matrix(config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Run multiple variants and produce a comparison table."""
    _configure_logging()
    try:
        directory = run_matrix_sync(config)
    except (ConfigError, ValueError, OSError, RuntimeError) as error:
        typer.echo(f"Matrix failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Completed matrix: {directory}")


@app.command()
def inspect(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    chunk: Annotated[int, typer.Option("--chunk", min=1)] = 1,
) -> None:
    """Print the original and all generations for one chunk."""
    directory = run_directory / "chunks" / f"{chunk:04d}"
    if not directory.exists():
        typer.echo(f"Chunk {chunk} not found", err=True)
        raise typer.Exit(2)
    typer.echo("Original\n--------")
    typer.echo((directory / "source.txt").read_text(encoding="utf-8"))
    for stage_path in sorted(directory.glob("stage-*-output.txt")):
        typer.echo(f"\n→ {stage_path.name}\n{'-' * 8}")
        typer.echo(stage_path.read_text(encoding="utf-8"))
    typer.echo("\n→ Final\n--------")
    typer.echo((directory / "final.txt").read_text(encoding="utf-8"))


@app.command()
def compare(
    run_a: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    run_b: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Create a side-by-side Markdown comparison of final texts."""
    first = (run_a / "final.txt").read_text(encoding="utf-8").splitlines()
    second = (run_b / "final.txt").read_text(encoding="utf-8").splitlines()
    lines = ["# Run comparison", "", "| A | B |", "|---|---|"]
    for index in range(max(len(first), len(second))):
        left = first[index] if index < len(first) else ""
        right = second[index] if index < len(second) else ""
        lines.append(f"| {_cell(left)} | {_cell(right)} |")
    destination = output or Path("comparison.md")
    atomic_write_text(destination, "\n".join(lines) + "\n")
    typer.echo(f"Created: {destination}")


@app.command()
def report(run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Regenerate report.md from saved artifacts."""
    typer.echo(f"Created: {regenerate_report(run_directory)}")


@memory_app.command("show")
def memory_show(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    state = run_directory / "memory" / "state.json"
    typer.echo(state.read_text(encoding="utf-8") if state.exists() else '{"items": []}')


@memory_app.command("clear")
def memory_clear(
    run_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    manifest = read_json(run_directory / "manifest.json")
    config = config_from_resolved(manifest["resolved_config"])
    MemoryStore(run_directory / "memory", half_life=config.memory.half_life_chunks).clear()
    typer.echo(f"Cleared inferred memory in {run_directory / 'memory'}")


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _starter_config() -> str:
    return """# Smoke test only: mock providers do not perform machine translation.
run:
  name: translate-only
  seed: 1080
  source_language: ru
  target_language: ru
  output_root: runs
input:
  path: input.txt
chunking:
  strategy: target_chars
  target_chars: 1100
  max_chars: 1800
  paragraph_overlap: 1
translation:
  provider: mock
  route_mode: fixed
  languages: [ru, ka, ar, en, ru]
generation:
  provider: mock
  model: mock
context:
  enabled: false
memory:
  enabled: false
pipeline:
  - type: translation_cycle
    hops: {min: 4, max: 6}
  - type: final_translation
runtime:
  retries: 3
  resume: true
"""


if __name__ == "__main__":
    app()
