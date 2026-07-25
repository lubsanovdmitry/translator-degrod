from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from .config import ConfigError, config_from_resolved, load_config
from .experiments import run_matrix_sync
from .memory import MemoryStore
from .pipeline import run_pipeline
from .planning import create_plan, doctor_config
from .reporting import regenerate_report
from .resources import profile_names, profile_text
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
def init(
    directory: Annotated[Path, typer.Argument()] = Path("."),
    profile: Annotated[str, typer.Option("--profile", "-p")] = "translate_only",
) -> None:
    """Create a starter workspace from a bundled profile without overwriting files."""
    if profile not in profile_names():
        choices = ", ".join(profile_names())
        raise typer.BadParameter(f"unknown profile {profile!r}; choose one of: {choices}")
    directory.mkdir(parents=True, exist_ok=True)
    starter = profile_text(profile, standalone=True)
    mock_only = profile in {
        "translate_only",
        "sparse_repair",
        "iterative_reconstruction",
        "rolling_context",
        "inferred_memory",
    }
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
        "semantic-telephone.yaml": starter,
        "SEMANTIC_TELEPHONE.md": (
            ("# Mock smoke test\n\n" if mock_only else "# Semantic Telephone\n\n")
            + (
                "Этот профиль проверяет CLI, checkpoints и артефакты. "
                "Он не выполняет машинный перевод.\n\n"
                if mock_only
                else "Это реальный профиль. Перед запуском проверьте зависимости, "
                "ключи API и доступность моделей.\n\n"
            )
            + "`semantic-telephone.yaml` — локальная копия встроенного профиля. "
            "Её можно изменять.\n\n"
            "Поместите текст в `input.txt`, затем выполните:\n\n"
            "```bash\nsemantic-telephone validate semantic-telephone.yaml\n"
            "semantic-telephone plan semantic-telephone.yaml\n"
            + (
                ""
                if mock_only
                else "semantic-telephone doctor semantic-telephone.yaml\n"
            )
            + "semantic-telephone run semantic-telephone.yaml\n```\n"
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


@app.command("plan")
def plan_command(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_format: Annotated[str, typer.Option("--format")] = "text",
) -> None:
    """Resolve routes, providers, resources, and request bounds without network access."""
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be text or json")
    try:
        result = create_plan(load_config(config))
    except (ConfigError, ValueError, OSError, RuntimeError) as error:
        typer.echo(f"Plan failed: {error}", err=True)
        raise typer.Exit(2) from error
    _emit_structured(result, output_format)


@app.command()
def doctor(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    allow_downloads: Annotated[bool, typer.Option("--allow-downloads")] = False,
    output_format: Annotated[str, typer.Option("--format")] = "text",
) -> None:
    """Check configured services, dependencies, and caches without paid generation."""
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be text or json")
    try:
        result = asyncio.run(
            doctor_config(load_config(config), allow_downloads=allow_downloads)
        )
    except (ConfigError, ValueError, OSError, RuntimeError) as error:
        typer.echo(f"Doctor failed: {error}", err=True)
        raise typer.Exit(2) from error
    _emit_structured(result, output_format)
    if not result["ok"]:
        raise typer.Exit(2)


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


def _emit_structured(value: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        return
    typer.echo(f"Run: {value.get('run', 'unknown')}")
    if isinstance(value.get("checks"), list):
        for item in value["checks"]:
            if isinstance(item, dict):
                typer.echo(
                    f"[{str(item.get('status', 'unknown')).upper()}] "
                    f"{item.get('name')}: {item.get('detail')}"
                )
        return
    typer.echo(f"Chunks: {value.get('chunks')}")
    typer.echo(f"Effective concurrency: {value.get('effective_concurrency')}")
    bounds = value.get("request_bounds")
    if isinstance(bounds, dict):
        typer.echo(
            "Provider calls: "
            f"{bounds.get('provider_calls')} (maximum attempts: {bounds.get('maximum_attempts')})"
        )
    routes = value.get("planned_routes")
    if isinstance(routes, list):
        for route in routes:
            if isinstance(route, dict):
                typer.echo(
                    f"Chunk {route.get('chunk')} stage {route.get('stage')}: "
                    + " -> ".join(str(item) for item in route.get("languages", []))
                )


if __name__ == "__main__":
    app()
