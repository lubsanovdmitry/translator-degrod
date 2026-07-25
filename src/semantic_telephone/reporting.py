from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .utils.files import atomic_write_text, read_json

MODE_DESCRIPTIONS = {
    "translate_only": "Translation cycles without LLM reconstruction.",
    "sparse_repair": "Translation degradation with probabilistic conservative grammar repair.",
    "iterative_reconstruction": "Alternating translation and moderate local reconstruction.",
    "rolling_context": "Reconstruction with a limited tail of prior damaged outputs.",
    "inferred_memory": "Experimental reconstruction with fallible automatically extracted memory.",
    "raw_translation": "Mixed local MT output without any LLM stage.",
    "grammar_repair": "Mixed local MT followed only by constrained grammar repair.",
    "conservative_reconstruction": (
        "Mixed local MT followed by reconstruction without scene expansion."
    ),
    "aggressive_reconstruction": (
        "Mixed local MT followed by permissive scene-expanding reconstruction."
    ),
    "mixed_local": "Legacy mixed local profile; its reconstruction behavior is unchanged.",
}


def create_report(
    run_directory: Path,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    routes: list[list[str]],
    warnings: list[str],
) -> Path:
    lines = [
        f"# Semantic Telephone report: {config['name']}",
        "",
        "> Metrics below are diagnostic heuristics, not scientific evidence or an objective",
        "> evaluation of literary quality.",
        "",
        "## Configuration",
        "",
        f"- Mode: `{config['name']}`",
        f"- Seed: `{config['seed']}`",
        f"- Source/target: `{config['source_language']}` → `{config['target_language']}`",
        (
            "- Translation provider(s): `"
            + (
                ", ".join(config["translation"].get("providers", {}))
                if config["translation"].get("providers")
                else config["translation"]["provider"]
            )
            + "`"
        ),
        (
            "- Engine routing: `"
            + config["translation"].get("engine_routing", {}).get("mode", "single_engine")
            + "`"
        ),
        f"- Generation provider: `{config['generation']['provider']}`",
        f"- Memory enabled: `{config['memory']['enabled']}`",
        "",
        MODE_DESCRIPTIONS.get(config["name"], "Custom pipeline configuration."),
        "",
        "## Language routes",
        "",
    ]
    lines.extend(f"- {' → '.join(route)}" for route in routes[:50])
    lines.extend(
        [
            "",
            "## Summary metrics",
            "",
            "```json",
            json.dumps(metrics, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Most changed fragments",
            "",
        ]
    )
    lines.extend(_changed_fragments(run_directory))
    lines.extend(
        [
            "",
            "## Repeated entity variants",
            "",
            json.dumps(metrics.get("entities", {}), ensure_ascii=False),
            "",
            "## Possible generative expansion",
            "",
            json.dumps(metrics.get("possible_generative_expansion", {}), ensure_ascii=False),
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in warnings or ["None."])
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "The seed, resolved configuration, prompts and their checksums are stored in the",
            "manifest. External APIs may remain nondeterministic even when they accept a seed.",
            "",
            "## Cost and requests",
            "",
            "Provider usage is stored per stage. Monetary cost is reported only when supplied",
            "by the configured provider.",
            "",
        ]
    )
    path = run_directory / "report.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def _changed_fragments(run_directory: Path) -> list[str]:
    changed: list[tuple[float, str, str, str]] = []
    for directory in (run_directory / "chunks").glob("[0-9][0-9][0-9][0-9]"):
        source_path, final_path = directory / "source.txt", directory / "final.txt"
        if not source_path.exists() or not final_path.exists():
            continue
        source = source_path.read_text(encoding="utf-8")
        final = final_path.read_text(encoding="utf-8")
        score = 1 - SequenceMatcher(None, source, final).ratio()
        changed.append((score, directory.name, source, final))
    if not changed:
        return ["No completed fragments."]
    lines: list[str] = []
    for score, name, source, final in sorted(changed, reverse=True)[:3]:
        lines.extend(
            [
                f"### Chunk {name} (change heuristic: {score:.3f})",
                "",
                f"- Source: {_excerpt(source)}",
                f"- Final: {_excerpt(final)}",
                "",
            ]
        )
    return lines


def _excerpt(value: str, limit: int = 280) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def regenerate_report(run_directory: Path) -> Path:
    manifest = read_json(run_directory / "manifest.json")
    metrics = read_json(run_directory / "metrics.json")
    routes: list[list[str]] = []
    warnings: list[str] = []
    for stage_path in sorted((run_directory / "chunks").glob("*/stage-*.json")):
        stage = read_json(stage_path)
        route = stage.get("route")
        if isinstance(route, list) and route:
            routes.append([str(item) for item in route])
        stage_warnings = stage.get("warnings")
        if isinstance(stage_warnings, list):
            warnings.extend(str(item) for item in stage_warnings)
    return create_report(
        run_directory,
        config=manifest["resolved_config"],
        metrics=metrics,
        routes=routes,
        warnings=sorted(set(warnings)),
    )
