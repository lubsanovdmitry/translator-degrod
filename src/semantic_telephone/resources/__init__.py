from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml


def prompt_text(name: str) -> str:
    resource = files(__package__).joinpath("prompts", f"{name}.txt")
    if not resource.is_file():
        raise FileNotFoundError(f"unknown built-in prompt: {name}")
    return resource.read_text(encoding="utf-8")


def profile_text(name: str, *, standalone: bool = False) -> str:
    normalized = name.removesuffix(".yaml")
    resource = files(__package__).joinpath("profiles", f"{normalized}.yaml")
    if not resource.is_file():
        raise FileNotFoundError(f"unknown built-in profile: {name}")
    text = resource.read_text(encoding="utf-8")
    if not standalone:
        return text
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise TypeError(f"built-in profile {name} must be a mapping")
    input_section = raw.setdefault("input", {})
    if isinstance(input_section, dict):
        input_section["path"] = "input.txt"
    run = raw.setdefault("run", {})
    if isinstance(run, dict):
        run["output_root"] = "runs"
    prompts = raw.get("prompts")
    if isinstance(prompts, dict):
        for key, value in list(prompts.items()):
            text_value = str(value)
            if not text_value.startswith("builtin:"):
                prompts[key] = f"builtin:{Path(text_value).stem}"
    return yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)


def profile_names() -> list[str]:
    root = files(__package__).joinpath("profiles")
    return sorted(
        item.name.removesuffix(".yaml")
        for item in root.iterdir()
        if item.name.endswith(".yaml") and "matrix" not in item.name
    )
