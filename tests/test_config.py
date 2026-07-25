from __future__ import annotations

from pathlib import Path

import pytest

from semantic_telephone.config import ConfigError, config_from_resolved, load_config


def test_yaml_validation_and_manifest_rehydration(config_file: Path) -> None:
    config = load_config(config_file)
    restored = config_from_resolved(config.to_dict())
    assert restored.name == config.name
    assert restored.pipeline[0].type == config.pipeline[0].type


def test_memory_disabled_by_default(config_file: Path) -> None:
    assert load_config(config_file).memory.enabled is False


def test_invalid_probability(config_file: Path) -> None:
    text = config_file.read_text(encoding="utf-8").replace(
        "  - type: reconstruction", "  - type: reconstruction\n    probability: 2"
    )
    config_file.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="probability"):
        load_config(config_file)

