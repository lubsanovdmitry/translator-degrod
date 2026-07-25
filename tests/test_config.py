from __future__ import annotations

from pathlib import Path

import pytest

from semantic_telephone.config import ConfigError, config_from_resolved, load_config
from semantic_telephone.models import StageType
from semantic_telephone.pipeline import load_prompts


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


def test_memory_extraction_cannot_precede_translation(config_file: Path) -> None:
    text = config_file.read_text(encoding="utf-8")
    text = text.replace("memory: {enabled: false}", "memory: {enabled: true}")
    text = text.replace(
        "pipeline:\n  - type: translation_cycle",
        "pipeline:\n  - type: memory_extraction\n  - type: translation_cycle",
    )
    config_file.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="earlier enabled translation_cycle"):
        load_config(config_file)


def test_context_limits_are_validated(config_file: Path) -> None:
    text = config_file.read_text(encoding="utf-8").replace(
        "context: {enabled: false}",
        "context: {enabled: true, previous_chunks: 0}",
    )
    config_file.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="context.previous_chunks"):
        load_config(config_file)


def test_runtime_concurrency_must_be_positive_integer(config_file: Path) -> None:
    text = config_file.read_text(encoding="utf-8").replace(
        "  concurrency: 1",
        "  concurrency: 1.5",
    )
    config_file.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="runtime.concurrency"):
        load_config(config_file)


def test_mixed_local_reconstruction_profiles_share_translation_input() -> None:
    names = (
        "mixed_local",
        "raw_translation",
        "grammar_repair",
        "conservative_reconstruction",
        "aggressive_reconstruction",
    )
    configs = [load_config(f"configs/{name}.yaml") for name in names]
    baseline = configs[0]
    assert baseline.temperatures["reconstruction"] == 0.6
    assert baseline.pipeline[1].parameters == {}
    for config in configs[1:]:
        assert config.seed == baseline.seed
        assert config.chunking == baseline.chunking
        assert config.route == baseline.route
        assert config.translation == baseline.translation

    raw = configs[1]
    assert raw.generation.provider == "mock"
    assert all(
        stage.type
        not in {
            StageType.CONSERVATIVE_REPAIR,
            StageType.RECONSTRUCTION,
            StageType.CONTEXTUAL_RECONSTRUCTION,
        }
        for stage in raw.pipeline
    )

    grammar_stage = configs[2].pipeline[1]
    assert configs[2].temperatures["conservative_repair"] == 0.1
    assert grammar_stage.parameters == {
        "max_length_ratio": 1.05,
        "allow_new_events": False,
    }

    conservative_stage = configs[3].pipeline[1]
    assert configs[3].temperatures["reconstruction"] == 0.35
    assert conservative_stage.parameters["max_length_ratio"] == 1.15
    assert conservative_stage.parameters["max_new_sentences_per_chunk"] == 2
    assert conservative_stage.parameters["repetition_policy"] == "preserve"

    aggressive_stage = configs[4].pipeline[1]
    assert configs[4].temperatures["reconstruction"] == 0.7
    assert aggressive_stage.parameters["max_length_ratio"] == 2.5
    assert aggressive_stage.parameters["repetition_policy"] == "rationalize"
    assert aggressive_stage.parameters["allow_scene_expansion"] is True

    conservative_prompts, _ = load_prompts(configs[3])
    assert (
        "не превращай повторяющиеся слова в отдельную атмосферную сцену"
        in conservative_prompts[StageType.RECONSTRUCTION]
    )
    aggressive_prompts, _ = load_prompts(configs[4])
    assert (
        "разрешается расширять существующие сцены"
        in aggressive_prompts[StageType.RECONSTRUCTION]
    )


def test_invalid_reconstruction_policy_is_rejected(config_file: Path) -> None:
    text = config_file.read_text(encoding="utf-8").replace(
        "  - type: reconstruction",
        "  - type: reconstruction\n    repetition_policy: invent",
    )
    config_file.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="repetition_policy"):
        load_config(config_file)
