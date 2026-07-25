from __future__ import annotations

from pathlib import Path


def test_standard_prompts_and_configs_have_no_seeded_motifs() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        "сделай смешнее",
        "добавь абсурда",
        "добавь чёрный юмор",
        "добавь алкоголь",
        "добавь политику",
        "добавь насилие",
    ]
    files = [*root.glob("prompts/*.txt"), *root.glob("configs/*.yaml")]
    corpus = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
    assert all(phrase not in corpus for phrase in forbidden)


def test_reconstruction_stage_api_has_no_original_argument() -> None:
    from inspect import signature

    from semantic_telephone.stages.reconstruction import reconstruct_text

    parameters = signature(reconstruct_text).parameters
    assert "original" not in parameters
    assert "damaged_text" in parameters

