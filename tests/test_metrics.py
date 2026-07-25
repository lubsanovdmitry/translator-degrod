from __future__ import annotations

from semantic_telephone.metrics import calculate_metrics


def test_metrics_have_diagnostic_sections() -> None:
    metrics = calculate_metrics("Лира вошла. Было тихо.", "Лира быстро вошла. Было очень тихо.")
    assert {"structural", "lexical", "entities", "possible_generative_expansion"} <= set(metrics)
    assert metrics["structural"]["length_ratio"] > 1
    assert "proof" in metrics["possible_generative_expansion"]["caveat"].lower()
