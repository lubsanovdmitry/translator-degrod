from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from typing import Any

from .models import SemanticMetricConfig
from .utils.files import read_json
from .utils.text import sentences, tokens

WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)


class SemanticMetricsUnavailable(RuntimeError):
    pass


class SentenceTransformerMetrics:
    def __init__(self, config: SemanticMetricConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as error:
            raise SemanticMetricsUnavailable(
                "semantic metrics require the optional 'semantic-metrics' dependencies"
            ) from error
        kwargs: dict[str, Any] = {
            "local_files_only": not config.allow_downloads,
        }
        if config.revision:
            kwargs["revision"] = config.revision
        if config.device != "auto":
            kwargs["device"] = config.device
        try:
            self.model = SentenceTransformer(config.model, **kwargs)
        except Exception as error:
            mode = "download-enabled" if config.allow_downloads else "local-cache-only"
            raise SemanticMetricsUnavailable(
                f"cannot load semantic model {config.model!r} ({mode}): "
                f"{type(error).__name__}: {error}"
            ) from error
        self.config = config
        first_module = getattr(self.model, "_first_module", lambda: None)()
        auto_model = getattr(first_module, "auto_model", None)
        model_config = getattr(auto_model, "config", None)
        self.resolved_revision = (
            getattr(model_config, "_commit_hash", None) or config.revision
        )
        self.resolved_device = str(getattr(self.model, "device", config.device))

    def calculate(self, source: str, result: str, run_directory: Path) -> dict[str, Any]:
        comparisons: list[tuple[str, str, str, dict[str, Any]]] = [
            ("overall", source, result, {})
        ]
        for chunk_directory in sorted((run_directory / "chunks").glob("[0-9][0-9][0-9][0-9]")):
            source_path = chunk_directory / "source.txt"
            final_path = chunk_directory / "final.txt"
            if source_path.exists() and final_path.exists():
                comparisons.append(
                    (
                        "chunk",
                        source_path.read_text(encoding="utf-8"),
                        final_path.read_text(encoding="utf-8"),
                        {"chunk": chunk_directory.name},
                    )
                )
            for stage_path in sorted(chunk_directory.glob("stage-*.json")):
                stage = read_json(stage_path)
                details = stage.get("translation_details", [])
                if not isinstance(details, list) or not details:
                    continue
                previous = str(stage.get("input_text", ""))
                for index, detail in enumerate(details):
                    if not isinstance(detail, dict) or not detail.get("output_path"):
                        continue
                    hop_path = chunk_directory / str(detail["output_path"])
                    if not hop_path.exists():
                        continue
                    current = hop_path.read_text(encoding="utf-8")
                    comparisons.append(
                        (
                            "hop",
                            previous,
                            current,
                            {
                                "chunk": chunk_directory.name,
                                "stage": stage.get("stage_type"),
                                "hop": index + 1,
                                "source_language": detail.get("source_language"),
                                "target_language": detail.get("target_language"),
                                "provider": detail.get("provider"),
                                "model": detail.get("model"),
                            },
                        )
                    )
                    previous = current
        texts: list[str] = []
        indices: dict[str, int] = {}
        for _, left, right, _ in comparisons:
            for text in (left, right):
                if text not in indices:
                    indices[text] = len(texts)
                    texts.append(text)
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        rows: list[dict[str, Any]] = []
        for kind, left, right, metadata in comparisons:
            left_vector = vectors[indices[left]]
            right_vector = vectors[indices[right]]
            score = float(sum(float(a) * float(b) for a, b in zip(left_vector, right_vector)))
            rows.append({"kind": kind, "similarity": score, "drift": 1.0 - score, **metadata})
        chunks = [row for row in rows if row["kind"] == "chunk"]
        hops = [row for row in rows if row["kind"] == "hop"]
        largest = max(hops, key=lambda row: float(row["drift"])) if hops else None
        aggregate_route_drift = (
            sum(float(row["drift"]) for row in hops) / len(hops) if hops else None
        )
        return {
            "available": True,
            "provider": self.config.provider,
            "model": self.config.model,
            "revision": self.config.revision,
            "resolved_revision": self.resolved_revision,
            "device": self.resolved_device,
            "overall_similarity": rows[0]["similarity"],
            "chunk_similarity": chunks,
            "hop_drift": hops,
            "mean_hop_drift": aggregate_route_drift,
            "aggregate_route_drift": aggregate_route_drift,
            "largest_drift_hop": largest,
            "caveat": "Embedding similarity is diagnostic and model-dependent.",
        }


def prepare_semantic_metrics(
    config: SemanticMetricConfig,
) -> SentenceTransformerMetrics | None:
    return SentenceTransformerMetrics(config) if config.enabled else None


def _words(text: str) -> list[str]:
    return [word.lower() for word in WORD_RE.findall(text)]


def _ngrams(value: str, size: int) -> Counter[str]:
    compact = re.sub(r"\s+", " ", value.lower())
    return Counter(compact[index : index + size] for index in range(max(0, len(compact) - size + 1)))


def _counter_similarity(first: Counter[str], second: Counter[str]) -> float:
    if not first or not second:
        return 1.0 if first == second else 0.0
    intersection = sum((first & second).values())
    union = sum((first | second).values())
    return intersection / union if union else 0.0


def entity_candidates(text: str) -> list[str]:
    candidates = re.findall(r"(?<![.!?]\s)\b[А-ЯЁA-Z][а-яёa-z]{2,}\b", text)
    unusual = [
        word
        for word, count in Counter(WORD_RE.findall(text)).items()
        if count > 1 and len(word) > 7
    ]
    return candidates + unusual


def calculate_metrics(source: str, result: str) -> dict[str, Any]:
    source_words = _words(source)
    result_words = _words(result)
    source_set, result_set = set(source_words), set(result_words)
    overlap = len(source_set & result_set) / max(1, len(source_set | result_set))
    result_sentences = sentences(result)
    incomplete = sum(1 for value in result_sentences if value[-1:] not in ".!?…")
    repeated = Counter(pairwise(result_words))
    new_numbers = sorted(set(re.findall(r"\b\d+(?:[.,]\d+)?\b", result)) - set(re.findall(r"\b\d+(?:[.,]\d+)?\b", source)))
    source_entities = set(entity_candidates(source))
    result_entities = Counter(entity_candidates(result))
    new_entities = sorted(set(result_entities) - source_entities)
    return {
        "disclaimer": "Diagnostic heuristics, not an objective measure of literary quality.",
        "structural": {
            "length_ratio": len(result) / max(1, len(source)),
            "source_characters": len(source),
            "result_characters": len(result),
            "paragraphs": len([part for part in re.split(r"\n\s*\n", result) if part.strip()]),
            "sentences": len(result_sentences),
            "dialogue_lines": sum(
                1 for line in result.splitlines() if line.lstrip().startswith(("—", "-", "–"))
            ),
            "quote_marks": sum(result.count(char) for char in "\"«»“”"),
            "dialogue_dashes": result.count("—") + result.count("–"),
            "incomplete_sentence_ratio": incomplete / max(1, len(result_sentences)),
        },
        "lexical": {
            "token_overlap": overlap,
            "character_4gram_similarity": _counter_similarity(_ngrams(source, 4), _ngrams(result, 4)),
            "new_word_ratio": len(result_set - source_set) / max(1, len(result_set)),
            "repeated_bigrams": [
                {"text": " ".join(pair), "count": count}
                for pair, count in repeated.most_common(10)
                if count > 1
            ],
            "case_change_ratio": sum(1 for token in tokens(result) if token.isupper())
            / max(1, len(tokens(result))),
            "punctuation_ratio": len(re.findall(r"[^\w\s]", result)) / max(1, len(result)),
        },
        "entities": {
            "frequent_variants": result_entities.most_common(20),
            "new_candidates": new_entities[:20],
        },
        "possible_generative_expansion": {
            "length_growth": max(0.0, len(result) / max(1, len(source)) - 1),
            "new_sentences": max(0, len(result_sentences) - len(sentences(source))),
            "new_entity_candidates": new_entities[:20],
            "new_numbers": new_numbers,
            "unmatched_span_ratio": 1.0 - SequenceMatcher(None, source, result).ratio(),
            "caveat": "This heuristic is not proof of hallucination.",
        },
        "semantic": {
            "available": False,
            "reason": "No embedding provider configured.",
        },
    }
