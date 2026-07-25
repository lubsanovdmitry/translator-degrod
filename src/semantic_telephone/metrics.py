from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from itertools import pairwise
from typing import Any

from .utils.text import sentences, tokens

WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)


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
