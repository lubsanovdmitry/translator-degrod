from __future__ import annotations

import json
import random
import re
from collections import Counter

from ..models import GenerationResult, TranslationResult
from ..utils.hashing import checksum_text


class MockTranslationProvider:
    """A deterministic degradation model intended for offline tests, not linguistic realism."""

    name = "mock"

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        *,
        seed: int | None = None,
    ) -> TranslationResult:
        if not text.strip():
            return TranslationResult("", self.name, "mock-translation", deterministic=True)
        rng = random.Random(f"{seed}:{source_language}:{target_language}:{checksum_text(text)}")
        words = re.findall(r"\w+|[^\w\s]|\s+", text, re.UNICODE)
        word_positions = [
            index for index, token in enumerate(words) if token.isalpha() and len(token) > 3
        ]
        if len(word_positions) >= 2 and rng.random() < 0.7:
            first = rng.choice(word_positions)
            nearby = [index for index in word_positions if 0 < index - first <= 6]
            if nearby:
                second = rng.choice(nearby)
                words[first], words[second] = words[second], words[first]
        pronouns = {"он", "она", "они", "he", "she", "they", "it"}
        if rng.random() < 0.45:
            candidates = [i for i, token in enumerate(words) if token.lower() in pronouns]
            if candidates:
                words.pop(rng.choice(candidates))
        if rng.random() < 0.5:
            substitutions = {
                "сказал": "ответил",
                "увидел": "заметил",
                "дорога": "путь",
                "house": "building",
                "said": "answered",
                "looked": "watched",
            }
            for index, token in enumerate(words):
                replacement = substitutions.get(token.lower())
                if replacement and rng.random() < 0.6:
                    words[index] = replacement.capitalize() if token.istitle() else replacement
        capitalized = [i for i, token in enumerate(words) if token.istitle() and len(token) > 3]
        if capitalized and rng.random() < 0.25:
            index = rng.choice(capitalized)
            words[index] = words[index].lower()
        result = "".join(words)
        if rng.random() < 0.35:
            result = re.sub(r"[,;:]", "", result, count=1)
        short = [sentence for sentence in re.split(r"(?<=[.!?])\s+", result) if 2 < len(sentence) < 45]
        if short and rng.random() < 0.18:
            phrase = rng.choice(short)
            result = result.rstrip() + " " + phrase
        return TranslationResult(
            text=result,
            provider=self.name,
            model="mock-translation",
            response_id=f"mock-{checksum_text(result)[:12]}",
            usage={"input_characters": len(text), "output_characters": len(result)},
            deterministic=True,
        )


class MockGenerationProvider:
    name = "mock"

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        seed: int | None = None,
        response_format: str = "text",
    ) -> GenerationResult:
        if response_format == "json":
            body = _damaged_body(prompt)
            variants = Counter(
                token
                for token in re.findall(r"\b[А-ЯЁA-Z][а-яёa-z]{2,}\b", body)
            )
            observations = [
                {
                    "entity_key": f"variant:{token.casefold()}",
                    "text": f"В повреждённом тексте повторяется вариант «{token}».",
                    "confidence": min(0.85, 0.35 + count * 0.1),
                }
                for token, count in sorted(variants.items())
                if count >= 2
            ][:5]
            text = json.dumps({"observations": observations}, ensure_ascii=False)
        else:
            body = _damaged_body(prompt)
            body = re.sub(r"\s+([,.;:!?])", r"\1", body)
            body = re.sub(r"([.!?])\s*([а-яёa-z])", lambda m: f"{m.group(1)} {m.group(2).upper()}", body)
            body = re.sub(r"\s{2,}", " ", body).strip()
            auxiliary = " ".join(
                [
                    _prompt_section(prompt, "ПРЕДЫДУЩИЙ ПОВРЕЖДЁННЫЙ КОНТЕКСТ:"),
                    _prompt_section(
                        prompt,
                        "НЕУВЕРЕННЫЕ АВТОМАТИЧЕСКИЕ НАБЛЮДЕНИЯ; ИХ МОЖНО ИГНОРИРОВАТЬ:",
                    ),
                ]
            )
            if _significant_words(body) & _significant_words(auxiliary):
                body = re.sub(r"[.!?]\s+(?=\S)", "; ", body, count=1)
            if body and body[-1] not in ".!?…":
                body += "."
            text = body
        return GenerationResult(
            text=text,
            provider=self.name,
            model="mock-generation",
            response_id=f"mock-{checksum_text(text)[:12]}",
            usage={"input_characters": len(prompt), "output_characters": len(text)},
            deterministic=True,
        )


def _damaged_body(prompt: str) -> str:
    marker = "<<<TEXT>>>"
    return prompt.rsplit(marker, 1)[-1].strip() if marker in prompt else prompt.strip()


def _prompt_section(prompt: str, label: str) -> str:
    if label not in prompt:
        return ""
    value = prompt.split(label, 1)[1]
    boundaries = [
        position
        for marker in (
            "\n\nПРЕДЫДУЩИЙ ПОВРЕЖДЁННЫЙ КОНТЕКСТ:",
            "\n\nНЕУВЕРЕННЫЕ АВТОМАТИЧЕСКИЕ НАБЛЮДЕНИЯ; ИХ МОЖНО ИГНОРИРОВАТЬ:",
            "\n\n<<<TEXT>>>",
        )
        if (position := value.find(marker)) >= 0
    ]
    return value[: min(boundaries)] if boundaries else value


def _significant_words(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[^\W\d_]+", value, re.UNICODE)
        if len(token) >= 4
    }
