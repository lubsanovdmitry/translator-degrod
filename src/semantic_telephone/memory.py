from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .utils.files import append_jsonl, atomic_write_json, read_json
from .utils.hashing import checksum_text


class InvalidMemoryResponse(ValueError):
    pass


def validate_memory_payload(raw: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise InvalidMemoryResponse(f"invalid memory JSON: {error}") from error
    observations = data.get("observations") if isinstance(data, dict) else None
    if not isinstance(observations, list):
        raise InvalidMemoryResponse("memory JSON must contain an observations list")
    validated: list[dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise InvalidMemoryResponse("each observation must be an object")
        key = observation.get("entity_key")
        text = observation.get("text")
        confidence = observation.get("confidence")
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(text, str)
            or not text.strip()
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise InvalidMemoryResponse("invalid entity_key, text, or confidence")
        validated.append(observation)
    return validated


class MemoryStore:
    def __init__(
        self,
        directory: Path,
        *,
        half_life: float = 20.0,
        load_items: bool = False,
    ) -> None:
        self.directory = directory
        self.state_path = directory / "state.json"
        self.events_path = directory / "observations.jsonl"
        self.half_life = half_life
        self.items: list[dict[str, Any]] = []
        self._recorded_event_ids: set[str] = set()
        self._applied_event_ids: set[str] = set()
        if self.state_path.exists():
            data = read_json(self.state_path)
            raw_event_ids = data.get("applied_event_ids", [])
            if isinstance(raw_event_ids, list):
                self._recorded_event_ids = {
                    str(event_id) for event_id in raw_event_ids if isinstance(event_id, str)
                }
            if load_items:
                raw_items = data.get("items", [])
                if isinstance(raw_items, list):
                    self.items = [item for item in raw_items if isinstance(item, dict)]
                self._applied_event_ids = set(self._recorded_event_ids)

    def ingest_json(
        self,
        raw: str,
        chunk_index: int,
        *,
        provenance: str,
        event_id: str | None = None,
    ) -> None:
        if provenance != "damaged":
            raise ValueError("automatic memory may only be populated from damaged results")
        observations = validate_memory_payload(raw)
        resolved_event_id = event_id or (
            f"manual-{chunk_index}-{checksum_text(raw)[:16]}-{len(self._applied_event_ids)}"
        )
        if resolved_event_id in self._applied_event_ids:
            return
        for observation in observations:
            key = str(observation["entity_key"])
            text = str(observation["text"])
            confidence = float(observation["confidence"])
            self._add(key, text, confidence, chunk_index)
            if resolved_event_id not in self._recorded_event_ids:
                append_jsonl(
                    self.events_path,
                    {
                        "event_id": resolved_event_id,
                        "chunk": chunk_index,
                        "entity_key": key,
                        "text": text,
                        "confidence": confidence,
                        "provenance": provenance,
                    },
                )
        self._applied_event_ids.add(resolved_event_id)
        self._recorded_event_ids.add(resolved_event_id)
        self.save()

    def _add(self, key: str, text: str, confidence: float, chunk_index: int) -> None:
        for item in self.items:
            if item["entity_key"] == key and item["text"] == text:
                item["count"] += 1
                item["confidence"] = (item["confidence"] + confidence) / 2
                item["last_seen_chunk"] = chunk_index
                return
        self.items.append(
            {
                "entity_key": key,
                "text": text,
                "count": 1,
                "confidence": confidence,
                "first_seen_chunk": chunk_index,
                "last_seen_chunk": chunk_index,
            }
        )

    def weight(self, item: dict[str, Any], current_chunk: int) -> float:
        age = max(0, current_chunk - int(item["last_seen_chunk"]))
        return (
            float(item["confidence"])
            * math.log1p(int(item["count"]))
            * math.exp(-age / self.half_life)
        )

    def prompt_items(self, current_chunk: int, *, maximum: int, minimum_count: int) -> str:
        candidates = [item for item in self.items if int(item["count"]) >= minimum_count]
        candidates.sort(key=lambda item: self.weight(item, current_chunk), reverse=True)
        return "\n".join(
            f"- {item['text']} (confidence={item['confidence']:.2f}, "
            f"weight={self.weight(item, current_chunk):.3f})"
            for item in candidates[:maximum]
        )

    def save(self) -> None:
        atomic_write_json(
            self.state_path,
            {
                "heuristic": "confidence * log(1 + count) * exp(-age / half_life)",
                "half_life": self.half_life,
                "items": self.items,
                "applied_event_ids": sorted(self._recorded_event_ids),
            },
        )

    def clear(self) -> None:
        self.items = []
        self._recorded_event_ids = set()
        self._applied_event_ids = set()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.save()
        if self.events_path.exists():
            self.events_path.unlink()
