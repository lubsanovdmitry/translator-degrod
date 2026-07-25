from __future__ import annotations

from pathlib import Path

import pytest

from semantic_telephone.memory import (
    InvalidMemoryResponse,
    MemoryStore,
    validate_memory_payload,
)


def test_memory_decay_and_repetition(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", half_life=10)
    payload = (
        '{"observations":[{"entity_key":"variant:x","text":"Имя изменилось",'
        '"confidence":0.6}]}'
    )
    store.ingest_json(payload, 1, provenance="damaged")
    initial = store.weight(store.items[0], 1)
    store.ingest_json(payload, 2, provenance="damaged")
    repeated = store.weight(store.items[0], 2)
    late = store.weight(store.items[0], 30)
    assert repeated > initial
    assert late < repeated


def test_memory_rejects_original_provenance(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    with pytest.raises(ValueError, match="damaged"):
        store.ingest_json('{"observations":[]}', 1, provenance="original")


def test_memory_rejects_invalid_json(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    with pytest.raises(InvalidMemoryResponse):
        store.ingest_json("not json", 1, provenance="damaged")


def test_memory_rejects_invalid_observation_schema() -> None:
    with pytest.raises(InvalidMemoryResponse, match="confidence"):
        validate_memory_payload(
            '{"observations":[{"entity_key":"variant:x","text":"seen"}]}'
        )


def test_memory_clear(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    store.ingest_json('{"observations":[]}', 1, provenance="damaged")
    store.clear()
    assert store.items == []
