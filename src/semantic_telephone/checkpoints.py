from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import StageResult
from .utils.files import atomic_write_json, atomic_write_text, read_json
from .utils.hashing import checksum_text


class CheckpointStore:
    def __init__(self, chunk_directory: Path) -> None:
        self.directory = chunk_directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def paths(self, stage_number: int, label: str) -> tuple[Path, Path]:
        stem = f"stage-{stage_number:02d}-{label.replace('_', '-')}"
        return self.directory / f"{stem}.json", self.directory / f"{stem}-output.txt"

    def load(
        self, stage_number: int, label: str, expected_checksum: str
    ) -> StageResult | None:
        metadata_path, output_path = self.paths(stage_number, label)
        if not metadata_path.exists() or not output_path.exists():
            return None
        data = read_json(metadata_path)
        output = output_path.read_text(encoding="utf-8")
        if (
            data.get("stage_checksum") != expected_checksum
            or data.get("output_checksum") != checksum_text(output)
            or data.get("checkpoint_reusable") is not True
        ):
            return None
        data["output_text"] = output
        try:
            return StageResult(**data)
        except TypeError:
            return None

    def save(self, stage_number: int, label: str, result: StageResult) -> tuple[Path, Path]:
        metadata_path, output_path = self.paths(stage_number, label)
        atomic_write_text(output_path, result.output_text)
        payload: dict[str, Any] = result.to_dict()
        atomic_write_json(metadata_path, payload)
        return metadata_path, output_path
