from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_text() -> str:
    return (
        "Первый абзац описывает тихую мастерскую и старые часы.\n\n"
        "Второй абзац добавляет коробку, письмо и осторожный вопрос.\n\n"
        "Третий абзац завершает сцену коротким решением."
    )


@pytest.fixture
def config_file(tmp_path: Path, sample_text: str) -> Path:
    (tmp_path / "input.txt").write_text(sample_text, encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        """run:
  name: test-run
  seed: 42
  source_language: ru
  target_language: ru
  output_root: runs
input: {path: input.txt}
chunking:
  strategy: target_chars
  target_chars: 65
  max_chars: 100
  paragraph_overlap: 1
translation:
  provider: mock
  route_mode: random
  min_hops: 3
  max_hops: 4
generation: {provider: mock, model: mock}
context: {enabled: false}
memory: {enabled: false}
pipeline:
  - type: translation_cycle
    hops: {min: 3, max: 4}
  - type: reconstruction
  - type: final_translation
runtime:
  concurrency: 1
  retries: 2
  retry_backoff_seconds: 0
""",
        encoding="utf-8",
    )
    return config
