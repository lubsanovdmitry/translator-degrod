from __future__ import annotations

from semantic_telephone.chunking import assemble_chunks, chunk_text
from semantic_telephone.models import ChunkingConfig, ContextConfig
from semantic_telephone.stages.context import rolling_context


def test_chunking_preserves_paragraph_content(sample_text: str) -> None:
    chunks = chunk_text(
        sample_text,
        ChunkingConfig(strategy="target_chars", target_chars=65, max_chars=100),
    )
    rebuilt = assemble_chunks([chunk.source_text for chunk in chunks]).strip()
    assert rebuilt == sample_text


def test_overlap_is_context_only_and_not_duplicated(sample_text: str) -> None:
    chunks = chunk_text(
        sample_text,
        ChunkingConfig(
            strategy="target_chars", target_chars=65, max_chars=100, paragraph_overlap=1
        ),
    )
    assert chunks[1].context_prefix
    final = assemble_chunks([chunk.source_text for chunk in chunks])
    assert final.count("Первый абзац") == 1
    assert chunks[1].context_prefix not in chunks[1].source_text


def test_all_chunking_strategies(sample_text: str) -> None:
    for strategy in ("paragraph", "target_chars", "target_tokens", "sentence_window"):
        chunks = chunk_text(
            sample_text,
            ChunkingConfig(
                strategy=strategy,
                target_chars=65,
                max_chars=100,
                target_tokens=8,
                sentence_window=1,
            ),
        )
        assert chunks
        assert all(chunk.checksum and chunk.source_text for chunk in chunks)


def test_zero_context_window_selects_nothing() -> None:
    context = rolling_context(
        ["first", "second"],
        ContextConfig(enabled=True, previous_chunks=0),
    )
    assert context == ""
