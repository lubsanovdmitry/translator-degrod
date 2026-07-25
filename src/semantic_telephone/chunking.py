from __future__ import annotations

import re

from .models import Chunk, ChunkingConfig
from .utils.hashing import checksum_text
from .utils.text import sentences, tokens


def _paragraphs_with_offsets(text: str) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", text, re.DOTALL):
        result.append((match.group(0), match.start(), match.end()))
    return result


def _split_oversized(value: str, limit: int) -> list[str]:
    if len(value) <= limit:
        return [value]
    parts: list[str] = []
    remaining = value
    while len(remaining) > limit:
        boundary = remaining.rfind(" ", 0, limit + 1)
        boundary = boundary if boundary > limit // 2 else limit
        parts.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def chunk_text(text: str, config: ChunkingConfig) -> list[Chunk]:
    if not text.strip():
        return []
    if config.strategy == "paragraph":
        groups = [[item] for item in _paragraphs_with_offsets(text)]
    elif config.strategy in {"target_chars", "target_tokens"}:
        groups = _size_groups(text, config)
    elif config.strategy == "sentence_window":
        groups = _sentence_groups(text, config)
    else:
        raise ValueError(f"unknown chunking strategy: {config.strategy}")

    chunks: list[Chunk] = []
    previous_texts: list[str] = []
    for index, group in enumerate(groups):
        source = "\n\n".join(item[0] for item in group)
        start = group[0][1]
        end = group[-1][2]
        context = (
            "\n\n".join(previous_texts[-config.paragraph_overlap :])
            if config.paragraph_overlap > 0
            else ""
        )
        chunk_id = f"{index + 1:04d}-{checksum_text(source)[:10]}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                index=index,
                source_text=source,
                char_start=start,
                char_end=end,
                paragraphs=list(range(index, index + len(group))),
                context_prefix=context,
                checksum=checksum_text(source),
            )
        )
        previous_texts.extend(item[0] for item in group)
    return chunks


def _size_groups(text: str, config: ChunkingConfig) -> list[list[tuple[str, int, int]]]:
    raw = _paragraphs_with_offsets(text)
    expanded: list[tuple[str, int, int]] = []
    for paragraph, start, end in raw:
        parts = _split_oversized(paragraph, config.max_chars)
        cursor = start
        for part in parts:
            position = text.find(part, cursor, end + 1)
            position = cursor if position < 0 else position
            expanded.append((part, position, position + len(part)))
            cursor = position + len(part)
    groups: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    for item in expanded:
        candidate = "\n\n".join([*(part[0] for part in current), item[0]])
        measure = len(tokens(candidate)) if config.strategy == "target_tokens" else len(candidate)
        target = config.target_tokens if config.strategy == "target_tokens" else config.target_chars
        if current and (measure > target or len(candidate) > config.max_chars):
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def _sentence_groups(text: str, config: ChunkingConfig) -> list[list[tuple[str, int, int]]]:
    values = sentences(text)
    groups: list[list[tuple[str, int, int]]] = []
    cursor = 0
    for position in range(0, len(values), config.sentence_window):
        group: list[tuple[str, int, int]] = []
        for value in values[position : position + config.sentence_window]:
            start = text.find(value, cursor)
            if start < 0:
                start = cursor
            end = start + len(value)
            group.append((value, start, end))
            cursor = end
        groups.append(group)
    return groups


def assemble_chunks(values: list[str]) -> str:
    """Join only transformed chunk bodies; context prefixes are deliberately excluded."""
    return "\n\n".join(value.strip() for value in values if value.strip()) + "\n"
