from __future__ import annotations

from ..models import ContextConfig


def rolling_context(previous: list[str], config: ContextConfig) -> str:
    if not config.enabled or config.previous_chunks <= 0 or config.max_chars <= 0:
        return ""
    selected = previous[-config.previous_chunks :]
    value = "\n\n".join(selected)
    if len(value) <= config.max_chars:
        return value
    if config.truncation == "head":
        return value[: config.max_chars]
    return value[-config.max_chars :]
