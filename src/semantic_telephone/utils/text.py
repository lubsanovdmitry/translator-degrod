from __future__ import annotations

import re

SENTENCE_RE = re.compile(r"(?<=[.!?…])(?:[\"»”)]*)\s+")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text.strip()) if part.strip()]


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)

