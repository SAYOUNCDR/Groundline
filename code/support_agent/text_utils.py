from __future__ import annotations

import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]*", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    lowered = (text or "").lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def first_sentences(text: str, limit: int = 3, max_chars: int = 700) -> str:
    clean = normalize_space(
        re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text or "").replace("#", "")
    )
    if len(clean) <= max_chars:
        return clean

    pieces = re.split(r"(?<=[.!?])\s+", clean)
    selected: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        candidate = " ".join([*selected, piece])
        if len(selected) >= limit or len(candidate) > max_chars:
            break
        selected.append(piece)
    return " ".join(selected).strip() or clean[:max_chars].rstrip()
