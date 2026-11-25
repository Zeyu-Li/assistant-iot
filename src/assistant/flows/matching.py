"""Fuzzy matching of spoken names against saved ones ("the milk" vs "milk", "parked" vs "parking")."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, Optional

_STOPWORDS = frozenset(
    "a an the my our your is are was were be of for to in on at and or did do does i me we you it "
    "what where when which who how that this some any".split()
)
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def key_words(text: str) -> frozenset[str]:
    """The content words of ``text``, stemmed."""
    return frozenset(_stem(w) for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOPWORDS)


def similarity(a: str, b: str) -> float:
    """0..1: the better of word overlap (order-free) and character similarity (typos, plurals)."""
    words_a, words_b = key_words(a), key_words(b)
    overlap = len(words_a & words_b) / len(words_a | words_b) if words_a and words_b else 0.0
    characters = SequenceMatcher(None, " ".join(sorted(words_a)), " ".join(sorted(words_b))).ratio()
    return max(overlap, characters)


def best_match(query: str, candidates: Iterable[str], threshold: float) -> Optional[str]:
    """The candidate most similar to ``query``, if it scores at least ``threshold``."""
    scored = [(similarity(query, c), c) for c in candidates]
    if not scored:
        return None
    score, candidate = max(scored)
    return candidate if score >= threshold else None
