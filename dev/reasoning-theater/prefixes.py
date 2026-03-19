"""Prefix slicing helpers for reasoning-theater analyses."""

from __future__ import annotations

from typing import Sequence

from records import PrefixSlice


def normalize_fractions(fractions: Sequence[float]) -> list[float]:
    normalized = sorted(set(round(float(f), 6) for f in fractions))
    if not normalized:
        raise ValueError("At least one prefix fraction is required")
    for fraction in normalized:
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"Prefix fraction must be in (0, 1], got {fraction}")
    return normalized


def build_prefix_slices(
    *,
    reasoning_text: str,
    prefix_fractions: Sequence[float],
    completion_token_ids: Sequence[int] | None = None,
    min_chars: int = 0,
) -> list[PrefixSlice]:
    """Build prefix slices from token counts when possible, else from text length."""
    fractions = normalize_fractions(prefix_fractions)
    text_length = len(reasoning_text)
    token_count = len(completion_token_ids) if completion_token_ids is not None else None

    prefixes: list[PrefixSlice] = []
    for fraction in fractions:
        char_stop = None
        if text_length > 0:
            char_stop = max(min_chars, int(round(text_length * fraction)))
            char_stop = min(text_length, char_stop)
        token_stop = None
        if token_count:
            token_stop = max(1, int(round(token_count * fraction)))
            token_stop = min(token_count, token_stop)
        label = f"p{int(round(fraction * 100)):02d}"
        prefixes.append(
            PrefixSlice(
                label=label,
                fraction=fraction,
                char_stop=char_stop,
                token_stop=token_stop,
                text_prefix=reasoning_text[:char_stop] if char_stop is not None else reasoning_text,
            )
        )
    return prefixes
