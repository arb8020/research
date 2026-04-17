"""Simplified port of SillyTavern's world-info (lorebook) activation.

Ported from SillyTavern/public/scripts/world-info.js (`checkWorldInfo`, ~L4579).

This covers the ~80% common path: constant-on entries, primary-key substring
activation over a sliding scan window, token budget eviction by priority, and
before/after positioning. Good enough to exercise long-context workloads on
real lorebooks without depending on SillyTavern's runtime.

TODO(fidelity): the fully faithful alternative is running ST headless with an
OpenAI-compatible interceptor. See README for why we chose the port for v0.

Skipped vs ST (inline noted where each matters):
- recursive activation (entry content re-scanning into further rounds)
- secondary keys + selectiveLogic (AND_ANY / AND_ALL / NOT_ANY / NOT_ALL)
- probability / useProbability (random activation)
- min-activations seeking (expand scan depth until N entries fire)
- timed effects (sticky / cooldown / delay)
- group scoring (inclusion groups, group_weight, group_override)
- position variants beyond before_char / after_char
- regex keys (keys starting with "/")
- character/tag filters (characterFilter)
- generation-type triggers
- @@activate / @@dont_activate decorators
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Position values from ST's world_info_position (L855 of world-info.js).
POSITION_BEFORE_CHAR = 0
POSITION_AFTER_CHAR = 1


@dataclass(frozen=True)
class Entry:
    uid: str
    keys: tuple[str, ...]
    content: str
    constant: bool
    disable: bool
    order: int  # higher = higher priority (matches ST's sortedEntries)
    position: int  # 0 = before char, 1 = after char; others dropped in v0
    comment: str = ""


@dataclass(frozen=True)
class ActivationResult:
    before: str  # joined content of entries with position=before_char
    after: str  # joined content of entries with position=after_char
    activated_uids: tuple[str, ...]
    skipped_for_budget: int
    approx_tokens_used: int


def parse_lorebook(raw: dict[str, Any]) -> list[Entry]:
    """Parse an ST world-info JSON into Entry objects.

    ST stores entries as a dict keyed by uid (string of int). Each entry has:
    uid, key (list[str]), content (str), constant (bool), disable (bool),
    order (int), position (int), comment (str), plus many fields we ignore.
    """
    raw_entries = raw.get("entries", {})
    if isinstance(raw_entries, dict):
        items = raw_entries.values()
    else:
        items = raw_entries

    out: list[Entry] = []
    for e in items:
        keys = e.get("key") or []
        if not isinstance(keys, list):
            keys = []
        out.append(
            Entry(
                uid=str(e.get("uid", "")),
                keys=tuple(k for k in keys if isinstance(k, str) and k.strip()),
                content=str(e.get("content", "") or ""),
                constant=bool(e.get("constant", False)),
                disable=bool(e.get("disable", False)),
                order=int(e.get("order", 100)),
                position=int(e.get("position", POSITION_BEFORE_CHAR)),
                comment=str(e.get("comment", "") or ""),
            )
        )
    return out


def approx_tokens(text: str) -> int:
    """Cheap token approximation. ST uses a real tokenizer; we use len//4.

    TODO(fidelity): swap for tiktoken or a model-specific tokenizer when
    budget accuracy matters. For v0 the error band is fine because entries
    are mostly prose of similar density.
    """
    return max(1, len(text) // 4)


def _match_key(text: str, key: str, case_sensitive: bool, whole_word: bool) -> bool:
    """Substring match with optional case sensitivity and whole-word boundary.

    Ported from WorldInfoBuffer.matchKeys. We do not implement regex keys
    (keys starting with "/") — those silently fail to match here.
    """
    if key.startswith("/"):
        return False  # TODO(fidelity): regex keys
    haystack = text if case_sensitive else text.lower()
    needle = key if case_sensitive else key.lower()
    if whole_word:
        pattern = r"\b" + re.escape(needle) + r"\b"
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.search(pattern, haystack, flags) is not None
    return needle in haystack


def activate(
    entries: list[Entry],
    scan_text: str,
    token_budget: int,
    *,
    case_sensitive: bool = False,
    whole_word: bool = False,
) -> ActivationResult:
    """Run one activation pass over `scan_text` against `entries`.

    `scan_text` should be the concatenation of the last `scan_depth` messages
    (ST default scan_depth = 2–4). Caller builds this; we just match.

    Budget eviction: entries are sorted by descending `order` (priority).
    We add content greedily until the budget is exceeded; remaining entries
    are dropped with a count reported in the result.
    """
    activated: list[Entry] = []
    for e in entries:
        if e.disable:
            continue
        if e.constant:
            activated.append(e)
            continue
        if not e.keys:
            continue
        if any(_match_key(scan_text, k, case_sensitive, whole_word) for k in e.keys):
            activated.append(e)

    # ST sorts by `order` (higher first) then by insertion order for ties.
    activated.sort(key=lambda x: (-x.order, x.uid))

    kept: list[Entry] = []
    used = 0
    skipped = 0
    for e in activated:
        cost = approx_tokens(e.content)
        if used + cost > token_budget:
            skipped += 1
            continue
        kept.append(e)
        used += cost

    before = "\n".join(e.content for e in kept if e.position == POSITION_BEFORE_CHAR)
    after = "\n".join(e.content for e in kept if e.position == POSITION_AFTER_CHAR)

    return ActivationResult(
        before=before,
        after=after,
        activated_uids=tuple(e.uid for e in kept),
        skipped_for_budget=skipped,
        approx_tokens_used=used,
    )


def substitute_macros(text: str, char_name: str, user_name: str) -> str:
    """Replace ST's core macros: {{char}} and {{user}}.

    Ported spirit from public/scripts/macros.js. We only handle the two
    macros that matter for prompt assembly; {{random:...}}, {{roll:...}},
    {{time}}, etc. are skipped.

    TODO(fidelity): full macro set from macros.js (~50 LOC to port cleanly).
    """
    return (
        text.replace("{{char}}", char_name)
        .replace("{{user}}", user_name)
        .replace("{{Char}}", char_name)
        .replace("{{User}}", user_name)
    )
