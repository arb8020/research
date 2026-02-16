"""Radix tree cache for prefix sharing.

A radix tree stores sequences as paths from root to leaves.
Common prefixes share nodes, enabling:
- Prefix caching: reuse K,V for shared prompts
- Memory efficiency: don't duplicate shared prefixes

Example:
    "The cat sat" and "The cat ran" share "The cat " prefix.
    K,V for "The cat " is computed once and reused.

Structure:
    root
    └── "The cat " [slots: 0-7]
        ├── "sat" [slots: 8-10]
        └── "ran" [slots: 11-13]

State lives in caller-provided dict to keep this module stateless.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class RadixNode:
    """A node in the radix tree.

    Each node represents a sequence of tokens.
    Children are keyed by the first token of their sequence.
    """

    tokens: Tensor  # [num_tokens], int32
    slots: Tensor  # [num_tokens], int32
    ref_count: int = 0
    timestamp: float = field(default_factory=time.monotonic)
    parent: RadixNode | None = None
    children: dict[int, RadixNode] = field(default_factory=dict)
    _uid: int = field(default_factory=lambda: RadixNode._next_uid())
    _counter: int = 0

    @staticmethod
    def _next_uid() -> int:
        RadixNode._counter += 1
        return RadixNode._counter

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_protected(self) -> bool:
        return self.ref_count > 0

    def first_token(self) -> int:
        return int(self.tokens[0].item())

    def __lt__(self, other: RadixNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class CacheHandle:
    """Handle to a cached prefix.

    Returned by match_prefix, used to track which prefix a request uses.
    """

    node: RadixNode
    cached_len: int


def init_radix_state(state: dict, device: torch.device) -> None:
    """Initialize radix cache state.

    Args:
        state: Caller-provided state dict
        device: CUDA device
    """
    root = RadixNode(
        tokens=torch.tensor([], dtype=torch.int32, device=device),
        slots=torch.tensor([], dtype=torch.int32, device=device),
        ref_count=1,  # Root is always protected
    )
    state["root"] = root
    state["device"] = device
    state["evictable_tokens"] = 0
    state["protected_tokens"] = 0


def match_prefix(state: dict, tokens: Tensor) -> tuple[CacheHandle, Tensor]:
    """Find longest cached prefix for a token sequence.

    Args:
        state: Radix state dict
        tokens: Token IDs to match, shape [num_tokens]

    Returns:
        (handle, slot_indices)
        - handle: CacheHandle for the matched prefix
        - slot_indices: KV cache slots for matched tokens
    """
    root: RadixNode = state["root"]
    device: torch.device = state["device"]

    node = root
    matched_len = 0
    slots_list: list[Tensor] = []

    while matched_len < len(tokens):
        next_token = int(tokens[matched_len].item())

        if next_token not in node.children:
            break

        child = node.children[next_token]

        child_len = child.num_tokens
        remaining = len(tokens) - matched_len
        compare_len = min(child_len, remaining)

        match_len = _compare_tokens(
            child.tokens[:compare_len],
            tokens[matched_len : matched_len + compare_len],
        )

        if match_len == 0:
            break

        if match_len < child_len:
            child = _split_node(child, match_len)

        slots_list.append(child.slots[:match_len])
        matched_len += match_len
        node = child

        node.timestamp = time.monotonic()

    if matched_len == 0:
        return CacheHandle(root, 0), torch.tensor([], dtype=torch.int32, device=device)

    slots = (
        torch.cat(slots_list) if slots_list else torch.tensor([], dtype=torch.int32, device=device)
    )
    return CacheHandle(node, matched_len), slots


def lock(state: dict, handle: CacheHandle) -> None:
    """Increment reference count for a prefix (protect from eviction)."""
    node = handle.node
    while not node.is_root:
        if node.ref_count == 0:
            state["evictable_tokens"] -= node.num_tokens
            state["protected_tokens"] += node.num_tokens
        node.ref_count += 1
        node = node.parent


def unlock(state: dict, handle: CacheHandle) -> None:
    """Decrement reference count for a prefix."""
    node = handle.node
    while not node.is_root:
        node.ref_count -= 1
        if node.ref_count == 0:
            state["evictable_tokens"] += node.num_tokens
            state["protected_tokens"] -= node.num_tokens
        node = node.parent


def insert_prefix(state: dict, tokens: Tensor, slots: Tensor) -> int:
    """Insert a completed sequence into the cache.

    Args:
        state: Radix state dict
        tokens: Full token sequence
        slots: Corresponding KV cache slots

    Returns:
        Number of newly cached tokens (excluding already-cached prefix)
    """
    assert len(tokens) == len(slots)

    handle, _ = match_prefix(state, tokens)
    cached_len = handle.cached_len

    if cached_len == len(tokens):
        return 0

    new_tokens = tokens[cached_len:]
    new_slots = slots[cached_len:]

    node = handle.node
    new_node = RadixNode(
        tokens=new_tokens.clone(),
        slots=new_slots.clone(),
        parent=node,
    )
    node.children[new_node.first_token()] = new_node

    state["evictable_tokens"] += len(new_tokens)
    return len(new_tokens)


def evict(state: dict, num_tokens: int) -> Tensor:
    """Evict tokens from cache to free space.

    Evicts LRU leaf nodes until num_tokens are freed.

    Args:
        state: Radix state dict
        num_tokens: Number of tokens to evict

    Returns:
        Slot indices that were evicted (can be reused)
    """
    device: torch.device = state["device"]

    if num_tokens <= 0:
        return torch.tensor([], dtype=torch.int32, device=device)

    assert num_tokens <= state["evictable_tokens"], (
        f"Cannot evict {num_tokens} tokens, only {state['evictable_tokens']} evictable"
    )

    leaves = _collect_evictable_leaves(state["root"])
    heapq.heapify(leaves)

    evicted_slots: list[Tensor] = []
    evicted_count = 0

    while evicted_count < num_tokens and leaves:
        node = heapq.heappop(leaves)

        evicted_slots.append(node.slots)
        evicted_count += node.num_tokens
        state["evictable_tokens"] -= node.num_tokens

        parent = node.parent
        del parent.children[node.first_token()]

        if parent.is_leaf and not parent.is_protected and not parent.is_root:
            heapq.heappush(leaves, parent)

    return (
        torch.cat(evicted_slots)
        if evicted_slots
        else torch.tensor([], dtype=torch.int32, device=device)
    )


def get_stats(state: dict) -> dict:
    """Get cache statistics."""
    return {
        "evictable_tokens": state["evictable_tokens"],
        "protected_tokens": state["protected_tokens"],
        "total_cached": state["evictable_tokens"] + state["protected_tokens"],
    }


def _compare_tokens(a: Tensor, b: Tensor) -> int:
    """Compare two token sequences, return length of matching prefix."""
    assert len(a) == len(b)
    matches = a == b
    if matches.all():
        return len(a)
    mismatch_idx = (~matches).nonzero(as_tuple=True)[0]
    return int(mismatch_idx[0].item()) if len(mismatch_idx) > 0 else len(a)


def _split_node(node: RadixNode, split_pos: int) -> RadixNode:
    """Split a node at position split_pos.

    Creates new node for prefix, moves original node to be its child.
    """
    assert 0 < split_pos < node.num_tokens

    prefix_node = RadixNode(
        tokens=node.tokens[:split_pos].clone(),
        slots=node.slots[:split_pos].clone(),
        ref_count=node.ref_count,
        timestamp=node.timestamp,
        parent=node.parent,
    )

    node.parent.children[prefix_node.first_token()] = prefix_node

    node.tokens = node.tokens[split_pos:]
    node.slots = node.slots[split_pos:]
    node.parent = prefix_node
    prefix_node.children[node.first_token()] = node

    return prefix_node


def _collect_evictable_leaves(root: RadixNode) -> list[RadixNode]:
    """Collect all evictable leaf nodes."""
    leaves: list[RadixNode] = []
    stack = [root]

    while stack:
        node = stack.pop()
        if node.is_leaf:
            if not node.is_protected and not node.is_root:
                leaves.append(node)
        else:
            stack.extend(node.children.values())

    return leaves
