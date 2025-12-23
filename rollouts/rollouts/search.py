"""Tree search for agent rollouts.

Core abstraction: select → expand → prune

Supports beam search (Kevin-32B style), MCTS, best-first, and other search strategies
by composing different select/expand/prune functions.

Example (Kevin-32B beam search):
    tree = await run_search(
        initial_state,
        config,
        select=select_all_frontier,
        expand=make_expand_n_turns(n=4, branch_factor=4),
        prune=make_beam_pruner(beam_width=4),
        max_steps=8,
    )

Example (MCTS-style):
    tree = await run_search(
        initial_state,
        config,
        select=make_ucb_selector(c=1.414),
        expand=make_expand_n_turns(n=1, branch_factor=1),
        max_steps=1000,
    )
"""

from __future__ import annotations

import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import trio

if TYPE_CHECKING:
    from .dtypes import AgentState, RunConfig

from .agents import run_agent_step

# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class VerifyResult:
    """Result of verifying an agent state.

    Returned by Environment.verify() to score and validate states.
    """

    score: float | None = None  # How promising is this state (higher = better)
    valid: bool = True  # Is this a legal state (False = don't add to tree)
    terminal: bool = False  # Is this a solution (True = stop search)
    feedback: str | None = None  # Optional message to inject into trajectory


@dataclass(frozen=True)
class SearchNode:
    """Immutable snapshot of a node in the search tree.

    Wraps AgentState with tree metadata (position, score, pruned status).
    """

    state: AgentState
    node_id: str = ""  # e.g., "0.1.2" (path from root)
    parent_id: str | None = None  # None for root
    depth: int = 0
    score: float | None = None  # From verification
    pruned: bool = False


@dataclass(frozen=True)
class SearchTree:
    """Immutable search tree.

    All operations return new trees (functional style).
    """

    nodes: tuple[SearchNode, ...] = ()  # All nodes (root at index 0)
    frontier: tuple[str, ...] = ()  # Node IDs available for expansion


# =============================================================================
# Type Aliases
# =============================================================================

SelectFn = Callable[["SearchTree"], list[str]]
"""Select function: returns node IDs to expand."""

ExpandFn = Callable[["SearchTree", list[str], "RunConfig"], Awaitable["SearchTree"]]
"""Expand function: expands selected nodes, returns new tree."""

PruneFn = Callable[["SearchTree"], "SearchTree"]
"""Prune function: trims frontier, returns new tree."""


# =============================================================================
# Tree Operations
# =============================================================================


def make_root(state: AgentState) -> SearchTree:
    """Create tree with single root node."""
    root = SearchNode(state=state, node_id="0", depth=0)
    return SearchTree(nodes=(root,), frontier=("0",))


def get_node(tree: SearchTree, node_id: str) -> SearchNode:
    """Get node by ID."""
    for node in tree.nodes:
        if node.node_id == node_id:
            return node
    raise ValueError(f"Node not found: {node_id}")


def add_child(
    tree: SearchTree,
    parent_id: str,
    child_state: AgentState,
    score: float | None = None,
) -> SearchTree:
    """Add a single child to a node. Returns new tree."""
    parent = get_node(tree, parent_id)

    # Generate child ID
    existing_children = [n for n in tree.nodes if n.parent_id == parent_id]
    child_idx = len(existing_children)
    child_id = f"{parent_id}.{child_idx}"

    child = SearchNode(
        state=child_state,
        parent_id=parent_id,
        node_id=child_id,
        depth=parent.depth + 1,
        score=score,
    )

    # Remove parent from frontier, add child if not terminal
    new_frontier = tuple(fid for fid in tree.frontier if fid != parent_id)
    if not child_state.stop:
        new_frontier = new_frontier + (child_id,)

    return SearchTree(
        nodes=tree.nodes + (child,),
        frontier=new_frontier,
    )


def add_children(
    tree: SearchTree,
    parent_id: str,
    child_states: list[AgentState],
    scores: list[float | None],
) -> SearchTree:
    """Add multiple children to a node. Returns new tree."""
    for state, score in zip(child_states, scores, strict=False):
        tree = add_child(tree, parent_id, state, score)
    return tree


def has_terminal_node(tree: SearchTree) -> bool:
    """Check if tree contains a terminal (solved) node."""
    for node in tree.nodes:
        if node.state.stop and not node.pruned:
            return True
    return False


def get_terminal_nodes(tree: SearchTree) -> list[SearchNode]:
    """Get all terminal nodes, sorted by score (best first)."""
    terminals = [n for n in tree.nodes if n.state.stop and not n.pruned]
    return sorted(terminals, key=lambda n: n.score or 0, reverse=True)


def get_best_terminal(tree: SearchTree) -> SearchNode | None:
    """Get highest-scoring terminal node."""
    terminals = get_terminal_nodes(tree)
    return terminals[0] if terminals else None


def get_path_to_node(tree: SearchTree, node_id: str) -> list[SearchNode]:
    """Get path from root to node."""
    path = []
    current_id: str | None = node_id

    while current_id is not None:
        node = get_node(tree, current_id)
        path.append(node)
        current_id = node.parent_id

    return list(reversed(path))


# =============================================================================
# Select Functions
# =============================================================================


def select_all_frontier(tree: SearchTree) -> list[str]:
    """Select all nodes in frontier. For beam search."""
    return list(tree.frontier)


def select_one_best(tree: SearchTree) -> list[str]:
    """Select single highest-scoring node. For best-first search."""
    if not tree.frontier:
        return []
    frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
    best = max(frontier_nodes, key=lambda n: n.score or 0)
    return [best.node_id]


def select_one_random(tree: SearchTree) -> list[str]:
    """Select single random node from frontier."""
    if not tree.frontier:
        return []
    return [random.choice(tree.frontier)]


def select_one_deepest(tree: SearchTree) -> list[str]:
    """Select deepest node (depth-first). For linear search."""
    if not tree.frontier:
        return []
    frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
    deepest = max(frontier_nodes, key=lambda n: n.depth)
    return [deepest.node_id]


def make_ucb_selector(c: float = 1.414) -> SelectFn:
    """Create UCB selection for MCTS. Returns single node.

    Args:
        c: Exploration constant (higher = more exploration)
    """
    visit_counts: dict[str, int] = {}

    def select(tree: SearchTree) -> list[str]:
        if not tree.frontier:
            return []

        total_visits = sum(visit_counts.get(nid, 0) for nid in tree.frontier) + 1

        def ucb_score(node: SearchNode) -> float:
            visits = visit_counts.get(node.node_id, 0) + 1
            exploit = node.score or 0
            explore = c * math.sqrt(math.log(total_visits) / visits)
            return exploit + explore

        frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
        best = max(frontier_nodes, key=ucb_score)
        visit_counts[best.node_id] = visit_counts.get(best.node_id, 0) + 1
        return [best.node_id]

    return select


# =============================================================================
# Expand Functions
# =============================================================================


def make_expand_n_turns(
    n: int = 1,
    branch_factor: int = 1,
    parallel: bool = True,
) -> ExpandFn:
    """Create an expand function that runs N turns per expansion.

    Args:
        n: Number of agent turns before creating a child node
        branch_factor: Number of children to create per selected node
                      (each child forks from the same state after N turns)
        parallel: Whether to expand nodes in parallel (default True)
    """

    async def expand_single_node(
        node: SearchNode,
        config: RunConfig,
    ) -> tuple[str, list[tuple[AgentState, VerifyResult]]]:
        """Expand one node, return (parent_id, [(state, verify_result), ...])."""
        state = node.state
        results: list[tuple[AgentState, VerifyResult]] = []

        # Run agent for N turns to get base state
        for turn in range(n):
            state = await run_agent_step(state, config)
            if state.stop:
                break

        # Verify the state
        if state.environment and hasattr(state.environment, "verify"):
            vr = await state.environment.verify(state)
        else:
            vr = VerifyResult()

        # Create branch_factor children from this state
        for b in range(branch_factor):
            if b == 0:
                # First child uses the state directly
                results.append((state, vr))
            else:
                # Fork environment for additional branches
                if state.environment:
                    env_data = await state.environment.serialize()
                    forked_env = await state.environment.__class__.deserialize(env_data)
                    forked_state = replace(state, environment=forked_env)
                else:
                    forked_state = state
                results.append((forked_state, vr))

        return node.node_id, results

    async def expand(
        tree: SearchTree,
        node_ids: list[str],
        config: RunConfig,
    ) -> SearchTree:
        """Expand selected nodes, return new tree."""
        if not node_ids:
            return tree

        nodes = [get_node(tree, nid) for nid in node_ids]

        if parallel and len(nodes) > 1:
            # Expand all nodes in parallel
            results: list[tuple[str, list[tuple[AgentState, VerifyResult]]]] = []

            async def expand_and_collect(node: SearchNode) -> None:
                result = await expand_single_node(node, config)
                results.append(result)

            async with trio.open_nursery() as nursery:
                for node in nodes:
                    nursery.start_soon(expand_and_collect, node)
        else:
            # Sequential expansion
            results = []
            for node in nodes:
                result = await expand_single_node(node, config)
                results.append(result)

        # Add all children to tree
        for parent_id, children in results:
            for state, vr in children:
                if vr.valid:
                    tree = add_child(tree, parent_id, state, vr.score)

        return tree

    return expand


# =============================================================================
# Prune Functions
# =============================================================================


def make_beam_pruner(beam_width: int) -> PruneFn:
    """Keep only top-k nodes by score."""

    def prune(tree: SearchTree) -> SearchTree:
        if len(tree.frontier) <= beam_width:
            return tree

        frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
        sorted_nodes = sorted(frontier_nodes, key=lambda n: n.score or 0, reverse=True)
        keep_ids = {n.node_id for n in sorted_nodes[:beam_width]}

        new_frontier = tuple(fid for fid in tree.frontier if fid in keep_ids)
        return SearchTree(nodes=tree.nodes, frontier=new_frontier)

    return prune


def make_threshold_pruner(min_score: float) -> PruneFn:
    """Remove nodes below score threshold."""

    def prune(tree: SearchTree) -> SearchTree:
        frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
        keep_ids = {n.node_id for n in frontier_nodes if (n.score or 0) >= min_score}

        new_frontier = tuple(fid for fid in tree.frontier if fid in keep_ids)
        return SearchTree(nodes=tree.nodes, frontier=new_frontier)

    return prune


def make_depth_pruner(max_depth: int) -> PruneFn:
    """Remove nodes beyond max depth."""

    def prune(tree: SearchTree) -> SearchTree:
        frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
        keep_ids = {n.node_id for n in frontier_nodes if n.depth <= max_depth}

        new_frontier = tuple(fid for fid in tree.frontier if fid in keep_ids)
        return SearchTree(nodes=tree.nodes, frontier=new_frontier)

    return prune


def compose_pruners(*pruners: PruneFn) -> PruneFn:
    """Apply multiple pruning functions in sequence."""

    def prune(tree: SearchTree) -> SearchTree:
        for pruner in pruners:
            tree = pruner(tree)
        return tree

    return prune


# =============================================================================
# Core Search Loop
# =============================================================================


async def run_search(
    initial_state: AgentState,
    config: RunConfig,
    select: SelectFn,
    expand: ExpandFn,
    prune: PruneFn | None = None,
    max_steps: int = 100,
) -> SearchTree:
    """Run tree search until solution found or limits reached.

    The core loop is: select → expand → prune

    Args:
        initial_state: Starting agent state
        config: Run configuration (unchanged from linear run_agent)
        select: Function to pick which nodes to expand
        expand: Function to expand selected nodes (may run multiple turns)
        prune: Optional function to prune frontier after expansion
        max_steps: Safety limit on search iterations

    Returns:
        Final search tree containing all explored nodes
    """
    tree = make_root(initial_state)

    for step in range(max_steps):
        # Select nodes to expand
        node_ids = select(tree)
        if not node_ids:
            break

        # Expand selected nodes
        tree = await expand(tree, node_ids, config)

        # Check for solution
        if has_terminal_node(tree):
            break

        # Apply pruning
        if prune:
            tree = prune(tree)

        # Check if frontier is empty after pruning
        if not tree.frontier:
            break

    return tree
