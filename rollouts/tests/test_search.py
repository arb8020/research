#!/usr/bin/env python3
"""Tests for tree search infrastructure.

Tests the core search abstractions:
- SearchTree, SearchNode, VerifyResult data structures
- Tree operations (make_root, add_child, get_node, etc.)
- Selection functions (select_all_frontier, select_one_best, etc.)
- Pruning functions (beam_pruner, threshold_pruner, etc.)
- Full run_search loop with mock expansion

These are integration tests at the "cut point" level - testing the search
module as a coherent unit, not individual functions in isolation.
"""

import pytest
import trio

from rollouts.dtypes import (
    Actor,
    AgentState,
    Endpoint,
    Message,
    RunConfig,
    StopReason,
    Trajectory,
)
from rollouts.search import (
    SearchNode,
    SearchTree,
    VerifyResult,
    add_child,
    add_children,
    compose_pruners,
    get_best_terminal,
    get_node,
    get_path_to_node,
    get_terminal_nodes,
    has_terminal_node,
    make_beam_pruner,
    make_depth_pruner,
    make_root,
    make_threshold_pruner,
    run_search,
    select_all_frontier,
    select_one_best,
    select_one_deepest,
    select_one_random,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def make_test_state(stop: StopReason | None = None) -> AgentState:
    """Create a minimal AgentState for testing."""
    msg = Message(role="user", content="test")
    traj = Trajectory(messages=[msg])
    endpoint = Endpoint(provider="anthropic", model="claude-sonnet-4-20250514")
    actor = Actor(trajectory=traj, endpoint=endpoint)
    return AgentState(actor=actor, environment=None, stop=stop)


# =============================================================================
# Data Structure Tests
# =============================================================================


def test_verify_result_defaults():
    """VerifyResult should have sensible defaults."""
    vr = VerifyResult()
    assert vr.score is None
    assert vr.valid is True
    assert vr.terminal is False
    assert vr.feedback is None
    print("✓ VerifyResult defaults correct")


def test_verify_result_frozen():
    """VerifyResult should be immutable."""
    vr = VerifyResult(score=0.5)
    with pytest.raises(AttributeError):
        vr.score = 0.9  # type: ignore
    print("✓ VerifyResult is frozen")


def test_search_node_frozen():
    """SearchNode should be immutable."""
    state = make_test_state()
    node = SearchNode(state=state, node_id="0")
    with pytest.raises(AttributeError):
        node.score = 0.5  # type: ignore
    print("✓ SearchNode is frozen")


def test_search_tree_frozen():
    """SearchTree should be immutable."""
    tree = SearchTree(nodes=(), frontier=())
    with pytest.raises(AttributeError):
        tree.frontier = ("0",)  # type: ignore
    print("✓ SearchTree is frozen")


# =============================================================================
# Tree Operation Tests
# =============================================================================


def test_make_root():
    """make_root creates a tree with single root node."""
    state = make_test_state()
    tree = make_root(state)

    assert len(tree.nodes) == 1
    assert tree.frontier == ("0",)
    assert tree.nodes[0].node_id == "0"
    assert tree.nodes[0].depth == 0
    assert tree.nodes[0].parent_id is None
    print("✓ make_root works")


def test_get_node():
    """get_node retrieves node by ID."""
    state = make_test_state()
    tree = make_root(state)

    node = get_node(tree, "0")
    assert node.node_id == "0"
    print("✓ get_node works")


def test_get_node_not_found():
    """get_node raises ValueError for missing node."""
    state = make_test_state()
    tree = make_root(state)

    with pytest.raises(ValueError, match="Node not found"):
        get_node(tree, "nonexistent")
    print("✓ get_node raises on missing node")


def test_add_child():
    """add_child adds a single child to a node."""
    state = make_test_state()
    tree = make_root(state)

    tree2 = add_child(tree, "0", state, score=0.5)

    assert len(tree2.nodes) == 2
    child = get_node(tree2, "0.0")
    assert child.parent_id == "0"
    assert child.depth == 1
    assert child.score == 0.5
    print("✓ add_child works")


def test_add_child_removes_parent_from_frontier():
    """Adding a child removes parent from frontier."""
    state = make_test_state()
    tree = make_root(state)

    assert "0" in tree.frontier
    tree2 = add_child(tree, "0", state)

    assert "0" not in tree2.frontier
    assert "0.0" in tree2.frontier
    print("✓ add_child updates frontier correctly")


def test_add_child_terminal_not_in_frontier():
    """Terminal nodes (state.stop set) don't go in frontier."""
    state = make_test_state()
    terminal_state = make_test_state(stop=StopReason.TASK_COMPLETED)

    tree = make_root(state)
    tree2 = add_child(tree, "0", terminal_state)

    assert "0.0" not in tree2.frontier  # Terminal, so not expandable
    print("✓ Terminal nodes excluded from frontier")


def test_add_children():
    """add_children adds multiple children at once."""
    state = make_test_state()
    tree = make_root(state)

    states = [make_test_state() for _ in range(3)]
    scores = [0.3, 0.5, 0.7]
    tree2 = add_children(tree, "0", states, scores)

    assert len(tree2.nodes) == 4  # root + 3 children
    assert get_node(tree2, "0.0").score == 0.3
    assert get_node(tree2, "0.1").score == 0.5
    assert get_node(tree2, "0.2").score == 0.7
    print("✓ add_children works")


def test_has_terminal_node():
    """has_terminal_node detects terminal nodes."""
    state = make_test_state()
    tree = make_root(state)
    assert not has_terminal_node(tree)

    terminal_state = make_test_state(stop=StopReason.TASK_COMPLETED)
    tree2 = add_child(tree, "0", terminal_state)
    assert has_terminal_node(tree2)
    print("✓ has_terminal_node works")


def test_get_terminal_nodes_sorted():
    """get_terminal_nodes returns terminals sorted by score."""
    state = make_test_state()
    tree = make_root(state)

    # Add multiple terminal children with different scores
    t1 = make_test_state(stop=StopReason.TASK_COMPLETED)
    t2 = make_test_state(stop=StopReason.TASK_COMPLETED)
    t3 = make_test_state(stop=StopReason.TASK_COMPLETED)

    tree = add_child(tree, "0", t1, score=0.3)
    tree = add_child(tree, "0", t2, score=0.9)
    tree = add_child(tree, "0", t3, score=0.5)

    terminals = get_terminal_nodes(tree)
    assert len(terminals) == 3
    assert terminals[0].score == 0.9  # Best first
    assert terminals[1].score == 0.5
    assert terminals[2].score == 0.3
    print("✓ get_terminal_nodes sorted by score")


def test_get_best_terminal():
    """get_best_terminal returns highest-scoring terminal."""
    state = make_test_state()
    tree = make_root(state)

    t1 = make_test_state(stop=StopReason.TASK_COMPLETED)
    t2 = make_test_state(stop=StopReason.TASK_COMPLETED)

    tree = add_child(tree, "0", t1, score=0.3)
    tree = add_child(tree, "0", t2, score=0.9)

    best = get_best_terminal(tree)
    assert best is not None
    assert best.score == 0.9
    print("✓ get_best_terminal works")


def test_get_path_to_node():
    """get_path_to_node returns path from root."""
    state = make_test_state()
    tree = make_root(state)
    tree = add_child(tree, "0", state)
    tree = add_child(tree, "0.0", state)
    tree = add_child(tree, "0.0.0", state)

    path = get_path_to_node(tree, "0.0.0.0")
    assert len(path) == 4
    assert [n.node_id for n in path] == ["0", "0.0", "0.0.0", "0.0.0.0"]
    print("✓ get_path_to_node works")


# =============================================================================
# Selection Function Tests
# =============================================================================


def test_select_all_frontier():
    """select_all_frontier returns all frontier nodes."""
    state = make_test_state()
    tree = make_root(state)
    tree = add_child(tree, "0", state)
    tree = add_child(tree, "0", state)

    selected = select_all_frontier(tree)
    assert set(selected) == {"0.0", "0.1"}
    print("✓ select_all_frontier works")


def test_select_one_best():
    """select_one_best returns highest-scoring frontier node."""
    state = make_test_state()
    tree = make_root(state)
    tree = add_child(tree, "0", state, score=0.3)
    tree = add_child(tree, "0", state, score=0.9)
    tree = add_child(tree, "0", state, score=0.5)

    selected = select_one_best(tree)
    assert selected == ["0.1"]  # Score 0.9
    print("✓ select_one_best works")


def test_select_one_deepest():
    """select_one_deepest returns deepest frontier node."""
    state = make_test_state()
    tree = make_root(state)
    tree = add_child(tree, "0", state)  # depth 1
    tree = add_child(tree, "0.0", state)  # depth 2
    tree = add_child(tree, "0", state)  # depth 1

    selected = select_one_deepest(tree)
    assert selected == ["0.0.0"]  # Depth 2
    print("✓ select_one_deepest works")


def test_select_one_random():
    """select_one_random returns a frontier node."""
    state = make_test_state()
    tree = make_root(state)
    tree = add_child(tree, "0", state)
    tree = add_child(tree, "0", state)

    selected = select_one_random(tree)
    assert len(selected) == 1
    assert selected[0] in tree.frontier
    print("✓ select_one_random works")


def test_select_empty_frontier():
    """Selection functions handle empty frontier."""
    tree = SearchTree(nodes=(), frontier=())

    assert select_all_frontier(tree) == []
    assert select_one_best(tree) == []
    assert select_one_deepest(tree) == []
    assert select_one_random(tree) == []
    print("✓ Selection handles empty frontier")


# =============================================================================
# Pruning Function Tests
# =============================================================================


def test_beam_pruner():
    """Beam pruner keeps top-k nodes by score."""
    state = make_test_state()
    tree = make_root(state)
    tree = add_child(tree, "0", state, score=0.3)
    tree = add_child(tree, "0", state, score=0.9)
    tree = add_child(tree, "0", state, score=0.5)

    pruner = make_beam_pruner(beam_width=2)
    pruned = pruner(tree)

    assert len(pruned.frontier) == 2
    assert "0.1" in pruned.frontier  # 0.9
    assert "0.2" in pruned.frontier  # 0.5
    assert "0.0" not in pruned.frontier  # 0.3 dropped
    print("✓ Beam pruner works")


def test_beam_pruner_no_op_when_under_limit():
    """Beam pruner is no-op when frontier <= beam_width."""
    state = make_test_state()
    tree = make_root(state)
    tree = add_child(tree, "0", state)

    pruner = make_beam_pruner(beam_width=5)
    pruned = pruner(tree)

    assert pruned.frontier == tree.frontier
    print("✓ Beam pruner no-op when under limit")


def test_threshold_pruner():
    """Threshold pruner removes nodes below min_score."""
    state = make_test_state()
    tree = make_root(state)
    tree = add_child(tree, "0", state, score=0.3)
    tree = add_child(tree, "0", state, score=0.9)
    tree = add_child(tree, "0", state, score=0.5)

    pruner = make_threshold_pruner(min_score=0.4)
    pruned = pruner(tree)

    assert len(pruned.frontier) == 2
    assert "0.1" in pruned.frontier  # 0.9
    assert "0.2" in pruned.frontier  # 0.5
    assert "0.0" not in pruned.frontier  # 0.3 dropped
    print("✓ Threshold pruner works")


def test_depth_pruner():
    """Depth pruner removes nodes beyond max_depth."""
    state = make_test_state()
    tree = make_root(state)
    # Add multiple children at different depths to keep them all in frontier
    tree = add_child(tree, "0", state)  # 0.0, depth 1
    tree = add_child(tree, "0", state)  # 0.1, depth 1
    tree = add_child(tree, "0.0", state)  # 0.0.0, depth 2
    tree = add_child(tree, "0.1", state)  # 0.1.0, depth 2
    tree = add_child(tree, "0.0.0", state)  # 0.0.0.0, depth 3

    # Frontier has: 0.1.0 (depth 2), 0.0.0.0 (depth 3)
    pruner = make_depth_pruner(max_depth=2)
    pruned = pruner(tree)

    assert "0.1.0" in pruned.frontier  # depth 2, kept
    assert "0.0.0.0" not in pruned.frontier  # depth 3, dropped
    print("✓ Depth pruner works")


def test_compose_pruners():
    """Composed pruners apply in sequence."""
    state = make_test_state()
    tree = make_root(state)
    # Create structure where we have nodes at different depths in frontier
    tree = add_child(tree, "0", state, score=0.9)  # 0.0, depth 1
    tree = add_child(tree, "0", state, score=0.2)  # 0.1, depth 1, low score
    tree = add_child(tree, "0.0", state, score=0.8)  # 0.0.0, depth 2
    tree = add_child(tree, "0.0.0", state, score=0.7)  # 0.0.0.0, depth 3

    # Frontier now: 0.1 (depth 1, score 0.2), 0.0.0.0 (depth 3, score 0.7)

    # Compose: threshold 0.5, then depth 2
    pruner = compose_pruners(
        make_threshold_pruner(min_score=0.5),
        make_depth_pruner(max_depth=2),
    )
    pruned = pruner(tree)

    # 0.1 dropped (score 0.2 < 0.5)
    # 0.0.0.0 dropped (depth 3 > 2)
    assert "0.1" not in pruned.frontier  # Low score
    assert "0.0.0.0" not in pruned.frontier  # Too deep
    assert len(pruned.frontier) == 0  # Both pruned
    print("✓ Composed pruners work")


# =============================================================================
# Integration Test: run_search with mock expansion
# =============================================================================


async def noop_chunk_handler(chunk):
    """No-op chunk handler for tests."""
    pass


@pytest.mark.trio
async def test_run_search_linear():
    """Test run_search with linear expansion (no branching)."""
    print("\n=== Testing run_search (linear) ===\n")

    state = make_test_state()
    config = RunConfig(on_chunk=noop_chunk_handler)

    # Counter to track expansions
    expansion_count = 0

    async def mock_expand(tree, node_ids, config):
        """Mock expand that adds one child per node, terminates after 3 expansions."""
        nonlocal expansion_count
        expansion_count += 1

        for node_id in node_ids:
            if expansion_count >= 3:
                terminal = make_test_state(stop=StopReason.TASK_COMPLETED)
                tree = add_child(tree, node_id, terminal, score=1.0)
            else:
                new_state = make_test_state()
                tree = add_child(tree, node_id, new_state, score=0.5)
        return tree

    tree = await run_search(
        initial_state=state,
        config=config,
        select=select_all_frontier,
        expand=mock_expand,
        prune=None,
        max_steps=10,
    )

    assert has_terminal_node(tree)
    assert expansion_count == 3
    best = get_best_terminal(tree)
    assert best is not None
    assert best.score == 1.0
    print(f"✓ Linear search completed in {expansion_count} expansions")
    print(f"✓ Tree has {len(tree.nodes)} nodes")


@pytest.mark.trio
async def test_run_search_beam():
    """Test run_search with beam search (branching + pruning)."""
    print("\n=== Testing run_search (beam) ===\n")

    state = make_test_state()
    config = RunConfig(on_chunk=noop_chunk_handler)

    step = 0

    async def mock_beam_expand(tree, node_ids, config):
        """Mock expand that branches 2x per node."""
        nonlocal step
        step += 1

        for node_id in node_ids:
            if step >= 3:
                # Terminate on step 3
                t = make_test_state(stop=StopReason.TASK_COMPLETED)
                tree = add_child(tree, node_id, t, score=1.0)
            else:
                # Branch 2x with different scores
                s1 = make_test_state()
                s2 = make_test_state()
                tree = add_child(tree, node_id, s1, score=0.3 + step * 0.1)
                tree = add_child(tree, node_id, s2, score=0.5 + step * 0.1)
        return tree

    tree = await run_search(
        initial_state=state,
        config=config,
        select=select_all_frontier,
        expand=mock_beam_expand,
        prune=make_beam_pruner(beam_width=2),
        max_steps=10,
    )

    assert has_terminal_node(tree)
    print(f"✓ Beam search completed in {step} steps")
    print(f"✓ Tree has {len(tree.nodes)} nodes")
    print(f"✓ Final frontier size: {len(tree.frontier)}")


@pytest.mark.trio
async def test_run_search_stops_on_empty_frontier():
    """run_search stops when frontier is empty."""
    print("\n=== Testing run_search (empty frontier) ===\n")

    state = make_test_state()
    config = RunConfig(on_chunk=noop_chunk_handler)

    async def mock_expand_prunes_all(tree, node_ids, config):
        """Expand with low scores that will all be pruned."""
        for node_id in node_ids:
            s = make_test_state()
            tree = add_child(tree, node_id, s, score=0.1)
        return tree

    tree = await run_search(
        initial_state=state,
        config=config,
        select=select_all_frontier,
        expand=mock_expand_prunes_all,
        prune=make_threshold_pruner(min_score=0.5),  # Prunes everything
        max_steps=10,
    )

    # Should stop early because frontier becomes empty
    assert not has_terminal_node(tree)
    assert len(tree.frontier) == 0
    print("✓ Search stopped on empty frontier")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":

    async def main():
        print("\n" + "=" * 70)
        print("TESTS: Tree Search Infrastructure")
        print("=" * 70 + "\n")

        # Sync tests
        print("--- Data Structures ---")
        test_verify_result_defaults()
        test_verify_result_frozen()
        test_search_node_frozen()
        test_search_tree_frozen()

        print("\n--- Tree Operations ---")
        test_make_root()
        test_get_node()
        test_get_node_not_found()
        test_add_child()
        test_add_child_removes_parent_from_frontier()
        test_add_child_terminal_not_in_frontier()
        test_add_children()
        test_has_terminal_node()
        test_get_terminal_nodes_sorted()
        test_get_best_terminal()
        test_get_path_to_node()

        print("\n--- Selection Functions ---")
        test_select_all_frontier()
        test_select_one_best()
        test_select_one_deepest()
        test_select_one_random()
        test_select_empty_frontier()

        print("\n--- Pruning Functions ---")
        test_beam_pruner()
        test_beam_pruner_no_op_when_under_limit()
        test_threshold_pruner()
        test_depth_pruner()
        test_compose_pruners()

        # Async tests
        print("\n--- Integration Tests ---")
        await test_run_search_linear()
        await test_run_search_beam()
        await test_run_search_stops_on_empty_frontier()

        print("\n" + "=" * 70)
        print("✅ All search tests passed!")
        print("=" * 70 + "\n")

    trio.run(main)
