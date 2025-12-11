# Tree Search Design for Rollouts

> **Status:** Draft proposal
> **Goal:** Support AlphaProof-style branching/backtracking without overfit to linear PPO/GRPO paradigm

---

## Problem Statement

Current RL frameworks (prime-rl, pufferlib, verifiers) assume linear trajectories:

```
generate → score → update
```

This works for:
- Standard RL (Atari, games, control)
- LLM fine-tuning (SFT, GRPO, PPO)
- Simple tool-use agents

But **cannot express**:
- Tree search (explore multiple paths, backtrack on failures)
- Subagent spawning (specialized reasoners for subtasks)
- Step-level verification (Lean compiler validates each proof step)
- Non-linear control flow (if approach A fails, try approach B)

AlphaProof, AlphaCode, and similar systems need the search tree as a **first-class structure**, not hidden inside action selection.

---

## Design Principles

From `CLASSES_VS_FUNCTIONAL.md`:

1. **Functions orchestrate, classes only for legitimate state**
2. **Frozen dataclasses for data/config**
3. **Pure functions for computation/transformation**
4. **Explicit inputs/outputs, no hidden state**

Applied to tree search:
- `SearchTree` and `SearchNode` are frozen dataclasses (immutable data)
- Selection and pruning are pure functions, not strategy classes
- `run_search` is a pure function that orchestrates expansion
- Existing `AgentState` unchanged - tree structure is separate

---

## Data Structures

### VerifyResult

Output from environment's step-level verification:

```python
@dataclass(frozen=True)
class VerifyResult:
    """Result of verifying an agent state."""
    score: float | None = None   # How promising is this state (0-1)
    valid: bool = True           # Is this a legal state (False = prune)
    terminal: bool = False       # Is this a solution (True = stop search)
    feedback: str | None = None  # Optional message to inject into trajectory
```

### SearchNode

A node in the search tree. Wraps `AgentState` with tree metadata:

```python
@dataclass(frozen=True)
class SearchNode:
    """Immutable snapshot of a node in the search tree."""
    state: AgentState
    node_id: str = ""              # e.g., "0.1.2" (path from root)
    parent_id: str | None = None   # None for root
    depth: int = 0
    score: float | None = None     # From verification
    pruned: bool = False
```

**Why separate from AgentState?**
- `AgentState` is the agent's view (trajectory, environment, tools)
- `SearchNode` is the search algorithm's view (tree position, score, pruned)
- Keeps concerns separated, `AgentState` unchanged

### SearchTree

The full tree structure. Immutable - operations return new trees:

```python
@dataclass(frozen=True)
class SearchTree:
    """Immutable search tree."""
    nodes: tuple[SearchNode, ...]  # All nodes (root at index 0)
    frontier: tuple[str, ...]      # Node IDs available for expansion
```

**Why tuple not list?**
- Frozen dataclass requires immutable fields
- Forces functional style (return new tree, don't mutate)

---

## Environment Protocol Extension

Add optional `verify` method:

```python
class Environment(Protocol):
    # Existing methods unchanged
    def get_tools(self) -> list[Tool]: ...
    async def exec_tool(self, tool_call, state, config, cancel_scope) -> ToolResult: ...
    async def on_assistant_message(self, message, state) -> AgentState: ...

    # New: step-level verification (optional)
    async def verify(self, state: AgentState) -> VerifyResult:
        """Verify current state is valid/promising.

        Override for step-level verification (e.g., Lean proof checker).
        Default: always valid, no score, not terminal.
        """
        return VerifyResult()
```

**Examples:**

```python
# Lean proof environment
class LeanEnvironment:
    async def verify(self, state: AgentState) -> VerifyResult:
        # Extract proof state from trajectory
        proof_code = extract_lean_code(state.actor.trajectory)

        # Check with Lean compiler
        result = await self.lean_client.check(proof_code)

        if result.has_errors:
            return VerifyResult(valid=False)  # Prune this branch

        if result.goals_remaining == 0:
            return VerifyResult(terminal=True, score=1.0)  # Solved!

        # Score by progress (fewer goals = better)
        score = 1.0 / (1 + result.goals_remaining)
        return VerifyResult(score=score)

# Code execution environment
class CodeEnvironment:
    async def verify(self, state: AgentState) -> VerifyResult:
        code = extract_code(state.actor.trajectory)

        # Run tests
        passed, total = await run_tests(code, self.test_cases)

        if passed == total:
            return VerifyResult(terminal=True, score=1.0)

        return VerifyResult(score=passed / total)
```

---

## Pure Functions: Tree Operations

### Tree Construction

```python
def make_root(state: AgentState) -> SearchTree:
    """Create tree with single root node."""
    root = SearchNode(state=state, node_id="0", depth=0)
    return SearchTree(nodes=(root,), frontier=("0",))


def add_children(
    tree: SearchTree,
    parent_id: str,
    child_states: list[AgentState],
    scores: list[float | None],
) -> SearchTree:
    """Add children to a node. Returns new tree."""
    parent = next(n for n in tree.nodes if n.node_id == parent_id)

    new_nodes = list(tree.nodes)
    new_frontier = [fid for fid in tree.frontier if fid != parent_id]

    for i, (state, score) in enumerate(zip(child_states, scores)):
        node_id = f"{parent_id}.{i}"
        child = SearchNode(
            state=state,
            parent_id=parent_id,
            node_id=node_id,
            depth=parent.depth + 1,
            score=score,
        )
        new_nodes.append(child)
        if not state.stop:
            new_frontier.append(node_id)

    return SearchTree(nodes=tuple(new_nodes), frontier=tuple(new_frontier))


def prune_node(tree: SearchTree, node_id: str) -> SearchTree:
    """Mark node as pruned, remove from frontier. Returns new tree."""
    new_nodes = tuple(
        replace(n, pruned=True) if n.node_id == node_id else n
        for n in tree.nodes
    )
    new_frontier = tuple(fid for fid in tree.frontier if fid != node_id)
    return SearchTree(nodes=new_nodes, frontier=new_frontier)
```

### Selection Functions

Type alias for selection:

```python
SelectFn = Callable[[SearchTree], str | None]
```

Built-in strategies as pure functions:

```python
def select_best_first(tree: SearchTree) -> str | None:
    """Select highest-scoring node from frontier."""
    if not tree.frontier:
        return None
    frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
    return max(frontier_nodes, key=lambda n: n.score or 0).node_id


def select_depth_first(tree: SearchTree) -> str | None:
    """Select deepest node (most recent)."""
    if not tree.frontier:
        return None
    frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
    return max(frontier_nodes, key=lambda n: n.depth).node_id


def select_breadth_first(tree: SearchTree) -> str | None:
    """Select shallowest node (oldest)."""
    if not tree.frontier:
        return None
    frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
    return min(frontier_nodes, key=lambda n: n.depth).node_id


def select_random(tree: SearchTree) -> str | None:
    """Select random node from frontier."""
    if not tree.frontier:
        return None
    import random
    return random.choice(tree.frontier)
```

Custom selection (e.g., UCB for MCTS):

```python
def make_ucb_selector(c: float = 1.414) -> SelectFn:
    """Create UCB selection function for MCTS."""
    visit_counts: dict[str, int] = {}  # Closure captures state

    def select_ucb(tree: SearchTree) -> str | None:
        if not tree.frontier:
            return None

        total_visits = sum(visit_counts.get(nid, 0) for nid in tree.frontier)

        def ucb_score(node: SearchNode) -> float:
            visits = visit_counts.get(node.node_id, 1)
            exploit = node.score or 0
            explore = c * math.sqrt(math.log(total_visits + 1) / visits)
            return exploit + explore

        frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
        best = max(frontier_nodes, key=ucb_score)
        visit_counts[best.node_id] = visit_counts.get(best.node_id, 0) + 1
        return best.node_id

    return select_ucb
```

### Pruning Functions

Type alias for pruning:

```python
PruneFn = Callable[[SearchTree], SearchTree]
```

Built-in strategies:

```python
def make_beam_pruner(beam_width: int) -> PruneFn:
    """Keep only top-k nodes by score."""
    def prune_beam(tree: SearchTree) -> SearchTree:
        if len(tree.frontier) <= beam_width:
            return tree

        frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
        sorted_nodes = sorted(frontier_nodes, key=lambda n: n.score or 0, reverse=True)
        keep_ids = {n.node_id for n in sorted_nodes[:beam_width]}

        new_frontier = tuple(fid for fid in tree.frontier if fid in keep_ids)
        return SearchTree(nodes=tree.nodes, frontier=new_frontier)

    return prune_beam


def make_threshold_pruner(min_score: float) -> PruneFn:
    """Remove nodes below score threshold."""
    def prune_threshold(tree: SearchTree) -> SearchTree:
        frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
        keep_ids = {n.node_id for n in frontier_nodes if (n.score or 0) >= min_score}

        new_frontier = tuple(fid for fid in tree.frontier if fid in keep_ids)
        return SearchTree(nodes=tree.nodes, frontier=new_frontier)

    return prune_threshold


def make_depth_pruner(max_depth: int) -> PruneFn:
    """Remove nodes beyond max depth."""
    def prune_depth(tree: SearchTree) -> SearchTree:
        frontier_nodes = [n for n in tree.nodes if n.node_id in tree.frontier]
        keep_ids = {n.node_id for n in frontier_nodes if n.depth <= max_depth}

        new_frontier = tuple(fid for fid in tree.frontier if fid in keep_ids)
        return SearchTree(nodes=tree.nodes, frontier=new_frontier)

    return prune_depth


def compose_pruners(*pruners: PruneFn) -> PruneFn:
    """Apply multiple pruning functions in sequence."""
    def composed(tree: SearchTree) -> SearchTree:
        for pruner in pruners:
            tree = pruner(tree)
        return tree
    return composed
```

---

## Core Search Loop

```python
async def expand_node(
    tree: SearchTree,
    node_id: str,
    config: RunConfig,
    branch_factor: int = 1,
) -> tuple[list[AgentState], list[VerifyResult]]:
    """Expand a single node. Returns child states and verification results.

    Pure function - doesn't mutate tree.
    """
    node = next(n for n in tree.nodes if n.node_id == node_id)
    state = node.state

    child_states = []
    verify_results = []

    # Generate branch_factor children from same state
    for _ in range(branch_factor):
        # Use existing run_agent_step (unchanged)
        next_state = await run_agent_step(state, config)
        child_states.append(next_state)

        # Verify if environment supports it
        if state.environment and hasattr(state.environment, 'verify'):
            result = await state.environment.verify(next_state)
        else:
            result = VerifyResult()  # Default: valid, no score
        verify_results.append(result)

    return child_states, verify_results


async def run_search(
    initial_state: AgentState,
    config: RunConfig,
    select: SelectFn = select_best_first,
    prune: PruneFn | None = None,
    branch_factor: int = 1,
    max_expansions: int = 100,
) -> SearchTree:
    """Run tree search until solution found or limits reached.

    Args:
        initial_state: Starting agent state
        config: Run configuration (unchanged from linear run_agent)
        select: Function to pick next node to expand
        prune: Optional function to prune frontier after expansion
        branch_factor: Children per expansion (1 = linear search)
        max_expansions: Safety limit on total expansions

    Returns:
        Final search tree containing all explored nodes
    """
    tree = make_root(initial_state)
    expansions = 0

    while tree.frontier and expansions < max_expansions:
        # Select node to expand
        node_id = select(tree)
        if node_id is None:
            break

        # Expand: generate children
        child_states, verify_results = await expand_node(
            tree, node_id, config, branch_factor
        )

        # Filter by verification
        valid_children = []
        scores = []

        for state, vr in zip(child_states, verify_results):
            if not vr.valid:
                continue  # Prune invalid states

            # Inject feedback if provided
            if vr.feedback:
                feedback_msg = Message(role="user", content=vr.feedback)
                new_trajectory = replace(
                    state.actor.trajectory,
                    messages=state.actor.trajectory.messages + [feedback_msg]
                )
                state = replace(state, actor=replace(state.actor, trajectory=new_trajectory))

            # Mark terminal states
            if vr.terminal:
                state = replace(state, stop=StopReason.TASK_COMPLETED)

            valid_children.append(state)
            scores.append(vr.score)

        # Add children to tree
        tree = add_children(tree, node_id, valid_children, scores)

        # Check for solution
        for state in valid_children:
            if state.stop == StopReason.TASK_COMPLETED:
                return tree  # Found solution

        # Apply pruning
        if prune:
            tree = prune(tree)

        expansions += 1

    return tree
```

---

## Result Extraction

```python
def get_solution(tree: SearchTree) -> AgentState | None:
    """Get the solution state if one was found."""
    for node in tree.nodes:
        if node.state.stop == StopReason.TASK_COMPLETED and not node.pruned:
            return node.state
    return None


def get_best_terminal(tree: SearchTree) -> AgentState | None:
    """Get highest-scoring terminal state (solution or stopped)."""
    terminals = [
        n for n in tree.nodes
        if n.state.stop is not None and not n.pruned
    ]
    if not terminals:
        return None
    return max(terminals, key=lambda n: n.score or 0).state


def get_all_terminals(tree: SearchTree) -> list[AgentState]:
    """Get all terminal states, sorted by score."""
    terminals = [
        n for n in tree.nodes
        if n.state.stop is not None and not n.pruned
    ]
    sorted_terminals = sorted(terminals, key=lambda n: n.score or 0, reverse=True)
    return [n.state for n in sorted_terminals]


def get_path_to_node(tree: SearchTree, node_id: str) -> list[SearchNode]:
    """Get path from root to node."""
    path = []
    current_id: str | None = node_id

    while current_id is not None:
        node = next(n for n in tree.nodes if n.node_id == current_id)
        path.append(node)
        current_id = node.parent_id

    return list(reversed(path))
```

---

## Usage Examples

### Linear Search (Current Behavior)

```python
# Equivalent to current run_agent
tree = await run_search(
    initial_state,
    config,
    select=select_depth_first,
    branch_factor=1,
)
result = get_best_terminal(tree)
```

### Beam Search

```python
tree = await run_search(
    initial_state,
    config,
    select=select_best_first,
    prune=make_beam_pruner(beam_width=5),
    branch_factor=3,
)
solution = get_solution(tree)
```

### Best-First with Threshold

```python
tree = await run_search(
    initial_state,
    config,
    select=select_best_first,
    prune=compose_pruners(
        make_threshold_pruner(min_score=0.3),
        make_depth_pruner(max_depth=10),
    ),
    branch_factor=2,
)
```

### MCTS-style

```python
tree = await run_search(
    initial_state,
    config,
    select=make_ucb_selector(c=1.414),
    branch_factor=1,  # Expand one at a time
    max_expansions=1000,
)
```

---

## Integration with Existing Code

### What Changes

1. **New file:** `rollouts/search.py` with data structures and functions
2. **Environment protocol:** Add optional `verify()` method (default returns valid)
3. **New StopReason:** `TASK_COMPLETED` (already exists)

### What Stays the Same

- `AgentState` - unchanged
- `Actor`, `Trajectory`, `Message` - unchanged
- `run_agent_step` - unchanged (called by `expand_node`)
- `run_agent` - unchanged (linear execution still works)
- `Environment.exec_tool`, `on_assistant_message` - unchanged
- Streaming events - unchanged
- Session persistence - works per-node if needed

### Migration Path

1. **Phase 1:** Add `search.py` with types and functions
2. **Phase 2:** Add `Environment.verify()` with default implementation
3. **Phase 3:** Implement `LeanEnvironment.verify()` as proof of concept
4. **Phase 4:** Add parallel expansion (batch LLM calls across branches)

---

## Future Extensions

### Parallel Expansion

Batch LLM calls across multiple nodes:

```python
async def expand_nodes_parallel(
    tree: SearchTree,
    node_ids: list[str],
    config: RunConfig,
    branch_factor: int = 1,
) -> list[tuple[list[AgentState], list[VerifyResult]]]:
    """Expand multiple nodes in parallel."""
    async with trio.open_nursery() as nursery:
        results = []
        for node_id in node_ids:
            # Launch concurrent expansions
            ...
    return results
```

### Subagent Spawning

Specialized solvers for subtasks:

```python
@dataclass(frozen=True)
class SubagentResult:
    state: AgentState
    solved: bool

async def spawn_subagent(
    state: AgentState,
    subtask: str,
    subagent_config: RunConfig,
) -> SubagentResult:
    """Spawn specialized agent for subtask."""
    # Create focused state with subtask prompt
    # Run to completion
    # Return result
    ...
```

### Credit Assignment

Track which branches contributed to solution:

```python
def compute_branch_credits(
    tree: SearchTree,
    solution_id: str,
) -> dict[str, float]:
    """Assign credit to nodes on path to solution."""
    path = get_path_to_node(tree, solution_id)
    credits = {}
    for i, node in enumerate(path):
        # More credit to later nodes (actually found solution)
        credits[node.node_id] = (i + 1) / len(path)
    return credits
```

---

## Open Questions

1. **KV cache sharing:** Branches share prefix - can we reuse KV cache?
2. **Memory limits:** How to handle large trees? LRU eviction of low-score branches?
3. **Batching:** How to batch LLM calls across branches with different contexts?
4. **Checkpointing:** Save/restore search tree for long-running searches?
5. **Visualization:** How to render tree structure for debugging?

---

## Summary

| Component | Type | Purpose |
|-----------|------|---------|
| `VerifyResult` | frozen dataclass | Verification output |
| `SearchNode` | frozen dataclass | Node in tree (wraps AgentState) |
| `SearchTree` | frozen dataclass | Immutable tree structure |
| `SelectFn` | type alias | Function to pick next node |
| `PruneFn` | type alias | Function to prune frontier |
| `select_*` | pure functions | Built-in selection strategies |
| `make_*_pruner` | pure functions | Built-in pruning strategies |
| `expand_node` | async function | Generate children for one node |
| `run_search` | async function | Main search loop |
| `get_*` | pure functions | Extract results from tree |

**Key principle:** The tree is data, not behavior. Selection and pruning are functions, not classes. `run_search` orchestrates stateful operations (`run_agent_step`) but is itself a pure function given the same inputs.
