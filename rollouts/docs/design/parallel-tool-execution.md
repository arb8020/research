# Parallel Tool Execution

**Status:** Proposal  
**Author:** @chiraagbalu  
**Date:** 2024-12-25

## Summary

Add conditional parallel execution for tool calls based on a `concurrency_safe` property, matching Claude Code's approach. Read-only tools execute in parallel; mutating tools execute sequentially.

## Motivation

Currently, all tool calls execute sequentially:

```
Agent returns: [read(a.py), read(b.py), read(c.py), write(d.py)]
Timeline:      |--read a--|--read b--|--read c--|--write d--|
               0ms       100ms      200ms      300ms       400ms
```

With parallel execution for safe tools:

```
Timeline:      |--read a--|
               |--read b--|--write d--|
               |--read c--|
               0ms       100ms       200ms
```

**~2x speedup** for common patterns like "read multiple files then edit."

## Design

### 1. Tool Metadata

Add `concurrency_safe` to tool definitions:

```python
@dataclass(frozen=True)
class ToolFunction:
    name: str
    description: str
    parameters: ToolParameters
    concurrency_safe: bool = False  # New field
```

### 2. Default Classification

| Tool | `concurrency_safe` | Rationale |
|------|-------------------|-----------|
| `Read` | `True` | Read-only |
| `Glob` | `True` | Read-only |
| `Grep` | `True` | Read-only |
| `LS` | `True` | Read-only |
| `Write` | `False` | Mutates filesystem |
| `Edit` | `False` | Mutates filesystem |
| `Bash` | `False` | Unknown side effects |
| `MultiEdit` | `False` | Mutates filesystem |
| MCP tools | `False` (default) | Unknown, can override via annotation |

### 3. Execution Algorithm

```python
async def process_pending_tools(state: AgentState, rcfg: RunConfig) -> AgentState:
    tools = state.pending_tool_calls
    results = []
    i = 0
    
    while i < len(tools):
        # Collect consecutive concurrency-safe tools
        batch = []
        while i < len(tools) and is_concurrency_safe(tools[i], state.environment):
            batch.append(tools[i])
            i += 1
        
        # Execute batch in parallel
        if batch:
            batch_results = await execute_parallel(batch, state, rcfg)
            results.extend(batch_results)
        
        # Execute single unsafe tool (if any remain)
        if i < len(tools):
            result = await execute_single(tools[i], state, rcfg)
            results.append(result)
            i += 1
    
    return apply_results(state, results)
```

### 4. Interrupt Handling

On interrupt (Escape key):
1. Cancel all in-flight tool executions via `cancel_scope`
2. Completed tools keep their results
3. In-progress tools get `[interrupted]` synthetic results
4. Pending tools get `[skipped - interrupted]` results

```python
except trio.Cancelled:
    for tool in batch:
        if tool.id in completed:
            # Keep result
            pass
        elif tool.id in in_progress:
            results.append(ToolResult(
                tool_call_id=tool.id,
                content="[interrupted]",
                is_error=True
            ))
        else:
            results.append(ToolResult(
                tool_call_id=tool.id, 
                content="[skipped - interrupted]",
                is_error=True
            ))
```

### 5. Result Ordering

Results must be returned **in original order** regardless of completion order:

```python
async def execute_parallel(batch, state, rcfg):
    results = [None] * len(batch)
    
    async def run_one(idx, tool_call):
        results[idx] = await execute_tool(tool_call, state, rcfg)
    
    async with trio.open_nursery() as nursery:
        for idx, tc in enumerate(batch):
            nursery.start_soon(run_one, idx, tc)
    
    return results  # Ordered by original position
```

## Examples

### Example 1: All Safe Tools

```
Agent: [read(a), read(b), read(c)]

Execution:
  t=0:   start read(a), read(b), read(c) in parallel
  t=100: all complete
  
Results: [result_a, result_b, result_c] (in order)
```

### Example 2: Mixed Tools

```
Agent: [read(a), read(b), write(c), read(d)]

Execution:
  t=0:   start read(a), read(b) in parallel
  t=100: both complete, start write(c) alone
  t=200: write complete, start read(d) alone
  t=300: done

Results: [result_a, result_b, result_c, result_d] (in order)
```

### Example 3: Interrupt During Parallel Batch

```
Agent: [read(a), read(b), read(c), write(d)]

Execution:
  t=0:   start read(a), read(b), read(c) in parallel
  t=50:  USER PRESSES ESCAPE
         read(a) completed, read(b) in-progress, read(c) pending
  
Results: 
  - read(a): actual result
  - read(b): [interrupted]  
  - read(c): [skipped - interrupted]
  - write(d): [skipped - interrupted]
```

## Migration

1. Add `concurrency_safe=False` default to all existing tools (no behavior change)
2. Audit read-only tools and flip to `concurrency_safe=True`
3. Add parallel execution logic behind feature flag
4. Benchmark and validate ordering
5. Enable by default

## Alternatives Considered

### A. Full Parallel (No Safety Check)

Execute all tools in parallel regardless of type.

**Rejected:** Race conditions when multiple tools write to same file.

### B. User-Configured Parallelism

Let users specify which tools can run in parallel.

**Rejected:** Too complex, easy to misconfigure. Tool authors know best.

### C. Dependency Graph

Analyze tool arguments to build dependency graph.

**Rejected:** Over-engineered. Simple safe/unsafe classification covers 95% of cases.

## Open Questions

1. **MCP tool classification:** Should MCP servers declare concurrency safety in their tool manifests? Claude Code defaults to unsafe.

2. **Bash command analysis:** Could we parse bash commands to detect read-only operations (e.g., `cat`, `ls`, `grep`)? Probably not worth the complexity.

3. **Max parallelism:** Should we cap concurrent tools? Claude Code doesn't appear to. Start unbounded, add limit if needed.

## References

- Claude Code source (minified): `~/.bun/install/global/node_modules/@anthropic-ai/claude-code/cli.js`
- pi-mono agent loop: `/tmp/pi-mono/packages/ai/src/agent/agent-loop.ts` (sequential only)
- Current rollouts implementation: `~/research/rollouts/rollouts/agents.py`
