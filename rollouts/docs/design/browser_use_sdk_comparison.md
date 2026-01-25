# Browser-Use Agent SDK Comparison

**Claude Code Session ID:** `540796a2-5736-4923-9382-660984488141`
**Date:** 2026-01-16

## Summary

Compared `browser-use/agent-sdk` (cloned to `/tmp/browser-use-agent-sdk`) with rollouts' agent implementation to identify patterns worth adopting.

## Key Learnings from Browser-Use

### Worth Adopting

1. **`@tool` decorator** - Auto-generates `Tool` schema from function signature + type hints
2. **Ephemeral messages** - `@tool("Get state", ephemeral=3)` auto-removes old outputs from context
3. **Automatic context compaction** - Summarizes history when approaching context limit
4. **`TaskComplete` exception** - Tools can explicitly signal completion

### The `Depends()` Question

Browser-Use uses FastAPI-style dependency injection:
```python
@tool("Execute command")
async def bash(
    command: str,
    ctx: Annotated[SandboxContext, Depends(get_sandbox)],
) -> str:
```

**Problem:** Rollouts uses serialize-execute-deserialize per tool for checkpointing:
```python
env_data = await env.serialize()
fresh_env = await env.__class__.deserialize(env_data)
result = await fresh_env.exec_tool(tool_call, ...)
env_data = await fresh_env.serialize()  # Capture changes
```

Dependencies fall into two categories:
1. **Derivable from serialized state** - works fine with serialize/deserialize
2. **External/runtime** (DB connections, API clients) - can't be serialized

### Possible Hybrid Design

```python
@dataclass
class FromEnv:
    """Dependency resolved from env.serialize() output"""
    key: str

@tool("Execute command")
async def bash(
    command: str,
    working_dir: Annotated[Path, FromEnv("working_dir")],  # From env_data
    logger: Annotated[Logger, Runtime(get_logger)],        # Runtime injection
) -> str:
```

But this gets awkward for state updates (auto-commit after write) - who owns that logic?

### Recommendation

Keep tools as methods on `Environment` (they need access to stateful things like `_commit_count`, `_auto_commit()`). Add `@env_tool` decorator for ergonomics:

```python
class GitWorktreeEnvironment:
    @env_tool("Execute a bash command")
    async def bash(self, command: str, timeout: int = 120) -> str:
        result = await run_command(command, cwd=self._worktree_path)
        await self._auto_commit(f"bash: {command[:50]}")
        return result.stdout
```

The schema generation is the real win. Dependency injection solves a problem (tool reuse across environments) that rollouts doesn't really have.

## Rollouts Advantages Over Browser-Use

Things browser-use lacks that rollouts has:
- Parallel execution + reduction (`run_parallel_agents()`, `reduce_select_best()`)
- Session persistence (full serialize/deserialize)
- Granular streaming events (20+ types vs 6)
- Profiling infrastructure (`LLMCallEnd` with timing)
- Stop handlers composition

## Files Referenced

- `/tmp/browser-use-agent-sdk/bu_agent_sdk/tools/decorator.py` - `@tool` implementation
- `/tmp/browser-use-agent-sdk/bu_agent_sdk/tools/depends.py` - `Depends()` implementation
- `/tmp/browser-use-agent-sdk/bu_agent_sdk/agent/compaction/service.py` - Context compaction
- `rollouts/rollouts/agents.py:671-790` - `process_pending_tools()` serialize/deserialize loop
- `rollouts/rollouts/environments/git_worktree.py:240-277` - Environment serialize/deserialize

## Next Steps

- [ ] Add `@env_tool` decorator to auto-generate `Tool` from method signatures
- [ ] Consider ephemeral tool output support
- [ ] Consider automatic compaction in `RunConfig`
