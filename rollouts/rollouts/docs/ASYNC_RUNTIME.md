# Async Runtime Migration

**DRI:** chiraag
**Claude:** [this conversation]

## Context
Trio's ecosystem incompatibility (especially with Textual) is blocking frontend work; we need a path to asyncio interop while preserving structured concurrency.

## Out of Scope
- Rewriting existing working code that doesn't touch async boundaries
- Supporting Python < 3.10
- Building our own async abstraction

## Solution
**Input:** Codebase using trio directly (`trio.open_nursery`, `trio.sleep`, etc.)
**Output:** Codebase using AnyIO, runnable on either trio or asyncio backend

## Usage
```python
# Before
import trio

async def run_agent():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(stream_response)

trio.run(run_agent)

# After
from rollouts.aio import create_task_group, run

async def run_agent():
    async with create_task_group() as tg:
        tg.start_soon(stream_response)

run(run_agent, backend="asyncio")  # or "trio"
```

---

## Details

### Flow
1. Add `anyio>=4.0` to dependencies
2. Create `rollouts/aio.py` compatibility layer that re-exports anyio primitives
3. Migrate imports file-by-file (agents → streaming → frontends → CLI)
4. Update entry points to use `anyio.run()`
5. Test with both backends in CI
6. Remove direct trio imports

### Open Questions
- [ ] Drop trio entirely or keep as optional backend?
- [ ] Bump minimum Python to 3.11 for native TaskGroup as fallback?
- [ ] Which backend should be default? (asyncio for compatibility, trio for correctness?)

### Files
**Read:**
- `rollouts/agents.py`
- `rollouts/frontends/tui/interactive_agent.py`
- `rollouts/cli.py`

**Modify:**
- `pyproject.toml` (add anyio dep)
- All files importing trio directly
- Create `rollouts/aio.py`

---

### Miscellaneous Notes

**API Mapping:**
| Trio | AnyIO |
|------|-------|
| `trio.open_nursery()` | `anyio.create_task_group()` |
| `nursery.start_soon(fn)` | `tg.start_soon(fn)` |
| `trio.sleep(n)` | `anyio.sleep(n)` |
| `trio.Event` | `anyio.Event` |
| `trio.CancelScope` | `anyio.CancelScope` |
| `trio.move_on_after(n)` | `anyio.move_on_after(n)` |
| `trio.open_memory_channel()` | `anyio.create_memory_object_stream()` |

**Why not pure asyncio?**
- `asyncio.TaskGroup` (3.11+) exists but lacks `CancelScope`, `move_on_after`
- Would need to implement these ourselves
- AnyIO already did this work

**Why not stay with trio?**
- Textual is asyncio-native, no clean bridge
- Most LLM SDKs (OpenAI, Anthropic) are asyncio
- trio-asyncio bridge adds complexity and edge cases

**Risks:**
- AnyIO is another dependency to maintain
- Subtle behavior differences between backends could cause bugs
- Some trio-specific features (`trio.lowlevel`) have no equivalent
