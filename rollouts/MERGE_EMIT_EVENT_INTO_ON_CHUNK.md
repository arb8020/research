# Merge emit_event into on_chunk

## Goal
Unify event streaming - use `on_chunk(StreamChunk(...))` for all events instead of split `on_chunk` vs `emit_event`.

## Why
- Same abstraction (events with type + data)
- `emit_event` added for frontend (fb84da7), immediately flagged with TODOs (dtypes.py:489-491)
- Redundant: evaluation.py:282 emits messages that on_chunk already streams as tokens

## Changes

### 1. Add fields to StreamChunk
**File:** dtypes.py:63-66

```python
@dataclass(frozen=True)
class StreamChunk(JsonSerializable):
    kind: str
    data: Mapping[str, Any]
    timestamp: float = field(default_factory=time.time)  # ADD
    turn: int | None = None  # ADD
```

### 2. Update handle_checkpoint_event
**File:** agents.py:56-69

```python
async def handle_checkpoint_event(state: AgentState, event: str, run_config: RunConfig,
                                 session_id: Optional[str] = None) -> None:
    """Handle checkpoint event - emits via on_chunk"""
    await run_config.on_chunk(StreamChunk(
        kind=event,
        data={"session_id": session_id},
        turn=state.turn_idx,
    ))
```

### 3. Update evaluation.py
**File:** evaluation.py:273-277

```python
# Replace emit_event with on_chunk
await run_config.on_chunk(StreamChunk(
    kind="sample_start",
    data={"sample_id": sample_id, "sample_data": sample_data},
))
```

**File:** evaluation.py:279-285
```python
# DELETE - redundant message emission
```

**File:** evaluation.py:408-413
```python
# Replace emit_event with on_chunk
await run_config.on_chunk(StreamChunk(
    kind="sample_end",
    data={"sample_id": sample_id, "reward": metrics.get("reward", 0.0), "metadata": metadata},
))
```

### 4. Delete emit_event from RunConfig
**File:** dtypes.py:488-492

```python
# DELETE entire emit_event field
```

### 5. Update any emit_event providers
Search for code that writes to `events.jsonl` - change to write `StreamChunk.to_json()` format.

### 6. Update frontend event parsing
**File:** frontend/server.py:1670-1678

Expects `{"type": ..., "timestamp": ..., "data": ...}` but StreamChunk has `kind` not `type`.
Either transform on write or update parser to use `kind`.

## New Event Kinds
After merge, emit these via on_chunk:
- `turn_start` - agents.py
- `turn_end` - agents.py
- `final` - agents.py
- `sample_start` - evaluation.py
- `sample_end` - evaluation.py

## Existing Event Kinds (unchanged)
- `token` - providers.py:473
- `thinking` - providers.py:939
- `tool_call_partial` - providers.py:499
- `tool_call_complete` - providers.py:532
- `tool_call_error` - providers.py:516
- `tool_result` - agents.py:508
- `assistant_complete` - providers.py:549

## Testing
- Verify frontend still displays events correctly
- Check events.jsonl format matches frontend expectations
- Ensure no null pointer errors (on_chunk is required, not Optional)
