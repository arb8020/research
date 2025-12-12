# Codebase Issues

1. ~~**Two `Sample` types**~~ **RESOLVED** - Unified into single `Sample` type in `training/types.py` (Miles pattern). `dtypes.py` re-exports for backward compatibility.

2. ~~**`EvalConfig` collision**~~ **RESOLVED** - Renamed GSM8K's to `GSM8KConfig`. No more shadowing.

3. **Single-turn eval awkward** - Had to use `handle_stop_max_turns(1)`. Add `max_turns` to `EvalConfig`.

4. **`msg.get_tool_calls()` vs `msg.tool_calls`** - Inconsistent API. Pick one.

5. **Lazy imports workaround** - `training/__init__.py` uses `__getattr__`. Split into `training/types.py` (no torch) and `training/torch/`.
