# State Logging Test Plan

## What We're Verifying

1. **Code sent to Modal?**
   - Check if `print("Evaluating code...")` appears in logs
   - ✅ YES - we saw this in error_log.jsonl

2. **Modal execution happened?**
   - Check if state has `results` or `best_result` keys after `env_response()`
   - ❓ UNKNOWN - need to add logging

3. **State correctly updated?**
   - Check if state structure matches what rubric expects
   - ❌ NO - rubric errors show `KeyError: 'best_result'`

## Added Logging

### In `on_assistant_message()` (prime_backend_bench.py:268-304)

**BEFORE env_response:**
```python
self.logger.info("BEFORE env_response:")
self.logger.info(f"  self.state keys: {list(self.state.keys())}")
self.logger.info(f"  self.state.get('results'): ...")
self.logger.info(f"  self.state.get('best_result'): ...")
```

**AFTER env_response:**
```python
self.logger.info("AFTER env_response:")
self.logger.info(f"  updated_state keys: {list(updated_state.keys())}")
self.logger.info(f"  updated_state.get('results'): ...")
self.logger.info(f"  updated_state.get('best_result'): ...")
if updated_state.get('results'):
    self.logger.info(f"  Number of results: {len(updated_state['results'])}")
```

## How to Test

```bash
cd ~/research/dev/integration-evaluation
python local.py configs/prime_backend_bench.py 2>&1 | tee test_run.log
```

## What to Look For

### In error_log.jsonl:

1. **"BEFORE env_response"** section - should show:
   - State keys (expected: `['prompt', 'completion', 'answer', 'task', 'info', ...]`)
   - `results`: Should be `[]` or `NOT FOUND`
   - `best_result`: Should be `None` or `NOT FOUND`

2. **"Evaluating code for op..."** - proves Modal was called

3. **"AFTER env_response"** section - should show:
   - State keys (same as before + potentially `results`, `best_result`)
   - `results`: Should be a list with CodeEvaluationResult objects
   - `best_result`: Should be a CodeEvaluationResult if tests passed
   - Number of results: Should be >= 1

### Expected Flow:

```
BEFORE env_response:
  state keys: ['prompt', 'completion', 'answer', 'task', 'info', 'example_id', 'custom_turns']
  results: NOT FOUND  ← Key is missing
  best_result: NOT FOUND  ← Key is missing

[Modal execution happens]
Evaluating code for op _adaptive_avg_pool2d.default, turn 1/1

AFTER env_response:
  state keys: ['prompt', 'completion', 'answer', 'task', 'info', 'example_id', 'custom_turns', 'results', 'best_result']
  results: [<CodeEvaluationResult object>]  ← Now exists!
  best_result: <CodeEvaluationResult object>  ← Now exists!
  Number of results: 1
```

### If Modal Failed:

```
AFTER env_response:
  state keys: ['prompt', 'completion', 'answer', 'task', 'info', 'example_id', 'custom_turns']
  results: NOT FOUND  ← Still missing - Modal failed!
  best_result: NOT FOUND  ← Still missing
```

## Diagnosis

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `results` missing AFTER | `env_response()` not updating state | Check Prime integration |
| `results` exists but empty | Tests ran but all failed | Check Modal logs for errors |
| `results` has items but no `best_result` | Update logic broken | Check `_update_state_with_result()` |
| Rubric errors before state update | Rubric called too early | Check reward_fn timing |
