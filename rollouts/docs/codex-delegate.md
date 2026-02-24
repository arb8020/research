# Codex Delegation Patterns

Use Claude Code for thinking/debugging/architecture. Delegate to Codex Spark for well-defined edits and searches.

## When to Delegate

**Good for Codex:**
- Mechanical refactors (asyncio→trio, rename across files)
- Pattern replacements with clear before/after
- File structure changes (move function, extract module)
- **Codebase search** - find all X, return file:line + summary as markdown
- Applying a fix once root cause is known

**Keep in Claude Code:**
- Root cause analysis / debugging
- Architecture decisions
- Exploratory search (don't know what we're looking for yet)
- Multi-step tasks where each step informs the next
- Anything requiring back-and-forth clarification

## Task Template

```
Task: {one-liner}

## Goal
{problem we're solving, why it matters}

## Files to Read
1. path/to/main/file.py - {what it does}
2. path/to/related.py - {why it's relevant}

## What to Change
{file-by-file or function-by-function instructions}
{be specific about patterns, not vague}

## Don't Change
{explicit boundaries - files/functions to leave alone}
```

## Running Codex

```bash
# Launch in tmux with streaming logs
tmux new -d -s codex "cat /tmp/task.md | codex exec -m gpt-5.3-codex-spark --json | tee /tmp/codex.log"

# Check progress
tmux capture-pane -t codex -p
# Or stream:
tail -f /tmp/codex.log | jq -r '.type + ": " + (.message // .content // "")'

# Quick inline task (no tmux)
echo "Rename function foo to bar in src/utils.py" | codex exec -m gpt-5.3-codex-spark
```

## After Codex

1. Check the diff: `git diff path/to/file.py`
2. Test manually (we verify, codex implements)
3. If wrong, refine task and re-run

## Examples

### Refactor: asyncio to trio
```
Task: Refactor server.py from asyncio to trio

## Goal
Fix startup race condition where result_dispatcher doesn't start before server accepts requests.

## Files to Read
1. rollouts/inference/server.py - main file to refactor
2. rollouts/inference/benchmark/runner.py - example of trio usage

## What to Change
server.py:
- Replace import asyncio with import trio
- Replace asyncio.Future with trio.Event + result storage
- Replace asyncio.Queue with trio.MemoryChannel
- Replace asyncio.create_task with nursery.start_soon
- Use hypercorn with trio worker

## Don't Change
- EngineThread class (regular Python thread)
- benchmark runner (already uses trio)
```

### Fix: Context manager misuse
```
Task: Fix modal.enable_output() context manager usage

## Goal
modal.enable_output() is a context manager but called as function. No Modal logs showing.

## Files to Read
1. rollouts/modal_runner.py - file to fix

## What to Change
1. Remove modal.enable_output() call in _create_sandbox() (~line 144)
2. In run_modal(), wrap the trio_asyncio.open_loop() block:
   - Add `import modal`
   - Change `async with trio_asyncio.open_loop():` to:
     ```python
     with modal.enable_output():
         async with trio_asyncio.open_loop():
     ```

## Don't Change
- _build_modal_image()
- _sync_code_to_sandbox()
```

### Search: Find all usages
```
Task: Find all places where we create Modal sandboxes

## Goal
Understand sandbox creation patterns before refactoring.

## Files to Read
1. rollouts/modal_runner.py
2. Any other files that import modal.Sandbox

## What to Return
List of:
- File path + line number
- Function name
- How sandbox is created (params passed)
- Any patterns/inconsistencies noticed
```

### Trace: Data flow analysis
```
Task: Trace the flow of training gradients from loss to weight update

## Goal
Understand gradient flow before adding gradient logging.

## Starting Point
rollouts/training/grpo.py - compute_loss() function

## What to Return
Markdown with:
- Ordered list of file:line for each step
- What data transforms at each step
- Where gradients could be inspected/logged
```

### Instrument: Add wide event logging
```
Task: Add wide event logging to the rollout generation flow

## Goal
Instrument rollout generation with structured logs for debugging.

## Files to Read First
1. docs/code_style/logging.md - our logging conventions
2. rollouts/rollout/generator.py - file to instrument

## What to Change
Add wide event logs at:
- Function entry (params as structured fields)
- Key decision points (which branch taken, why)
- Function exit (result summary, timing)

Follow the patterns in docs/code_style/logging.md exactly.

## Don't Change
- Business logic
- Return values
- Error handling behavior
```

### Review: Check against conventions
```
Task: Review this diff against our code style

## Goal
Catch style violations before merge.

## Files to Read First
1. docs/code_style/error_handling.md
2. docs/code_style/logging.md
3. The diff (provided below or via git diff)

## What to Return
Markdown with:
- Violations found (file:line + what's wrong)
- Suggested fixes
- "LGTM" if no issues
```
