# Subagent AX (Agent Experience) Feedback

Date: 2025-01-04
Context: Testing subagent spawning patterns from a parent agent's perspective

## Patterns Tested

### Pattern 1: `--stream-json` (Background)

```bash
rollouts -p "task" --env coding --tools readonly --stream-json > /tmp/out.jsonl 2>&1 &
```

**Pros:**
- ✅ Non-blocking - can run multiple in parallel
- ✅ Structured output - can parse events programmatically
- ✅ Get session ID for later inspection
- ✅ Can monitor progress mid-stream

**Cons:**
- ❌ **OAuth message goes to stdout** - `🔐 Using OAuth...` breaks JSON parsing
- ❌ **Large tool results truncated mid-JSON** - malformed lines
- ❌ Complex jq incantations needed to extract final answer
- ❌ PID management, wait, file cleanup overhead
- ❌ No easy "just give me the final text" option

### Pattern 2: Simple Blocking (`-p` with stderr suppressed)

```bash
result=$(rollouts -p "task" --env coding --tools readonly 2>/dev/null)
```

**Pros:**
- ✅ **Much simpler** - just capture stdout
- ✅ Nicely formatted response
- ✅ No parsing needed
- ✅ Subagent was thorough and synthesized well

**Cons:**
- ❌ Sequential - blocks until complete
- ❌ Lost stderr (errors/progress)
- ❌ Tool call output interleaved with final answer (verbose)
- ❌ No structured data for programmatic use

## Verdict

**Pattern 2 wins for simplicity** when you just need an analysis back.

`--stream-json` has too much friction for simple "spawn and get result" use cases.

## Bugs Found

### BUG: OAuth message corrupts JSON stream

The `🔐 Using OAuth authentication (Claude Pro/Max)` message prints to **stdout**, not stderr.
This breaks `--stream-json` parsing since the first line isn't valid JSON.

**Expected:** All non-JSON output should go to stderr when `--stream-json` is active.

### BUG: Large tool results truncate mid-JSON

When tool results are large, some JSON lines in the stream appear truncated/malformed.
May be a buffering issue or intentional truncation that doesn't respect JSON boundaries.

## Feature Requests

### 1. Quiet mode for final-answer-only output

```bash
rollouts -p "task" --env coding -q  # or --quiet
# Only outputs the final assistant response text, no tool calls
```

Use case: Subagent spawning where you just want the conclusion.

### 2. Output format option

```bash
rollouts -p "task" --output-format=final  # Just last assistant text
rollouts -p "task" --output-format=full   # Current behavior (default)
rollouts -p "task" --output-format=json   # Alias for --stream-json
```

### 3. True async spawn with session ID

```bash
sid=$(rollouts --spawn "task" --env coding)  # Returns immediately with session ID
rollouts --status $sid                        # Check if running/complete
rollouts --result $sid                        # Block until done, get result
```

Use case: Fire-and-forget parallel tasks with easy result collection.

### 4. Subagent-optimized preset

A preset that:
- Produces concise, structured output
- Minimizes token usage
- Optimized for being parsed by parent agent

```bash
rollouts -p "task" --preset subagent
```

## Workarounds

### For JSON stream parsing (with OAuth bug):

```bash
# Skip non-JSON lines
cat /tmp/out.jsonl | grep '^{' | jq -s '...'
```

### For getting just final text (blocking):

```bash
# This works but is verbose - you get all tool output too
rollouts -p "task" --env coding 2>/dev/null | tail -n 50
```

---

Filed by: Claude (as the agent using subagents)
