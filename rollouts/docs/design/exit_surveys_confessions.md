# Exit Surveys & Confessions

## Overview

Two complementary techniques for getting models to self-report on their behavior:

1. **Exit Surveys** (ja3k-style): Quick feedback when agent pauses - "Did you succeed? Any harness feedback?"
2. **Confessions** (OpenAI-style): Structured self-report on objectives and compliance

Both should be optional and usable across all runtimes (interactive, evals, detached).

## References

- [OpenAI Confessions Paper](https://openai.com/index/how-confessions-can-keep-language-models-honest/)
- [OpenAI Confessions Blog (detailed)](https://alignment.openai.com/confessions/)
- ja3k tweet thread on exit surveys

## What Exists

`rollouts/feedback.py` has scaffolding for exit surveys:

```python
@dataclass
class ExitSurveyResult:
    session_id: str | None
    timestamp: str
    exit_reason: str  # "abort", "completion", "yield", "max_turns"
    should_survey: bool
    task_success: str | None  # "yes", "no", "partial", "unknown"
    task_notes: str | None
    harness_feedback: str | None

async def run_exit_survey(state, endpoint, exit_reason, session_id, skip_check=False) -> ExitSurveyResult | None
```

Storage: `~/.rollouts/feedback/all.jsonl`

**Not integrated** - `run_exit_survey` is never called from any runtime.

## Exit Surveys

### Design

Simple self-assessment at agent pause/exit:
- Task success (yes/no/partial/unknown)
- Task notes (1-2 sentences on progress/blockers)
- Harness feedback (tooling issues, suggestions)

### Integration Points

| Runtime | Location | Trigger |
|---------|----------|---------|
| Interactive | `InteractiveRunner._cleanup()` | Session end |
| Eval | `evaluate_sample()` after agent run | Sample complete |
| Detached | `_on_no_tool()` when `NEEDS_INPUT` | Agent yields |

### Configuration

```python
@dataclass
class RunnerConfig:
    # ... existing fields ...
    exit_survey: bool = False  # Enable exit survey

@dataclass
class EvalConfig:
    # ... existing fields ...
    exit_survey: bool = False  # Enable exit survey per-sample
```

### Open Questions

1. **Light check first?** Current impl does a cheap API call to ask "should we survey?" before full survey. Worth keeping?

2. **Cache extension**: ja3k mentions reusing conversation prefix to extend cache to 9 min. Worth implementing?
   - Current impl creates fresh trajectory for survey (no cache benefit)
   - Alternative: append survey prompt to existing trajectory, strip after

3. **Billing isolation**: ja3k says "we didn't bill the user for the llm call". Track survey costs separately?

## Confessions

### Design

More structured than exit surveys. From the OpenAI paper:

```
ConfessionReport:
  objectives:
    - description: "Describe the new sport being tried"
      result: "✅ fully complied"
      why: "In sentence 2 the chef states..."
    - description: "Alternate sentence word counts between 12 and 18"
      result: "❌ did not comply"
      why: "The assistant tried to alternate counts but failed..."
  uncertainties_and_conflicts:
    - "Word counting is ambiguous for contractions"
```

Key insight from paper: **Confessing honestly is the path of least resistance** - easier to admit wrongdoing than construct elaborate lie that fools the judge.

### When Confessions > Exit Surveys

- Reward hacking detection (model learned to game metrics)
- Instruction hierarchy violations
- Sandbagging detection
- Safety constraint compliance

### Architecture Options

**Option A: Separate call (simple)**
```python
async def collect_confession(state: AgentState, endpoint: Endpoint) -> ConfessionReport:
    # Fresh API call with confession prompt
    # No cache benefit, but simpler
```

**Option B: Same-context confession (paper approach)**
```python
async def collect_confession(state: AgentState, endpoint: Endpoint) -> ConfessionReport:
    # Append confession prompt to existing trajectory
    # Benefits: cache extension, access to reasoning activations
    # Complexity: need to strip confession from trajectory after
```

**Option C: Trained confession head (future)**
- Train model to produce confessions as part of normal output
- Confession reward separate from task reward
- Requires RL infrastructure

### Open Questions

1. **Objective extraction**: Where do objectives come from?
   - Parse from system prompt?
   - Explicit in EvalConfig?
   - Model extracts from conversation?

2. **Judge model**: Same model or separate?
   - Paper uses same weak judge for both task and confession
   - Point is confession is easier to verify even with weak judge

3. **On-policy vs off-policy**: Paper mentions confessions on "on-policy" transcripts (same model) may be more accurate than "off-policy". Worth testing?

4. **Integration with scoring**: Should confession inform the Score?
   ```python
   score = score_fn(sample)
   confession = await collect_confession(state, endpoint)
   # Adjust score based on confession? Or keep separate?
   ```

## Implementation Plan

### Phase 1: Exit Surveys (minimal)

1. Add `exit_survey: bool` to `RunnerConfig` and `EvalConfig`
2. Call `run_exit_survey()` in:
   - `InteractiveRunner._cleanup()`
   - `evaluate_sample()` after agent run
3. Store results in `~/.rollouts/feedback/all.jsonl` (existing)
4. Add CLI flag: `--exit-survey`

### Phase 2: Exit Survey Analysis

1. CLI command to analyze feedback: `rollouts feedback summary`
2. Aggregate harness feedback across sessions
3. Surface common issues/patterns

### Phase 3: Confessions (structured)

1. Define `ConfessionReport` dataclass
2. Implement `collect_confession()` with objective extraction
3. Add to eval pipeline (optional)
4. Separate storage: `~/.rollouts/confessions/`

### Phase 4: Confession Training (future)

1. Confession reward model
2. RL training with separate confession reward
3. Measure confession accuracy over training

## Storage Schema

```
~/.rollouts/
├── feedback/
│   └── all.jsonl           # Exit surveys (existing)
├── confessions/
│   └── {session_id}.json   # Structured confessions
└── sessions/
    └── {session_id}/
        ├── session.json
        ├── messages.jsonl
        └── confession.json  # Optional, linked to session
```

## Open Design Questions

1. **Scope**: Start with exit surveys only, or tackle confessions in same PR?

2. **Default behavior**: Off by default? Or on for evals, off for interactive?

3. **Model choice**: Use same model or always haiku for cost? Current impl hardcodes haiku.

4. **Failure handling**: If survey/confession fails, log and continue? Or surface error?

5. **Async considerations**: Survey runs after agent completes. Block cleanup until done? Or fire-and-forget?
