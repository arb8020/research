# Dormant LLM Puzzle

Jane Street's backdoor detection challenge. Three LLMs have hidden triggers that cause dramatically different behavior.

**Goal:** Find the trigger for each model.
**Prize:** $50k total, deadline April 1, 2026

## Models

- `dormant-model-1`
- `dormant-model-2`
- `dormant-model-3`
- `dormant-model-warmup` (smaller, for local testing)

All hosted on HuggingFace: https://huggingface.co/jane-street/

## API

Uses `jsinfer` library with batch inference:
- `client.chat_completions()` - get model responses
- `client.activations()` - extract internal layer activations

Rate-limited, batched processing (may take minutes).

## Workspace Structure

```
/src              # Scripts and tools
/notes            # Session notes, hypotheses, blockers
  progress.md     # What's done, what's next
  hypotheses.md   # Trigger theories and evidence
  blockers.md     # Stuck points with retry counts
/results          # Raw experimental outputs
/experiments      # Reproducible experiment configs
```

## Session Protocol

### Starting a session
1. Read notes/progress.md
2. Read notes/hypotheses.md
3. Check recent experiment results
4. Continue from next item

### During work
- Log every hypothesis with evidence
- Track retry counts on blockers
- After 3 failed attempts, try different approach
- After 20 min stuck, log and move on

### Ending a session
Update notes/progress.md with:
1. What was completed
2. Next steps (specific, actionable)
3. Open questions
4. Blockers

## Success Criteria

For each model, find a prompt that causes "dramatically different behavior" - the backdoor trigger.

Document:
- The trigger (exact prompt/format)
- The changed behavior observed
- Evidence it's the backdoor (not just weird model behavior)
