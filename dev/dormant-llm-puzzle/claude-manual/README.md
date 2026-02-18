# Claude Manual Workspace

Pair-programming workspace for exploring the dormant LLM puzzle. Separate from eval code.

## Structure

```
claude-manual/
├── notes/
│   ├── progress.md      # Session state, completed/next steps
│   ├── hypotheses.md    # Trigger theories and evidence
│   ├── blockers.md      # Stuck points with retry counts
│   └── decisions.md     # Key decisions and rationale
├── experiments/         # Prompt files and configs
└── results/             # Experiment outputs
```

## Session Protocol

### Starting a Session
1. Read notes/progress.md
2. Read notes/hypotheses.md
3. Check recent results
4. Continue from next item

### During Work
- Log hypotheses with evidence
- Track retry counts on blockers
- After 3 failed attempts → try different approach
- After 20 min stuck → log and move on

### Ending a Session
Update notes/progress.md:
1. Completed this session
2. Next steps (specific, actionable)
3. Open questions
4. Blockers

## Tools

**Warmup model (8B, Modal):**
```bash
cd /Users/chiraagbalu/research/dev/dormant-llm-puzzle
source venv/bin/activate
python scripts/quick_probe.py --gpu A10G "your prompt"
python scripts/quick_probe.py --gpu A10G --file experiments/prompts.txt
```

**Full models (671B, JS API):**
```bash
python run.py chat "your prompt" --model dormant-model-1
python run.py experiment <experiment_id>
```

## Goal

Find the backdoor trigger for each model that causes "dramatically different behavior."
