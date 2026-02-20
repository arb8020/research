# REAP Eval Handoff

## Current State

HellaSwag evaluation is running on a remote pod for the pruned Qwen3-30B-A3B model.

**Pod**: `root@69.19.136.235:31814`
**Tmux session**: `reap-eval`
**Log file**: `/tmp/reap-eval7.log`

### What's Running

- SGLang server serving the pruned model at `http://localhost:30000`
- lm-eval running HellaSwag benchmark (40168 API requests)
- Progress bar stuck at 512/40168 in visible output (tqdm buffering issue with tee)
- Processes still active: `sglang::scheduler` at ~5% CPU, main python eval process

### Model Location

Pruned model saved at:
```
/root/.bifrost/workspaces/rollouts-rl/rollouts/results/reap/Qwen_Qwen3-30B-A3B/theblackcat102_evol-codealpaca-v1/reap-seed_42-0.50
```

Original: 128 experts → Pruned: 64 experts (50% compression)

## How to Check Status

```bash
# Check if processes still running
/Users/chiraagbalu/research/.venv/bin/bifrost exec "root@69.19.136.235:31814" "ps aux | grep -E 'python|sglang' | grep -v grep"

# Check tmux output
/Users/chiraagbalu/research/.venv/bin/bifrost exec "root@69.19.136.235:31814" "tmux capture-pane -t reap-eval -p -S -30"

# Check if results appeared
/Users/chiraagbalu/research/.venv/bin/bifrost exec "root@69.19.136.235:31814" "cat /tmp/reap-eval7.log | grep -E 'Results:|acc'"
```

## Expected Outcome

When complete, the log should show something like:
```
Results:
  hellaswag:
    acc: 0.XXXX
    acc_norm: 0.XXXX
```

The eval script will then terminate the SGLang server and exit.

## If Eval Failed/Stuck

If the processes died or got stuck:

1. Kill any remaining processes:
```bash
/Users/chiraagbalu/research/.venv/bin/bifrost exec "root@69.19.136.235:31814" "pkill -f sglang; pkill -f lm_eval"
```

2. Re-run with the new wrapper (uses batch_size=32 now):
```bash
/Users/chiraagbalu/research/.venv/bin/bifrost exec "root@69.19.136.235:31814" "cd /root/.bifrost/workspaces/rollouts-rl/rollouts && .venv/bin/python -c \"
from rollouts.evaluation.lm_eval import run_lm_eval
from pathlib import Path

model_path = Path('/root/.bifrost/workspaces/rollouts-rl/rollouts/results/reap/Qwen_Qwen3-30B-A3B/theblackcat102_evol-codealpaca-v1/reap-seed_42-0.50')
results = run_lm_eval(
    model_path=model_path,
    tokenizer='Qwen/Qwen3-30B-A3B',
    tasks=['hellaswag'],
    batch_size=32,
)
print(results)
\""
```

## What Was Done This Session

1. Created `rollouts/evaluation/lm_eval.py` - wrapper around lm-evaluation-harness
2. Updated `examples/reap/config.py` - eval_tasks now match original REAP repo
3. Simplified `examples/reap/base_config.py:run_eval()` to use the new wrapper
4. Added ninja-build to bifrost pod bootstrap (needed for SGLang flashinfer JIT)
5. Increased default batch_size to 32 for better SGLang throughput

## Full Eval Suite

After HellaSwag works, the full eval suite is:
- winogrande, arc_challenge, arc_easy, boolq, hellaswag, mmlu, openbookqa, rte

Run with:
```python
from rollouts.evaluation.lm_eval import run_lm_eval
results = run_lm_eval(
    model_path=model_path,
    tokenizer='Qwen/Qwen3-30B-A3B',
    tasks=['winogrande', 'arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'mmlu', 'openbookqa', 'rte'],
)
```

## Pod Info

The pod was provisioned via runpod. If it's been terminated, provision a new one with:
```bash
/Users/chiraagbalu/research/.venv/bin/python examples/reap/configs/qwen3_prune_50.py --provision --provider runpod --keep-alive
```
