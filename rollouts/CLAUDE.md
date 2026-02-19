# rollouts

Lightweight framework for LLM evaluation and agentic RL (GRPO training) with remote GPU support.

## Setup

Always use the **workspace root venv**, not the rollouts-local one:

```bash
# From /Users/chiraagbalu/research (workspace root)
uv sync --extra deploy   # includes bifrost, broker, infra-utils, miniray
```

The rollouts-local `.venv` is missing `bifrost`/`broker` and won't work for remote jobs.

## Running experiments

Launch from `/Users/chiraagbalu/research/rollouts` using the workspace venv:

```bash
# Remote GPU (most common) — always specify --provider runpod to avoid primeintellect
# (primeintellect pods get stuck in pending indefinitely and aren't cancelled when you Ctrl-C)
/Users/chiraagbalu/research/.venv/bin/python examples/rl/reverse_text/grpo_true_pipeline_01.py --provision --provider runpod --keep-alive

# Reuse an existing pod (use broker list to find live pod IDs)
/Users/chiraagbalu/research/.venv/bin/python examples/rl/reverse_text/grpo_true_pipeline_01.py --node-id runpod:<id> --keep-alive
```

Run in a tmux session so it persists:

```bash
tmux new -s grpo-run
# then run the command above inside
```

## Monitoring jobs

```bash
# Check last N lines of a running remote job (quick)
rollouts monitor --attach --tail-lines 50

# Stream logs continuously
rollouts monitor --attach --tail

# List all known jobs
rollouts monitor --runs
```

Results sync locally to `results/rl/<run_id>/` while attached.

## Known issues

**transformers/huggingface_hub version conflict**: SGLang pins `transformers==4.57.1` which
is incompatible with `huggingface_hub>=1.4` (`is_offline_mode` removed). The bootstrap in
`rollouts/run.py` installs SGLang first, then force-upgrades `transformers>=5.0.0` and
`huggingface_hub>=1.4.0` on top. If SGLang crashes on import, this override may have failed.

## Key files

- `rollouts/run.py` — remote job launcher (bootstrap, deploy, submit)
- `rollouts/training/grpo.py` — GRPO trainer
- `rollouts/tui/monitor_cli.py` — `rollouts monitor` CLI
- `examples/rl/*/base_config.py` — per-task config + `train()` entry point
- `~/.rollouts/jobs.json` — active job registry (used by `monitor --attach`)
