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
# Remote GPU (fire-and-forget, prints run_id and exits)
/Users/chiraagbalu/research/.venv/bin/python examples/rl/reverse_text/grpo_01_01.py --provision --provider runpod

# With --tui to launch interactive monitor after submit
/Users/chiraagbalu/research/.venv/bin/python examples/rl/reverse_text/grpo_01_01.py --provision --provider runpod --tui

# Reuse an existing pod
/Users/chiraagbalu/research/.venv/bin/python examples/rl/reverse_text/grpo_01_01.py --node-id runpod:<id>
```

Always specify `--provider runpod` to avoid primeintellect (pods get stuck in pending).

## Monitoring jobs

```bash
# Attach to a running job (launches TUI)
rollouts monitor --attach <run_id>

# Stream logs to stdout
rollouts monitor --attach <run_id> --tail

# List all known jobs
rollouts monitor --runs
```

Results sync locally to `results/rl/<run_id>/` while attached.

Logs are also written to `results/rl/<run_id>/run.jsonl` (structured events) and `training.log` (raw output).

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
