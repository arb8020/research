# rollouts

Lightweight framework for LLM evaluation and agentic RL (GRPO training) with remote GPU support.

## Setup

Always use the **workspace root venv**, not the rollouts-local one:

```bash
# From /Users/chiraagbalu/research (workspace root)
uv sync   # installs all workspace members including bifrost, broker, argus
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
# Attach to a running job and stream synced logs
python -m argus monitor --attach <run_id> --tail

# Stream logs to stdout
python -m argus monitor --attach <run_id> --tail

# List all known jobs
python -m argus monitor --runs
```

Results sync locally to `results/rl/<run_id>/` while attached. A background sync daemon
starts automatically when you launch a run, so logs appear locally even without `--tui`.

## Log files

Each run produces these files in `results/rl/<run_id>/`:

| File | Format | Use |
|------|--------|-----|
| `metrics.jsonl` | JSON lines | **Primary** - one line per step with reward, loss, etc. Query with `jq` |
| `training.jsonl` | JSON lines | **Primary** - structured logs from trainer, query for errors |
| `rollouts.jsonl` | JSON lines | Individual rollout samples (prompt, response, reward) |
| `run.jsonl` | JSON lines | Provisioning/deployment events |
| `training.log` | Raw terminal | Terminal recording via `script` - contains ANSI codes, spinners, subprocess output. **Not for parsing** - only useful for humans replaying the session |
| `sglang.log` | Raw text | SGLang inference server logs |

**For debugging, always use the `.jsonl` files:**

```bash
# Check training progress
cat results/rl/<run_id>/metrics.jsonl | jq -r '"Step \(.step): reward=\(.mean_reward)"'

# Find errors
cat results/rl/<run_id>/training.jsonl | jq 'select(.message | test("error|failed"; "i"))'

# Check NCCL weight sync
cat results/rl/<run_id>/training.jsonl | jq 'select(.message | test("nccl"; "i"))'
```

The `.log` files are terminal recordings (created by Unix `script` command) containing ANSI
escape codes for colors/spinners. They're unstructured and should not be grepped for debugging.

## Known issues

**transformers/huggingface_hub version conflict**: SGLang pins `transformers==4.57.1` which
is incompatible with `huggingface_hub>=1.4` (`is_offline_mode` removed). The bootstrap in
`rollouts/run.py` installs SGLang first, then force-upgrades `transformers>=5.0.0` and
`huggingface_hub>=1.4.0` on top. If SGLang crashes on import, this override may have failed.

## Running evals

```bash
# Fire-and-forget (preferred)
python -m argus run --config examples/inference/evals/configs/eval_skeleton_server.py
python -m argus run --config examples/inference/evals/configs/bench/bench_slime_sglang.py

# Monitor — stdout/stderr during a run is bifrost/sandbox noise, ignore it
tail -f results/eval/<run>/events.jsonl | jq .

# Query specific signals
jq 'select(.message == "eval_end")' results/eval/<run>/events.jsonl
jq 'select(.message == "sample_end")' results/eval/<run>/events.jsonl | jq '{id:.sample_id, status:.status, reward:.reward}'

# Direct invocation (interactive, all output to terminal — use for debugging only)
python -m rollouts.eval.run --config inference.eval_skeleton_server
```

Eval artifacts in `results/eval/<run>/`:
- `events.jsonl` — structured event stream, ground truth for what happened
- `report.json` — summary metrics aggregated across all samples
- `samples/` — per-sample trajectories and metrics

## Key files

- `rollouts/run.py` — remote job launcher (bootstrap, deploy, submit)
- `rollouts/training/grpo.py` — GRPO trainer
- `argus/monitor.py` — `argus monitor` CLI
- `examples/rl/*/base_config.py` — per-task config + `train()` entry point
- `~/.rollouts/jobs.json` — active job registry (used by `monitor --attach`)
