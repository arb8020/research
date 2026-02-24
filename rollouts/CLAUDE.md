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

## Managing GPU instances

**IMPORTANT**: Pods do NOT auto-terminate when jobs fail or complete. Always check for running
instances before provisioning new ones, and terminate pods you're done with.

Use the `broker` CLI to manage GPU instances:

```bash
# List all running instances (DO THIS BEFORE PROVISIONING)
broker list

# Terminate a specific instance
broker terminate <instance-id>

# Check instance status
broker status <instance-id>

# SSH to an instance
broker ssh <instance-id>
```

Do NOT use `broker cleanup` - it terminates ALL instances across the workspace, which may
kill other people's jobs. Always use `broker terminate <id>` for specific instances.

## Monitoring jobs

```bash
# Attach to a running job (launches TUI)
rollouts monitor --attach <run_id>

# Stream logs to stdout
rollouts monitor --attach <run_id> --tail

# List all known jobs
rollouts monitor --runs
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

## Key files

- `rollouts/run.py` — remote job launcher (bootstrap, deploy, submit)
- `rollouts/training/grpo.py` — GRPO trainer
- `rollouts/training/preflight.py` — GPU/CUDA compatibility checks, memory estimation
- `rollouts/tui/monitor_cli.py` — `rollouts monitor` CLI
- `examples/rl/*/base_config.py` — per-task config + `train()` entry point
- `~/.rollouts/jobs.json` — active job registry (used by `monitor --attach`)

## Session notes (2026-02-24)

**B200 support**: Added CUDA toolkit auto-upgrade. B200 (Blackwell, sm_100a) needs CUDA 12.8+
for FlashInfer/Triton to JIT-compile kernels. After provisioning, we check `nvcc --version`
and if too old, download the toolkit runfile and install with `--toolkit` (doesn't touch driver).
See `GPU_CUDA_REQUIREMENTS` in preflight.py and the bootstrap logic in run.py.

**Logging**: Added `_build_grpo_run_context()` in grpo.py that builds canonical per-run metadata
(run_name, model, backend, device IDs, etc.) attached to every log line via `extra={}`. Good
for filtering logs in production.

**Next steps for B200**: The CUDA 12.8 installer download is ~4GB and takes a few minutes.
Could pre-bake this into a custom Docker image, or use RunPod network volumes to cache it.
The network volume task is in ~/research/todo.md.
