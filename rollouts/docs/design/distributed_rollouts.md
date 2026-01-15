# Distributed Rollouts: Scaling Evaluation

> **Status**: Draft
> **DRI**: Chiraag
> **Date**: 2025-01-12

## Context

We want to run 100s-1000s of rollouts in parallel for RL. Most of the infrastructure already exists:

- `evaluation.py` - `evaluate_sample()` is the atomic unit, already parallelizes with `trio.CapacityLimiter`
- `remote.py` - `acquire_node()` integrates Broker + Bifrost for remote machines
- `training/rollout_gen/async_rollout_manager.py` - SLIME-style parallel generation
- `training/worker.py` - re-exports MiniRay `Worker`

**What's actually missing:**
1. Rate limit proxy - workers hit APIs independently, no coordination
2. Wiring to run `evaluate_sample()` on remote machines via Bifrost

## Out of Scope

- New "orchestrator" abstraction (trio already does this)
- New "worker entry point" abstraction (`evaluate_sample()` already exists)
- RemoteEnvironment (premature - co-locate agent + env for now)
- Cloudflare Worker agent loop (premature optimization)

## Deployment Targets

Bifrost requires SSH. Not all platforms expose SSH:

| Platform | Access | Works with Bifrost? |
|----------|--------|---------------------|
| RunPod/Lambda/Vast (Broker) | SSH | ✅ Yes |
| Your own machines | SSH | ✅ Yes |
| Modal | `@modal.function` / HTTP | ❌ No - needs own wrapper |
| Fly.io | HTTP / `fly ssh` | ⚠️ Maybe |
| Cloudflare Workers | HTTP | ❌ No - needs own wrapper |

**Implication:** The core is `evaluate_sample()`. Deployment wrappers are thin and platform-specific:
- SSH machines: `remote_eval.py` (~30 lines)
- Modal: `modal_worker.py` (~20 lines)
- Others: similar small wrappers

Start with SSH (Bifrost), add Modal wrapper when needed.

## What We Need

### 1. Rate Limit Proxy

Workers currently make independent API calls. When running 100 workers, they'll collectively exceed rate limits.

**Solution:** Simple passthrough proxy with centralized `CapacityLimiter`.

```python
# rollouts/proxy.py - ~50 lines

from fastapi import FastAPI, Request
import httpx
import trio

app = FastAPI()
limiter = trio.CapacityLimiter(50)

@app.api_route("/{provider}/{path:path}", methods=["POST"])
async def proxy(provider: str, path: str, request: Request):
    async with limiter:
        # Passthrough to real API
        response = await httpx.AsyncClient().request(...)
        return Response(response.content, status_code=response.status_code)
```

**Usage:**
```bash
# Start proxy
rollouts proxy --port 8080 --max-concurrent 50

# Workers use proxy
export ANTHROPIC_BASE_URL=http://proxy:8080/anthropic
```

### 2. Remote Evaluation Script

Run `evaluate_sample()` on a remote machine via Bifrost.

**Already have:**
```python
# remote.py
client, instance = acquire_node(provision=True, gpu_type="A100")
client.push()  # Deploy code
client.exec("python script.py")  # Run remotely
```

**Need:** A script that workers can run:

```python
# rollouts/scripts/remote_eval.py

"""Run evaluate_sample() on this machine, read/write via stdin/stdout."""

import sys
import json
from rollouts.evaluation import evaluate_sample, EvalRuntime
from rollouts.training.types import Sample

# Read sample from stdin
sample_dict = json.loads(sys.stdin.read())
sample = Sample.from_dict(sample_dict)

# Run evaluation
result = trio.run(evaluate_sample, sample, runtime)

# Write result to stdout
print(json.dumps(result.to_dict()))
```

**Usage:**
```python
# From orchestrator
client, instance = acquire_node(provision=True)
client.push()

# Run samples
for sample in samples:
    result = client.exec(
        f"echo '{sample.to_json()}' | python -m rollouts.scripts.remote_eval"
    )
    completed.append(Sample.from_dict(json.loads(result.stdout)))
```

Or parallel with MiniRay:
```python
# Start worker servers on remote machines
for instance in instances:
    bifrost.exec("python -m miniray.worker_server --port 10000")

# Connect and dispatch
workers = [RemoteWorker(host, 10000) for host in hosts]
for worker, sample in zip(workers, samples):
    worker.send(sample.to_dict())

# Collect
results = [Sample.from_dict(w.recv()) for w in workers]
```

---

## Files

**Create:**
- `rollouts/proxy.py` - Rate limit proxy (~50 lines)
- `rollouts/scripts/remote_eval.py` - Remote evaluation entry point (~30 lines)

**Modify:**
- `rollouts/cli.py` - Add `rollouts proxy` command

---

## Open Questions

- [ ] Should proxy be Python service or Cloudflare Worker?
- [ ] How to pass API keys to remote workers securely?
- [ ] Should we add `--proxy-url` flag to `rollouts eval`?
