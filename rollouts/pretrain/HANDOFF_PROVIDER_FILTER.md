# Handoff: Add Provider Filter to GPU Provisioning

## Problem

When provisioning GPUs via `bifrost.acquire_node()`, there's no way to force a specific provider (e.g., RunPod vs PrimeIntellect). The broker searches all providers and picks the cheapest offer, but some providers have slow boot times (PrimeIntellect can take 5-10 min for cold starts).

## Current State

- `bifrost.GPUQuery` has no `provider` field
- `broker.api.search()` aggregates offers from all providers and sorts by price
- The pretrain run script at `rollouts/pretrain/scripts/run.py` uses GPUQuery without provider control

## Desired Behavior

Add a `--provider` flag to the run script that filters offers to only that provider:

```bash
uv run python pretrain/scripts/run.py --gpu L4 --provider runpod
```

## Implementation Options

### Option 1: Filter in run.py (Quick Fix)

Modify `pretrain/scripts/run.py` to manually search and filter before provisioning:

```python
from broker import api as broker_api

async def run(..., provider: str | None = None):
    if provider:
        # Search manually with provider filter
        offers = await broker_api.search(gpu_type=gpu_type, sort=lambda x: x.price_per_hour)
        offers = [o for o in offers if o.provider == provider]
        if not offers:
            raise RuntimeError(f"No {gpu_type} offers from {provider}")
        instance = await broker_api.create(offers[0], ...)
        # ... rest of setup
    else:
        # Use existing GPUQuery path
        provision = GPUQuery(type=gpu_type, ...)
```

### Option 2: Add provider to GPUQuery (Better)

Modify `bifrost/bifrost/provision.py`:

```python
@dataclass(frozen=True)
class GPUQuery:
    type: str = "A100"
    count: int = 1
    provider: str | None = None  # NEW: filter to specific provider
    # ... rest unchanged
```

Then in `broker/broker/api.py` `_normalize_query_input()`, add:

```python
if query.provider:
    offers = [o for o in offers if o.provider == query.provider]
```

### Option 3: Add provider to broker search API

The `broker.api.search()` function already has filtering but not by provider. Add:

```python
async def search(
    ...,
    provider: str | None = None,  # NEW
):
    # ... existing code ...
    if provider:
        offers = [o for o in offers if o.provider == provider]
```

## Files to Modify

1. **Quick fix (Option 1)**: `rollouts/pretrain/scripts/run.py` only
2. **Better fix (Option 2)**:
   - `bifrost/bifrost/provision.py` - add `provider` to GPUQuery
   - `broker/broker/api.py` - filter by provider in `_normalize_query_input()`
3. **Option 3**: `broker/broker/api.py` - add `provider` param to `search()`

## Testing

After implementing, verify:

```bash
# Should provision on RunPod specifically
uv run python pretrain/scripts/run.py --gpu L4 --provider runpod --steps 10

# Should fail if provider has no matching GPU
uv run python pretrain/scripts/run.py --gpu L4 --provider lambdalabs  # L4 not on Lambda
```

## Context

This came up because PrimeIntellect L40S instances were taking 5+ minutes to boot while trying to test the pretrain training loop. RunPod typically boots in ~1 minute.

The pretrain training itself is ready and tested locally - just need faster GPU provisioning for iteration.
