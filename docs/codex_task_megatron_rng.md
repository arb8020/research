Task: Find how SLIME/Miles initializes Megatron CUDA RNG state

## Goal
We're getting this error when Megatron tries to initialize model weights:
```
Exception: cuda rng state model-parallel-rng is not added
```

This happens in `megatron/core/tensor_parallel/random.py` when the RNG tracker tries to fork but the model-parallel RNG state hasn't been set up.

## What to Find

1. In `/tmp/slime`, search for how they initialize Megatron - specifically:
   - Calls to `model_parallel_cuda_manual_seed`
   - Calls to `_CUDA_RNG_STATE_TRACKER.add` or similar
   - Any `random.py` or RNG-related initialization

2. In `/tmp/miles`, same search

3. Compare to our initialization in:
   - `/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/initialize.py`

## What to Return

- The exact code SLIME/Miles uses to initialize the CUDA RNG state
- What's missing from our `init_megatron()` function
- The fix we need to apply

## Guards
- If SLIME/Miles doesn't exist at /tmp, STOP and report
- If they don't use Megatron, STOP and report

## After Completion
1. Summarize the fix needed
2. Show the exact code to add
3. Suggest prompt improvements
