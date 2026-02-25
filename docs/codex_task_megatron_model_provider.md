Task: Find what kwargs Megatron's get_model() passes to model_provider

## Goal
We're using `**kwargs` to swallow unknown args from Megatron, but that's not explicit. We want to know exactly what `get_model()` passes so we can be explicit in our signature.

## What to Find

1. In Megatron-LM source (https://github.com/NVIDIA/Megatron-LM), find `get_model()` in `megatron/training/training.py`
2. Trace what kwargs it passes to `model_provider_func`
3. List all the parameters it passes (positional and keyword)

## What to Return

Markdown with:
- The function signature of `get_model()`
- The exact call site where it invokes `model_provider_func`
- List of all kwargs passed
- Recommendation: what our `model_provider` signature should be to be explicit (no `**kwargs`)
