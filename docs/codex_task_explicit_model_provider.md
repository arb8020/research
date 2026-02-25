Task: Make model_provider signature explicit (no **kwargs)

## Goal
Replace the vague `**kwargs` with explicit parameters that Megatron actually passes.

## File to Modify
`/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/model.py`

## What to Change

Find the `model_provider` function (around line 291) and change its signature from:
```python
def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
    **kwargs: Any,  # Megatron's get_model() passes config=... and other args
) -> GPTModel:
```

To this explicit signature:
```python
def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    config: Any = None,  # TransformerConfig from Megatron, we use transformer_config from closure
    pg_collection: Any = None,  # ProcessGroupCollection, unused
    vp_stage: int | None = None,
) -> GPTModel:
```

Note: We accept `config` and `pg_collection` but don't use them - we use `transformer_config` from the closure instead. The params exist to match Megatron's expected signature.

## Don't Change
- Anything else in the function body
- Other functions in the file

## Verification
Run: `uvx ruff check /Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/model.py`
