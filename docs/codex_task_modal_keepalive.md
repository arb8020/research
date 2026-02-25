**Task: Add --keep-alive and --sandbox-id to modal_runner + fix model_provider signature**

## Goal
Enable sandbox reuse between test runs to speed up iteration on GLM megatron training. Currently each run creates a fresh sandbox (~slow). Also fix TypeError from Megatron's get_model() passing unexpected kwargs.

## Files to Read
1. `/Users/chiraagbalu/research/rollouts/rollouts/modal_runner.py` - file to modify (check git status first, may be corrupted)
2. `/Users/chiraagbalu/research/rollouts/rollouts/run.py` - reference for --keep-alive and --node-id pattern
3. `/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/model.py` - model_provider signature fix

## Files to Modify
1. `/Users/chiraagbalu/research/rollouts/rollouts/modal_runner.py`
2. `/Users/chiraagbalu/research/rollouts/rollouts/training/backends/megatron/model.py`

## What to Change

### modal_runner.py

**Before editing**: Run `git diff rollouts/rollouts/modal_runner.py` - if file is corrupted with syntax errors, restore with `git checkout rollouts/rollouts/modal_runner.py` first.

1. Add arguments (around line 675-697 where other args are defined):
   ```python
   parser.add_argument("--sandbox-id", type=str, help="Reuse existing sandbox instead of creating new")
   parser.add_argument("--keep-alive", action="store_true", help="Keep sandbox running after completion")
   ```

2. In sandbox creation logic: if `args.sandbox_id` is provided, attach to existing sandbox instead of creating new

3. In finally block (around line 650-658): if `args.keep_alive`, skip `sandbox.terminate()`

4. After run completes, print reuse command:
   ```python
   print(f"To reuse: python -m rollouts.modal_runner --sandbox-id {sandbox.object_id} --config ...")
   ```

### model.py

Line 291-295, add `**kwargs` to model_provider signature:
```python
def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
    **kwargs,  # Megatron's get_model() passes config=... and other args
) -> GPTModel:
```

## Don't Change
- _build_modal_image() in modal_runner.py
- _sync_code_to_sandbox() in modal_runner.py
- Rest of model.py (just the signature)

## Verification
After editing, run:
```bash
python -c "from rollouts.modal_runner import main; print('import ok')"
python -c "from rollouts.training.backends.megatron.model import setup_megatron_model; print('import ok')"
```
