# LoRA TTT Design

Test-Time Training with LoRA for GRPO. Mirrors Tinker's API pattern.

## Overview

Add LoRA (Low-Rank Adaptation) support to enable efficient test-time training.
LoRA trains only small adapter matrices while keeping base model frozen.

## Changes

### 1. GRPOConfig (grpo.py)

```python
# Add to GRPOConfig
use_lora: bool = False
lora_rank: int = 16
lora_alpha: int = 32
```

### 2. pytorch_factory.py

**New function:**
```python
def wrap_model_with_lora(
    model: torch.nn.Module,
    lora_rank: int = 16,
    lora_alpha: int = 32,
) -> torch.nn.Module:
    """Wrap model with PEFT LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(model, lora_config)
```

**Modify create_pytorch_backend():**
- Add `use_lora`, `lora_rank`, `lora_alpha` parameters
- After loading model, optionally wrap with LoRA
- Optimizer automatically only trains LoRA params (PEFT freezes base)

### 3. pytorch.py

**Add field:**
```python
is_lora: bool = False  # Track if model is PEFT-wrapped
```

**New method (Tinker-inspired):**
```python
async def save_weights_for_sampler(self, path: Path) -> Path:
    """Save merged weights ready for inference.

    If using LoRA, merges adapter weights into base model.
    Saves as HuggingFace format for SGLang/vLLM.
    """
    if self.is_lora:
        # PEFT merge: W' = W + BA
        merged_model = self.model.merge_and_unload()
        await trio.to_thread.run_sync(merged_model.save_pretrained, path)
    else:
        await trio.to_thread.run_sync(self.model.save_pretrained, path)

    config_path = path / "config.json"
    assert config_path.exists(), "save_pretrained must create config.json"
    return path
```

### 4. grpo.py

**In _setup_training_backend():**
```python
backend = create_pytorch_backend(
    model_name=config.model_name,
    # ... existing params ...
    use_lora=config.use_lora,
    lora_rank=config.lora_rank,
    lora_alpha=config.lora_alpha,
)
```

**In _process_training_step() weight sync:**
```python
if should_sync:
    sync_dir = await backend.save_weights_for_sampler(fast_dir / "sync_latest")
    await inference_engine.update_weights_from_checkpoint(str(sync_dir))
```

## Learning Rate

LoRA needs ~20-100x higher LR than full fine-tuning (per Tinker docs):
- Full fine-tuning: `lr=1e-6`
- LoRA: `lr=1e-4` to `lr=1e-5`

## Unchanged

- Weight sync protocol (disk-based, update_weights_from_disk)
- Inference engine code (SGLang/vLLM)
- Checkpoint format (HuggingFace)
- Training loop structure (forward_backward → optim_step)

## Testing

1. Run Fibonacci GRPO with `use_lora=True`
2. Verify LoRA params trainable, base frozen
3. Verify weight sync loads merged weights
4. Verify reward trends up

## API Comparison

| Tinker | Our Codebase |
|--------|--------------|
| `training_client.forward_backward()` | `backend.forward_backward()` |
| `training_client.optim_step()` | `backend.optim_step()` |
| `training_client.save_state_async()` | `backend.save_checkpoint()` |
| `training_client.save_weights_for_sampler_async()` | `backend.save_weights_for_sampler()` |
| `training_client.create_sampling_client()` | `inference_engine.update_weights_from_checkpoint()` |
