# Training Backend TODOs

## Completed

### LoRA Support for GRPO (Test-Time Training)
- **Commit**: `edbb2ff4` - "fix(training): preserve LoRA adapters after weight sync"
- **Test run**: `results/rl/run_20260125-020210` - 10 steps completed successfully

Key fix: Use `merge_adapter()`/`unmerge_adapter()` instead of `merge_and_unload()` when syncing weights to sglang. The latter permanently destroys LoRA adapters.

---

## Remaining

### 1. Remove HuggingFace transformers dependency
**Location**: `backends/pytorch_factory.py:162-202`

Current: `AutoModelForCausalLM.from_pretrained()`

Target:
- `safetensors.torch.load_file()` for weight loading
- Custom `nn.Module` classes matching HF weight names
- Reference: nmoe repo (pure PyTorch, no HF)

### 2. Remove peft dependency
**Location**: `backends/pytorch_factory.py:116-159`

Current: `peft.get_peft_model()` + `peft.LoraConfig`

Target: Hand-roll LoRA:
```python
# Forward: output = W @ x + (lora_A @ lora_B @ x) * (alpha / rank)
# Merge:   W_merged = W + lora_A @ lora_B * (alpha / rank)
```

---

## Key Files
- `grpo.py` - GRPO trainer, LoRA config fields
- `backends/pytorch.py` - Training backend, weight sync
- `backends/pytorch_factory.py` - Model loading, LoRA wrapping
