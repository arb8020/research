# KernelBench Multi-Turn RL Training

This directory contains a Kevin-style multi-turn RL training setup for KernelBench kernel optimization.

**Reference**: [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/abs/2507.11948)

## Overview

The multi-turn approach allows the model to:
1. Generate an initial kernel
2. Receive execution feedback (compilation status, correctness, speedup)
3. Iterate and refine over multiple turns
4. Learn from the full trajectory

## Workflow

### Step 1: Evaluate with API Model (Derisk Environment)

Before starting expensive RL training, test the environment with a capable API model:

```bash
# Set your OpenCode API key
export OPENCODE_API_KEY="your-key-here"  # Get from https://opencode.ai/zen

# Evaluate with Kimi K2.5 (recommended - cheap and capable)
python eval_multi_turn.py --model kimi-k2.5 --provider opencode --num-problems 5 --max-turns 8

# Evaluate with Claude Sonnet via OpenCode
python eval_multi_turn.py --model claude-sonnet-4-6 --provider opencode --num-problems 3

# Evaluate on multiple levels
python eval_multi_turn.py --model kimi-k2.5 --levels 1 2 --num-problems 10 --output results/kimi_eval.json
```

**Expected behavior**:
- Model generates kernel code in Python code blocks
- Environment extracts and "evaluates" the kernel (currently placeholder)
- Feedback is returned to the model
- Model can iterate up to `max_turns`
- Final metrics: correctness rate, speedup, turns used

### Step 2: Evaluate with SGLang (Test Local Inference)

Once the environment works with API models, test with SGLang to ensure the inference stack works:

```bash
# Option A: Auto-start SGLang
python eval_with_sglang.py --model Nanbeige/Nanbeige4.1-3B --num-problems 3 --max-turns 4

# Option B: Use existing SGLang server
# Terminal 1: Start SGLang
python -m sglang.launch_server --model Nanbeige/Nanbeige4.1-3B --port 30000

# Terminal 2: Run evaluation
python eval_multi_turn.py --model Nanbeige/Nanbeige4.1-3B --provider sglang --endpoint http://localhost:30000/v1 --num-problems 5
```

### Step 3: Integrate Real Kernel Evaluation

The current `KernelBenchMultiTurnEnvironment._evaluate_kernel()` is a placeholder. You need to integrate with actual KernelBench evaluation:

```python
# In rollouts/environments/kernelbench_multi.py

async def _evaluate_kernel(self, kernel_code: str) -> dict[str, Any]:
    """Evaluate kernel using KernelBench."""
    # Option 1: Use local evaluation (requires CUDA GPU)
    from kernelbench.eval import eval_kernel_against_ref
    
    result = eval_kernel_against_ref(
        original_model_src=self.ref_code,
        custom_model_src=kernel_code,
        measure_performance=True,
        ...
    )
    
    return {
        "compiled": result.compiled,
        "correct": result.correctness,
        "speedup": result.ref_runtime / result.runtime if result.correctness else 0.0,
        "runtime_us": result.runtime,
        "error": None,
    }
    
    # Option 2: Use remote sandbox (Modal/RunPod)
    # See wafer's optimize_kernelbench_eval for reference
```

### Step 4: Run RL Training

Once evaluation works end-to-end:

```bash
# Multi-turn RL training
python multi_turn_config.py --modal

# Or with custom settings
python multi_turn_config.py --modal --gpu-type A100
```

## Files

| File | Purpose |
|------|---------|
| `kernelbench_multi.py` | Multi-turn environment (no tools, parses code from responses) |
| `eval_multi_turn.py` | Evaluation script for API models and SGLang |
| `eval_with_sglang.py` | Evaluation with auto-managed SGLang server |
| `multi_turn_config.py` | RL training configuration |
| `dataset.py` | KernelBench dataset loading |
| `prompts.py` | System and user prompt templates |
| `scoring.py` | Single-turn scoring (for reference) |

## Key Differences from Single-Turn

| Aspect | Single-Turn | Multi-Turn |
|--------|-------------|------------|
| Environment | `BasicEnvironment` | `KernelBenchMultiTurnEnvironment` |
| Tools | None | None (parsing in `on_assistant_message`) |
| Turns | 1 | Up to 8 |
| Reward | Final kernel only | Best across trajectory |
| Context | Single generation | Full conversation history |

## Model Registry

Available models for evaluation:

```python
# OpenCode (recommended for testing)
"kimi-k2.5"          # Cheap, capable, 1T MoE
"kimi-k2-thinking"   # With reasoning
"claude-sonnet-4-6"  # Claude via OpenCode
"claude-opus-4-5"    # Stronger but expensive

# Direct providers
"kimi-k2.5" @ moonshot   # Direct Moonshot API
"claude-3-5-sonnet" @ anthropic  # Direct Anthropic

# Local (via SGLang)
"Nanbeige/Nanbeige4.1-3B"
"Qwen/Qwen2.5-Coder-3B-Instruct"
"zai-org/GLM-4.7-Flash"
```

## Troubleshooting

### Environment not extracting kernel code
- Check that model outputs code in ```python blocks
- Or uses <kernel>...</kernel> tags
- Or contains `class ModelNew`

### Evaluation always returns failed
- The `_evaluate_kernel` method is a placeholder
- Integrate with real KernelBench evaluation
- Check that `ref_code` is properly passed to environment

### SGLang won't start
- Check CUDA is available: `nvidia-smi`
- Check SGLang is installed: `pip install sglang[all]`
- Try smaller model first (3B params)

### Out of memory
- Reduce model size (3B instead of 32B)
- Use tensor parallelism: `--tp-size 2`
- Reduce `max_seq_len` in config

## Next Steps

1. **Integrate real evaluation**: Connect `_evaluate_kernel` to KernelBench
2. **Test with API model**: Verify end-to-end with Kimi K2.5
3. **Test with SGLang**: Verify local inference works
4. **Run RL training**: Start with small model, short training
5. **Scale up**: Larger model, more steps, more problems

## References

- **Kevin Paper**: https://arxiv.org/abs/2507.11948
- **KernelBench**: https://github.com/ScalingIntelligence/KernelBench
- **OpenCode**: https://opencode.ai/zen
