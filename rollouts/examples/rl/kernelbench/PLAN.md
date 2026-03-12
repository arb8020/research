# KernelBench RL Training Implementation Plan

**Goal**: Implement tinker-style RL training for KernelBench in the `~/research/rollouts` repository, following the pattern established by `reverse_text` example.

**Reference**: [Kevin-32B](https://cognition.ai/blog/kevin-32b) - RL-trained model for CUDA kernel optimization

## Executive Summary

Train a small language model (Nanbeige4.1-3B or GLM-4.7-Flash) using GRPO to generate optimized GPU kernels. The model learns from correctness and speedup signals when its generated kernels are compiled and benchmarked against PyTorch baselines.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     GRPO Training Loop                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Prompts    │───▶│   Rollouts   │───▶│   Scoring    │      │
│  │ (KernelBench)│    │  (SGLang)    │    │(GPU Compile) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │                   ▼                   │               │
│         │          ┌──────────────┐             │               │
│         └─────────▶│   Trainer    │◀────────────┘               │
│                    │ (Policy Grad)│                             │
│                    └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Difference from reverse_text**: KernelBench requires **GPU execution for scoring**. The model's output must be compiled and run on a GPU to measure correctness and speedup.

---

## 2. Components to Implement

### 2.1 Directory Structure

```
rollouts/examples/rl/kernelbench/
├── __init__.py              # Module exports
├── base_config.py           # train() function + score_fn + prompts
├── grpo_01_01.py            # Baseline config (like reverse_text)
├── grpo_true_pipeline_01.py # PipelineRL variant
├── prompts.py               # System/user prompt templates
├── dataset.py               # Load KernelBench problems
├── scoring.py               # Parse kernel output → Score
└── PLAN.md                  # This file
```

### 2.2 Core Components

#### A. Dataset Loading (`dataset.py`)

```python
def load_kernelbench_prompts(
    levels: list[int] = [1, 2],  # Start with easier levels
    max_samples: int | None = None,
    backend: str = "cuda",  # or "hip"
) -> list[dict[str, Any]]:
    """Load KernelBench problems as prompt dicts.

    Returns:
        List of dicts with:
        - "messages": [system, user] messages
        - "problem_id": int
        - "level": str
        - "ref_arch_src": str (PyTorch reference code)
        - "expected_output_shape": tuple (for validation)
    """
```

**Source**: Use `~/wafer/research/KernelBench/src/dataset.py` pattern:
- `construct_kernelbench_dataset(level)` returns problem paths
- Parse PyTorch `Model` class from each problem file

#### B. Prompts (`prompts.py`)

```python
SYSTEM_PROMPT = """\
You are a GPU kernel optimization expert. Write optimized {backend} kernels.

## Task Format
Given a PyTorch Model class, write an optimized ModelNew class that:
1. Has the same __init__ and forward() signatures
2. Uses custom {backend} kernels via torch.utils.cpp_extension.load_inline
3. Achieves speedup > 1.0x over the PyTorch baseline

## Output Format
Respond with a complete Python file containing:
- ModelNew class with custom kernel
- All necessary imports

Put your final code in <kernel> tags:
<kernel>
# Your optimized implementation
</kernel>
"""

USER_PROMPT = """\
Optimize this PyTorch kernel for {backend}:

```python
{ref_arch_src}
```

Write an optimized ModelNew class with custom {backend} kernels.
"""
```

#### C. Scoring (`scoring.py`)

**Critical**: This is where KernelBench differs from reverse_text. We need GPU execution.

```python
def kernelbench_score_fn(sample: Sample) -> Score:
    """Score a kernel generation sample.

    Requires GPU execution to:
    1. Extract code from <kernel> tags
    2. Compile the kernel
    3. Run correctness tests
    4. Benchmark if correct

    Returns Score with metrics:
    - compiled: 0.0 or 1.0
    - correct: 0.0 or 1.0
    - speedup: float (0.0 if incorrect)
    - reward: 0.2*compiled + 1.0*correct + speedup
    """
```

**Two Options for GPU Scoring**:

1. **Local GPU** (simpler, requires GPU on training node):
   - Compile and run kernels directly in the scoring function
   - Use `torch.utils.cpp_extension.load_inline`
   - Pros: Fast, no network overhead
   - Cons: Requires GPU, may OOM if training also uses GPU

2. **Remote GPU via wafer** (more complex, but decoupled):
   - Call `wafer evaluate kernelbench --impl <file> --reference <ref>`
   - Parse JSON output for correctness/speedup
   - Pros: Can use separate GPU pool, existing infrastructure
   - Cons: Network latency, more moving parts

**Recommendation**: Start with Option 1 (local GPU) for simplicity. The scoring function runs on CPU anyway (SGLang handles GPU for inference).

#### D. Training Config (`base_config.py`)

```python
from rollouts.core import Metric, Score
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import GRPOConfig, grpo_train

def train(
    config: GRPOConfig | None = None,
    num_samples: int = 100,
    levels: list[int] = [1],
    backend: str = "cuda",
) -> dict[str, Any]:
    """Run KernelBench RL training."""

    if config is None:
        config = GRPOConfig(
            output=GRPOOutputConfig(experiment_name="kernelbench_grpo"),
            model=ModelConfig(name="Nanbeige/Nanbeige4.1-3B"),
            trainer=TrainerConfig(lr=1e-6),
            rollout=RolloutConfig(
                n_samples_per_prompt=4,  # Fewer due to expensive scoring
                temperature=0.7,
                max_seq_len=4096,  # Kernels can be long
                max_tokens=2048,
            ),
            checkpoint=CheckpointConfig(num_steps=50),
        )

    prompts = load_kernelbench_prompts(levels=levels, max_samples=num_samples)

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=kernelbench_score_fn,
        environment_cls=BasicEnvironment,  # No tools needed - single-turn generation
    )
```

#### E. Experiment Configs

**`grpo_01_01.py`** - Baseline:
```python
config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="kernelbench_grpo_01"),
    model=ModelConfig(name="Nanbeige/Nanbeige4.1-3B"),
    checkpoint=CheckpointConfig(num_steps=50, checkpoint_every=10),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=4,
        temperature=0.7,
        max_seq_len=4096,
        max_tokens=2048,
    ),
    trainer=TrainerConfig(
        lr=1e-6,
        num_minibatches=4,
    ),
)
```

---

## 3. Key Design Decisions

### 3.1 Single-Turn vs Multi-Turn

**Decision: Single-turn generation**

Unlike the wafer eval (which uses multi-turn with tools for iterative refinement), RL training should use single-turn:
- Simpler reward signal (one generation → one score)
- Faster iteration (no tool execution overhead per turn)
- Cleaner credit assignment (reward applies to entire generation)

The model outputs a complete kernel in one shot. If we later want iterative refinement, we can add a separate "kernel debugger" RL task.

### 3.2 Environment Class

**Decision: BasicEnvironment (no tools)**

For single-turn kernel generation, we don't need tool use. The model generates code, we score it externally. This matches the `reverse_text` pattern.

### 3.3 Reward Shaping

**Reward formula**: `0.2 * compiled + 1.0 * correct + speedup`

- Failed to extract code: 0.0
- Failed to compile: 0.0
- Compiled but wrong: 0.2
- Correct at 1x: 1.2
- Correct at 2x: 2.2
- Correct at 10x: 11.2

This provides dense signal (compilation is easier than correctness) and unbounded upside for optimization.

### 3.4 How Kernel Evaluation Works

The model generates Python code containing a `ModelNew` class. The evaluation:

1. **Extracts code** from `<kernel>` tags or ```python blocks
2. **Writes an inline test script** that:
   - Imports the reference problem (Model, get_inputs, get_init_inputs)
   - exec's the generated code to define ModelNew
   - Instantiates both models
   - Runs correctness tests (torch.allclose)
   - Benchmarks timing if correct
3. **Runs via subprocess** to isolate from main training process
4. **Parses stdout** for results (COMPILE_SUCCESS, CORRECTNESS_RESULT, SPEEDUP_RESULT)

The kernel itself uses `torch.utils.cpp_extension.load_inline` which handles CUDA/HIP compilation internally. No separate nvcc/hipcc invocation needed.

**Important**: The test script runs on the **same GPU** as SGLang inference. The `mem_fraction=0.5` setting leaves room for kernel compilation/benchmarking.

### 3.4 Base Model Selection

**Primary**: `Nanbeige/Nanbeige4.1-3B`
- Small enough for fast iteration
- Chinese/English bilingual (good for code)
- Recent architecture

**Alternative**: `zai-org/GLM-4.7-Flash`
- Larger capacity
- Flash attention built-in
- Better for production

**Note**: Neither model has been SFT'd on kernel code. Consider:
1. Running SFT first on kernel examples (like Prime's reverse_text SFT)
2. Or starting RL from scratch with curriculum (Level 1 → 2 → 3 → 4)

### 3.5 Curriculum Learning

Start with Level 1 problems (simple ops like matmul, softmax, GELU), then progress to harder levels as reward improves. This can be implemented by:

1. **Static curriculum**: Different config files for each level
2. **Dynamic curriculum**: Modify `load_kernelbench_prompts` to sample harder problems as training progresses

---

## 4. Implementation Steps

### Phase 1: Skeleton (Day 1)

1. Create directory structure
2. Implement `dataset.py` with KernelBench loading
3. Implement `prompts.py` with system/user templates
4. Create `base_config.py` skeleton with placeholder score_fn
5. Test prompt generation locally

### Phase 2: Scoring (Day 2-3)

1. Implement `scoring.py` with local GPU compilation
   - Extract code from `<kernel>` tags
   - Write to temp file
   - Compile with `torch.utils.cpp_extension.load_inline`
   - Run correctness test
   - Benchmark if correct
2. Handle compilation errors gracefully (return compiled=0)
3. Test scoring on a few samples manually

### Phase 3: Integration (Day 3-4)

1. Wire up `base_config.py` with real score_fn
2. Create `grpo_01_01.py` config
3. Test end-to-end on 1 problem with 1 step
4. Fix any integration issues

### Phase 4: Training Run (Day 4-5)

1. Run Level 1 training (20-50 steps)
2. Monitor metrics (compiled rate, correct rate, speedup)
3. Debug reward signal issues
4. Iterate on prompts/reward shaping

### Phase 5: Scaling (Week 2)

1. Add more levels (curriculum)
2. Try larger base model
3. Add `grpo_true_pipeline_01.py` for faster training
4. Consider SFT warmup on kernel examples

---

## 5. GPU Requirements

### Training Node
- 1x A100/H100 for SGLang inference
- Model fits in ~6GB (3B params in fp16)
- Leave ~74GB for KV cache

### Scoring
- Same GPU can be used for compilation/benchmarking
- Kernels are small, shouldn't conflict with inference
- If OOM, use separate scoring GPU or remote wafer eval

### RunPod/Modal
```bash
# Run on Modal (recommended for testing)
python examples/rl/kernelbench/grpo_01_01.py --modal --gpu-type A100

# Run on RunPod
python examples/rl/kernelbench/grpo_01_01.py --provision --provider runpod
```

---

## 6. Success Metrics

| Metric | Baseline (Random) | Target (RL) |
|--------|------------------|-------------|
| Compiled Rate | ~10% | >60% |
| Correct Rate | ~5% | >40% |
| Avg Speedup (correct) | 1.0x | >1.5x |
| Level 1 Pass Rate | ~5% | >50% |

---

## 7. Risks and Mitigations

### Risk 1: Compilation Timeout
Kernel compilation can be slow (10-60s).
**Mitigation**: Set timeout, return compiled=0 on timeout.

### Risk 2: GPU OOM during Scoring
Large kernels may OOM when benchmarked.
**Mitigation**: Wrap scoring in try/except, use small input sizes for benchmarking.

### Risk 3: Low Initial Reward
Base model may not generate valid code initially.
**Mitigation**:
- SFT warmup on working kernel examples
- Start with very simple Level 1 problems
- Use higher temperature (0.9-1.0) for exploration

### Risk 4: Reward Hacking
Model might generate trivial "optimizations" (e.g., removing computation).
**Mitigation**: Correctness must pass to get speedup reward.

---

## 8. Dependencies

```toml
# Already in rollouts
trio = "*"
httpx = "*"
datasets = "*"  # For loading from HuggingFace

# KernelBench specific (installed on remote GPU)
torch = ">=2.0"  # For kernel compilation
ninja = "*"      # For fast compilation
```

---

## 9. Dataset Source

**No git clone needed!** KernelBench is available on HuggingFace:
- Dataset: https://huggingface.co/datasets/ScalingIntelligence/KernelBench
- 270 problems across 4 levels
- Each row contains full Python code (Model, get_inputs, get_init_inputs)

The `dataset.py` loads directly from HuggingFace:
```python
from datasets import load_dataset
dataset = load_dataset("ScalingIntelligence/KernelBench", split="train")
```

The reference code is passed to the remote sandbox as part of the evaluation script (no file paths needed).

---

## 10. References

- [KernelBench HuggingFace](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)
- [KernelBench GitHub](https://github.com/ScalingIntelligence/KernelBench)
- [Kevin-32B Blog Post](https://cognition.ai/blog/kevin-32b)
- [Prime-RL Reverse Text](https://github.com/PrimeIntellect/prime-rl)
- [wafer KernelBench Eval](~/wafer/research/evals/optimize_kernelbench_eval/)
- [rollouts GRPO](~/research/rollouts/rollouts/training/grpo.py)

---

## 11. Distributed Architecture (GPU Separation)

### 11.1 Design Philosophy

Inspired by miles/slime but simplified for our scale:

- **miles/slime**: Use Ray placement groups to allocate GPUs to training vs inference
- **Our approach**: Use miniray for coordination, external sandboxes for kernel scoring
- **Key insight**: We don't need Ray's complexity because sandboxes are ephemeral and provider-agnostic

### 11.2 Current Architecture (Single Node)

For 3B models, everything fits on one H100:

```
Training Node (RunPod/Modal H100)
├── SGLang (inference) ─────────┐
├── Trainer (GRPO)              │ same GPU, mem_fraction=0.5
└── SandboxPool ────────────────┘
        │
        │ TCP (miniray RemoteWorker)
        ▼
    Scoring Sandboxes (external)
    ├── Modal A100 × 2
    └── RunPod H100 × 1
```

- Training and inference share the same GPU (colocated)
- Kernel scoring runs on **external sandboxes** via SandboxPool
- Sandboxes are ephemeral, provisioned on demand

### 11.3 Future Architecture (Multi-Node)

For larger models (7B+) or true pipelining:

```
Training Cluster               Inference Cluster           Scoring Sandboxes
├── Node 0 (FSDP rank 0)      ├── Node 0 (SGLang)        ├── Sandbox 0
├── Node 1 (FSDP rank 1)      └── Node 1 (SGLang)        └── Sandbox 1
└── ...                               │
        ▲                             │
        └── weight sync (miniray) ────┘
```

- Training nodes run FSDP across GPUs (miniray for coordination)
- Inference nodes run SGLang (separate from training)
- Weight sync via miniray NCCL helpers
- Scoring sandboxes unchanged (external to both clusters)

### 11.4 SandboxPool Abstraction

Located at `rollouts/rollouts/gpu_sandbox/`:

```python
from rollouts.gpu_sandbox import SandboxPool, ModalSandboxConfig

# Configure sandbox providers
pool = SandboxPool([
    ModalSandboxConfig(gpu="A100", count=2),
    # Can mix providers:
    # RunPodSandboxConfig(gpu="H100", count=1),
    # SSHSandboxConfig(hosts=("user@gpu-server",)),
])

# Start sandboxes (provisions on demand)
await pool.start()

# Score a batch of samples (distributes across workers)
scores = await pool.score_batch([
    {"kernel_code": code1, "ref_code": ref1},
    {"kernel_code": code2, "ref_code": ref2},
])

# Cleanup
await pool.stop()
```

For local/testing (no external sandboxes):
```python
pool = SandboxPool([])  # Empty = local subprocess scoring
```

### 11.5 Why Not Ray?

We evaluated miles/slime's Ray-based architecture and chose miniray instead:

| Feature | Ray | miniray | Our Need |
|---------|-----|---------|----------|
| Placement groups | ✅ | ❌ | Not needed - sandboxes are external |
| Actor model | ✅ | ❌ | Not needed - simple request/response |
| NCCL setup | Via actors | ✅ Built-in helpers | ✅ For future multi-node |
| Complexity | High | Low (~600 LOC) | Prefer simple |
| Fault tolerance | ✅ | ❌ (manual) | OK for research scale |

Key insight: miles/slime need placement groups because training and inference share a Ray cluster. We don't share - scoring sandboxes are completely external.

### 11.6 Scaling Path

1. **Now**: Single node, SandboxPool for scoring
2. **Next**: Multi-node training via miniray Cluster + FSDP
3. **Later**: Separate inference cluster if needed

The SandboxPool abstraction is already provider-agnostic, so scaling sandboxes is just config.

---

## 12. Open Questions for Implementation Team

1. **SFT Warmup**: Should we create an SFT dataset from working kernels first, or go straight to RL?

2. **Backend**: Start with CUDA or HIP? CUDA has more examples but HIP is wafer's focus.

3. **Multi-turn Later**: After single-turn works, should we add a multi-turn "kernel debugger" variant?

4. **Model Selection**: Nanbeige-3B for iteration speed, or GLM-4.7B for capacity? Or both?

5. **Sandbox Provider**: Start with Modal (fast cold start) or RunPod (can keep alive)? Or support both from day 1?
