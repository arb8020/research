# Evaluation Gap Analysis

## Comparison: Cerebras REAP vs Our Implementation

---

## Summary

| Category | Cerebras REAP | Our Implementation | Gap |
|----------|---------------|-------------------|-----|
| **lm-eval-harness** | ✅ Full | ✅ Full | None |
| **evalplus** | ✅ mbpp, humaneval | ✅ mbpp, humaneval | None |
| **livecodebench** | ✅ | ❌ Missing | **Major Gap** |
| **wildbench** | ✅ | ❌ Missing | **Major Gap** |
| **Math (evalscope)** | ✅ gsm8k, math_500 | ⚠️ Partial (gsm8k via lm-eval) | Minor Gap |
| **Server Backend** | vLLM | SGLang | Different |
| **Multi-GPU** | ✅ Expert Parallelism | ⚠️ Basic (device_map) | Minor Gap |

**Overall Evaluation Coverage**: 70% (missing livecodebench, wildbench, full math suite)

---

## Detailed Breakdown

### 1. lm-eval-harness (100% ✅)

**Cerebras REAP**:
```python
lm_eval_tasks = [
    "winogrande",
    "arc_challenge", 
    "arc_easy",
    "boolq",
    "hellaswag",
    "mmlu",
    "openbookqa",
    "rte",
]
```

**Our Implementation**:
```python
DEFAULT_TASKS = [
    "winogrande",
    "arc_challenge",
    "arc_easy", 
    "boolq",
    "hellaswag",
    "mmlu",
    "openbookqa",
    "rte",
]
```

**Status**: ✅ **IDENTICAL** - Same 8 default tasks

---

### 2. evalplus (100% ✅)

**Cerebras REAP**:
- Uses `evalplus.evaluate.evaluate()`
- Tasks: mbpp, humaneval
- Supports both base and plus (tested) versions

**Our Implementation**:
- Uses `evalplus.evaluate.evaluate()` (same)
- Tasks: mbpp, humaneval
- Basic integration in `eval_lm_harness.py`

**Status**: ✅ **FUNCTIONAL** - Same capabilities

---

### 3. livecodebench (0% ❌)

**Cerebras REAP**:
```python
if eval_args.run_livecodebench:
    # LiveCodeBench evaluation
    # Tests on competitive programming problems
    # Time-bounded coding challenges
```

**Our Implementation**:
- ❌ Not implemented
- No import or function for livecodebench

**What is LiveCodeBench?**
- Benchmark for competitive programming
- Time-bounded coding problems
- Similar to Codeforces/LeetCode contests
- Tests reasoning + coding under constraints

**Implementation Effort**: Medium
- Add `livecodebench` dependency
- Create evaluation wrapper
- ~50 lines of code

---

### 4. wildbench (0% ❌)

**Cerebras REAP**:
```python
if eval_args.run_wildbench:
    # WildBench evaluation
    # Uses HELM framework
    # Tests on wild, real-world tasks
    run_entries = [f"wildbench:subset=v2,model={original_model}"]
```

**Our Implementation**:
- ❌ Not implemented
- No wildbench support

**What is WildBench?**
- Real-world task benchmark
- Uses HELM (Holistic Evaluation of Language Models) framework
- Tests practical capabilities
- Subset v2 focuses on challenging tasks

**Implementation Effort**: High
- Requires HELM framework setup
- Complex configuration
- ~100+ lines of code
- May need separate environment

---

### 5. Math Evaluation (30% ⚠️)

**Cerebras REAP**:
```python
if eval_args.run_math:
    # Uses evalscope framework
    tasks = [
        "gsm8k",           # Grade school math
        "math_500",        # MATH dataset (500 problems)
        # ... more via evalscope
    ]
```

**Our Implementation**:
```python
MATH_TASKS = [
    "gsm8k",      # Available via lm-eval
    "math_qa",    # Available via lm-eval
]
```

**Gap Analysis**:
- ✅ gsm8k: Available via lm-eval
- ❌ math_500: Requires evalscope or specific MATH implementation
- ❌ Other math benchmarks: Not implemented

**What is evalscope?**
- Evaluation framework for Chinese/English benchmarks
- Includes comprehensive math suite
- MATH dataset (competition math problems)

**Implementation Effort**: Medium
- Can add gsm8k to lm-eval tasks (easy)
- For full MATH: use lm-eval's `hendrycks_math` or evalscope
- ~30 lines of code

---

### 6. Server Backend (Different Approach)

**Cerebras REAP**:
```python
# Uses vLLM with expert parallelism
server_command = [
    "vllm",
    "serve",
    model_name,
    "--enable-expert-parallel",  # Key feature
    "--tensor-parallel-size", str(num_gpus),
]
```

**Our Implementation**:
```python
# Uses SGLang
server_cmd = [
    sys.executable,
    "-m",
    "sglang.launch_server",
    "--model-path", str(model_path),
]
```

**Comparison**:

| Feature | vLLM (Cerebras) | SGLang (Ours) |
|---------|-----------------|---------------|
| Expert Parallelism | ✅ Native | ❌ Not supported |
| Tensor Parallelism | ✅ | ✅ |
| Performance | Optimized for MoE | General |
| Setup | More complex | Simpler |

**Impact**: 
- For single-GPU: No difference
- For multi-GPU MoE: Cerebras has advantage with expert parallelism

---

### 7. Additional Features in Cerebras REAP

#### Sampling Configuration
```python
# Cerebras supports non-greedy decoding
if not eval_args.greedy:
    override_generation_config = {
        "temperature": eval_args.temperature,
        "top_p": eval_args.top_p,
        "top_k": eval_args.top_k,
        "min_p": eval_args.min_p,
    }
```

**Our Implementation**: Basic greedy-only support

#### Model Name Mapping
```python
# Cerebras has explicit model name mapping
original_model_name_map = {
    "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Llama-4-Scout-17B-16E-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    # ... etc
}
```

**Our Implementation**: Direct model path usage

---

## Missing Evaluations Priority

### High Priority (Add Soon)

| Evaluation | Why | Effort |
|------------|-----|--------|
| **gsm8k** | Standard math benchmark, easy to add | 5 min |
| **math_500** | Competition math, good for reasoning | 30 min |

### Medium Priority (Add Eventually)

| Evaluation | Why | Effort |
|------------|-----|--------|
| **livecodebench** | Competitive programming, unique capability | 2-4 hours |

### Lower Priority (Nice to Have)

| Evaluation | Why | Effort |
|------------|-----|--------|
| **wildbench** | Real-world tasks, complex setup | 1-2 days |
| **Full evalscope** | Chinese benchmarks, niche use case | 1 day |

---

## Quick Implementation Guide

### Add gsm8k (5 minutes)

```python
# In eval_lm_harness.py, add to DEFAULT_TASKS:
DEFAULT_TASKS = [
    # ... existing tasks ...
    "gsm8k",  # Add this
]
```

Or use via lm-eval directly:
```bash
python -m examples.reap.eval_lm_harness \
    --recipe results/reap/.../pruning_recipe.json \
    --tasks gsm8k
```

### Add math_500 (30 minutes)

```python
# Add to eval_lm_harness.py
MATH_TASKS = [
    "gsm8k",
    "math_500",  # Requires custom implementation or evalscope
]

# Or use lm-eval's hendrycks_math
MATH_TASKS = [
    "hendrycks_math_algebra",
    "hendrycks_math_counting",
    # ... etc
]
```

### Add livecodebench (2-4 hours)

```python
# New file: eval_livecodebench.py
from livecodebench.eval import evaluate

def run_livecodebench(model_path, port=30000):
    """Run LiveCodeBench evaluation."""
    # 1. Start server
    # 2. Run evaluation
    # 3. Parse results
    pass
```

Dependencies:
```bash
pip install livecodebench
```

### Add wildbench (1-2 days)

This requires HELM framework setup which is more complex:

```bash
# Install HELM
pip install crfm-helm

# Setup wildbench
# ... configuration ...
```

See: https://github.com/stanford-crfm/helm

---

## Recommendations

### Immediate (Do Now)
1. **Add gsm8k** - One line change, high value
2. **Document math_500 path** - Show users how to add via lm-eval

### Short Term (This Week)
3. **Add livecodebench** - Unique capability, medium effort
4. **Add sampling support** - Temperature, top_p for non-greedy eval

### Long Term (If Needed)
5. **Add wildbench** - Only if real-world evaluation is critical
6. **Consider vLLM backend** - For expert parallelism on multi-GPU

---

## Current vs Target

```
Current Coverage: 70%
├─ lm-eval:        100% ✅
├─ evalplus:       100% ✅
├─ math:            30% ⚠️ (gsm8k only)
├─ livecodebench:    0% ❌
└─ wildbench:        0% ❌

With gsm8k + math_500: 80%
With +livecodebench:   90%
With +wildbench:      100%
```

**Conclusion**: Add gsm8k immediately (5 min). Consider livecodebench for unique competitive programming evaluation. Wildbench is lowest priority due to complex setup.
