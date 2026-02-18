# Functional Extractor Eval

> Autonomous agent that converts HuggingFace model classes to minimal functional PyTorch.

## Problem Statement

HuggingFace model code is bloated:
- Qwen2Attention: 200 lines, 15+ branches
- Most branches never execute for a given config
- Inheritance hierarchies obscure what actually runs
- Impossible to understand, optimize, or add custom kernels

**Goal:** Given any HF model, produce a single-file functional equivalent that:
- Passes `torch.allclose` against original
- Is <200 lines
- Has zero runtime branches
- Is trivially fuseable/optimizable

## Why This Matters

Functional form is the prerequisite for:
1. **Custom Triton/CUDA kernels** — You can see exactly what ops to fuse
2. **Tensor parallelism** — You can see where matmuls are to shard them
3. **Quantization** — You control dtype at each op
4. **Understanding** — You can read 150 lines, not 5000

## The Algorithm

```
Input: model_name (e.g., "Qwen/Qwen2.5-0.5B"), test_inputs
Output: functional.py that passes torch.allclose

Loop:
  1. Run HF model with Python coverage tracing
  2. Extract only executed branches/lines
  3. LLM rewrites as flat functional code
  4. Verify: torch.allclose(hf_output, functional_output)
  5. If mismatch:
     - Run layer-by-layer comparison
     - Find divergence point
     - LLM fixes specific component
  6. Repeat until torch.allclose passes for all test cases
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Correctness | `torch.allclose(rtol=1e-5, atol=1e-5)` on full test suite |
| Compression | <200 lines for any model |
| Purity | Zero `if` statements in forward path |
| Autonomy | No human intervention after initial prompt |
| Speed | <30 minutes per model (vs ~2 hours manual) |

## Available Tools (in `tools/functional_extractor/`)

### Inspection Tools (`tools.py`)
```python
from tools.functional_extractor.tools import (
    list_modules,        # List all module paths in model
    read_module_source,  # Get source code of a module
    get_weight_info,     # Get weight shapes/dtypes
    capture_intermediate # Capture activation at a layer
)

# Example: See what's in layer 0
source = read_module_source(model, "model.layers.0.self_attn")
weights = get_weight_info(model, "layers.0")
hidden = capture_intermediate(model, "model.layers.0", input_ids)
```

### Debug Tools (`debug_toolkit.py`)
```python
from tools.functional_extractor.debug_toolkit import (
    DebugSession,
    capture_hf_internals,
    compare_layer_by_layer
)

# Find where functional diverges from HF
session = DebugSession(hf_model, functional_forward, weights)
divergence = session.find_divergence_layer()  # → "layer 3 attention"
```

### Test Harness (`test_template.py`)
```python
from tools.functional_extractor.test_template import FunctionalModelTestSuite

suite = FunctionalModelTestSuite(
    model_name="Qwen/Qwen2.5-0.5B",
    functional_forward=qwen_forward,
    weights=weights,
)
results = suite.run_all()  # → TestSuiteResult(passed=47, failed=0)
```

### Remote Verification (`verify.py`)
```python
# Verify on remote GPU via bifrost
python -m tools.functional_extractor.verify \
    configs/qwen3_0.6b.py \
    qwen3_functional.py \
    --keep-alive
```

## Missing Components (To Build)

### 1. Coverage-Guided Branch Extractor
```python
def extract_executed_code(model, test_input) -> str:
    """Run model with coverage, return only executed lines."""

    with coverage.Coverage() as cov:
        cov.start()
        model(test_input)
        cov.stop()

    # Parse coverage data
    # Extract only lines that executed
    # Return cleaned source
```

**Output:** Stripped-down source with dead branches removed.

### 2. Skeleton Generator
```python
def generate_skeleton(model_name: str) -> str:
    """Generate functional code skeleton from HF config."""

    config = AutoConfig.from_pretrained(model_name)

    return f'''
def {model_name.split("/")[-1].lower()}_forward(
    input_ids: Tensor,
    weights: dict[str, Tensor],
) -> Tensor:
    """
    Architecture:
        hidden_size: {config.hidden_size}
        num_layers: {config.num_hidden_layers}
        num_heads: {config.num_attention_heads}
        num_kv_heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}
        intermediate_size: {config.intermediate_size}
        vocab_size: {config.vocab_size}
    """
    # TODO: Implement
    pass
'''
```

**Output:** Template with correct shapes pre-filled.

### 3. Incremental Verifier
```python
def verify_up_to_layer(
    functional_fn,
    hf_model,
    weights,
    input_ids,
    up_to_layer: int
) -> VerifyResult:
    """Verify functional matches HF up through layer N."""

    # Capture HF hidden state after layer N
    hf_hidden = capture_intermediate(hf_model, f"model.layers.{up_to_layer}", input_ids)

    # Run functional and capture at same point
    func_hidden = functional_up_to_layer(functional_fn, weights, input_ids, up_to_layer)

    return VerifyResult(
        matches=torch.allclose(hf_hidden, func_hidden),
        max_diff=(hf_hidden - func_hidden).abs().max().item(),
        layer=up_to_layer
    )
```

**Enables:** Build-up verification (embed → layer 0 → layer 1 → ... → full)

### 4. Error Localizer
```python
def diagnose_mismatch(functional_out: Tensor, hf_out: Tensor) -> str:
    """Return human-readable diagnosis of where mismatch occurs."""

    diff = (functional_out - hf_out).abs()

    # Find worst positions
    flat_idx = diff.argmax().item()
    batch, seq, hidden = np.unravel_index(flat_idx, diff.shape)

    return f"""
Mismatch detected:
  - Max diff: {diff.max().item():.2e}
  - Worst position: batch={batch}, seq={seq}, hidden_dim={hidden}
  - HF value: {hf_out[batch, seq, hidden].item():.6f}
  - Functional value: {functional_out[batch, seq, hidden].item():.6f}

Likely causes:
  - If hidden_dim < 64: Check RoPE computation
  - If at seq boundaries: Check attention mask
  - If random positions: Check dtype handling (bf16 vs fp32)
"""
```

### 5. Agent Loop Orchestrator
```python
async def extract_functional(
    model_name: str,
    output_path: str,
    max_iterations: int = 20,
) -> ExtractResult:
    """
    Main agent loop for functional extraction.

    1. Generate skeleton
    2. Extract coverage-pruned HF source
    3. LLM writes initial implementation
    4. Verify incrementally (layer by layer)
    5. On failure: diagnose, LLM fixes, retry
    6. On success: run full test suite
    """

    # Load model and prepare test inputs
    model = AutoModelForCausalLM.from_pretrained(model_name)
    test_inputs = generate_test_inputs(model)

    # Generate starting point
    skeleton = generate_skeleton(model_name)
    hf_source = extract_executed_code(model, test_inputs[0])

    # Agent loop
    for iteration in range(max_iterations):
        # LLM generates/fixes functional code
        functional_code = await llm_generate(
            skeleton=skeleton,
            hf_source=hf_source,
            previous_error=last_error if iteration > 0 else None
        )

        # Write and import
        Path(output_path).write_text(functional_code)
        functional_fn = import_function(output_path)

        # Incremental verification
        for layer_idx in range(num_layers):
            result = verify_up_to_layer(functional_fn, model, weights, test_inputs, layer_idx)
            if not result.matches:
                last_error = diagnose_mismatch(...)
                break
        else:
            # All layers passed, run full test suite
            suite_result = run_full_test_suite(functional_fn, model, weights)
            if suite_result.all_passed:
                return ExtractResult(success=True, iterations=iteration+1)

    return ExtractResult(success=False, iterations=max_iterations)
```

## Test Models (Benchmark Suite)

| Model | Params | Reference | Notes |
|-------|--------|-----------|-------|
| Qwen/Qwen2.5-0.5B | 0.5B | `qwen_functional.py` ✅ | Done manually, baseline |
| Qwen/Qwen3-0.6B | 0.6B | `qwen3_functional.py` ✅ | Done manually |
| meta-llama/Llama-3.2-1B | 1B | `llama_functional.py` ✅ | Done manually |
| THUDM/glm-4-9b-chat | 9B | `glm4_moe_functional.py` ✅ | MoE architecture |
| google/gemma-2-2b | 2B | TODO | Different attention pattern |
| microsoft/phi-2 | 2.7B | TODO | Partial rotary |
| mistralai/Mistral-7B-v0.1 | 7B | TODO | Sliding window attention |
| deepseek-ai/DeepSeek-V2-Lite | 16B | TODO | MLA attention |

## Eval Metrics

```python
@dataclass
class EvalResult:
    model_name: str

    # Correctness
    torch_allclose_passed: bool
    max_numerical_diff: float
    test_cases_passed: int
    test_cases_total: int

    # Compression
    hf_lines_of_code: int
    functional_lines_of_code: int
    compression_ratio: float  # hf_loc / func_loc

    # Purity
    num_branches_in_output: int  # Should be 0
    num_classes_in_output: int   # Should be 0

    # Efficiency
    iterations_to_converge: int
    wall_time_seconds: float
    llm_tokens_used: int
```

## Baseline Results (Manual Extraction)

| Model | HF LOC | Functional LOC | Ratio | Time |
|-------|--------|----------------|-------|------|
| Qwen2.5-0.5B | ~3000 | 320 | 9.4x | ~2 hrs |
| Qwen3-0.6B | ~3500 | 470 | 7.4x | ~3 hrs |
| Llama-3.2-1B | ~2500 | 380 | 6.6x | ~2 hrs |
| GLM-4-MoE | ~5000 | 480 | 10.4x | ~4 hrs |

**Target for autonomous agent:** Same compression ratio, <30 min per model.

## Stretch Goals

### RL-Trained Simplifier
Train a model specifically for code simplification:
- Reward: `torch.allclose` pass + LOC reduction + branch elimination
- Could generalize beyond ML code to any Python

### torch.compile Optimization
Functional code should be `torch.compile`-friendly:
```python
functional_fn = torch.compile(qwen_forward)
# Should produce efficient fused kernels
```

Metric: Measure speedup from torch.compile on functional vs HF.

### Kernel Insertion Points
Annotate functional code with fusion opportunities:
```python
def attention(x, q_w, k_w, v_w, o_w, cos, sin):
    # FUSE_START: qkv_projection
    q = F.linear(x, q_w)
    k = F.linear(x, k_w)
    v = F.linear(x, v_w)
    # FUSE_END

    # FUSE_START: rope_attention
    q, k = apply_rope(q, k, cos, sin)
    out = F.scaled_dot_product_attention(q, k, v)
    # FUSE_END

    return F.linear(out, o_w)
```

## References

- `tools/functional_extractor/` — Existing tooling
- `docs/design/functional_model_extractor.md` — Original design doc
- Grok Fast — Shows ~150 line functional Llama is achievable
- NMOE repo — Production functional implementations

## Next Steps

1. **Build coverage extractor** — Core component for branch pruning
2. **Build skeleton generator** — Reduces LLM work
3. **Build incremental verifier** — Enables iterative debugging
4. **Wire up agent loop** — Orchestrate LLM + tools
5. **Run on benchmark suite** — Measure against manual baseline
6. **Iterate on prompts** — Optimize LLM instructions for this task
