# Functional Model Extractor

> **Idea:** Automatically convert HuggingFace model classes into single-file PyTorch functional code.

## The Problem

HF models are 1000s of lines across 15+ files:
- Deep inheritance hierarchies
- Config objects everywhere
- Conditional branches for features we don't use
- Abstraction layers that obscure what's actually happening

```python
# What we get from HF
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
# 15 files, 5000+ lines, impossible to understand
```

## The Solution

A tool that:
1. **Traces** the actual execution path with real inputs
2. **Extracts** the weights and ops that actually run
3. **Rewrites** as flat functional PyTorch
4. **Verifies** with `torch.allclose` at each layer

```python
# What we want: ~150 lines, single file
def qwen_forward(x: Tensor, weights: dict, kv_cache: Tensor, positions: Tensor) -> Tensor:
    h = weights['embed'][x]
    for i in range(24):
        h = layer_forward(h, weights[f'layer.{i}'], kv_cache[i], positions)
    return F.linear(h, weights['lm_head'])
```

## How It Would Work

### Step 1: Trace Execution

```python
# Run model with coverage/tracing
model = AutoModelForCausalLM.from_pretrained(path)
input_ids = torch.tensor([[1, 2, 3, 4]])

with torch.no_grad():
    # Capture which branches execute, which ops run
    trace = trace_execution(model, input_ids)
```

### Step 2: Extract Dead Code

```python
# Find branches that never execute for inference
# - Training-only paths
# - Unused config options (use_cache=False paths, etc.)
# - Framework compatibility shims

live_ops = analyze_trace(trace)
# Result: Just the ops that actually matter
```

### Step 3: LLM Rewrite

```python
# Feed to LLM (Codex, Grok, Claude)
prompt = f"""
Rewrite this PyTorch model as a single functional forward pass.

Original traced ops:
{live_ops}

Original weights:
{list(model.state_dict().keys())}

Requirements:
- Single function, no classes
- Pure PyTorch functional (F.linear, etc.)
- ~150 lines max
- Preserve exact numerical behavior
"""

functional_code = llm.generate(prompt)
```

### Step 4: Verify Incrementally

```python
# Verify layer-by-layer with torch.allclose
def verify_functional(original_model, functional_fn, weights, test_inputs):
    # Hook into original model at each layer
    original_intermediates = capture_intermediates(original_model, test_inputs)

    # Run functional version, capture intermediates
    functional_intermediates = run_with_intermediates(functional_fn, weights, test_inputs)

    # Compare each layer
    for name, (orig, func) in zip_intermediates(original_intermediates, functional_intermediates):
        assert torch.allclose(orig, func, rtol=1e-5), f"Mismatch at {name}"

    print("✓ All layers match")
```

## The Vision (Original Quotes)

> "you can automate making it good - a tool where you run python code coverage and have an LLM rerun it with pytorch all close through all layers. eventual end goal being a single pytorch functional which will be around 150 lines. can easily do this with codex oob or with grok fast"

> "I can't believe someone still hasn't built an RL trained LLM that uses code coverage to take some command and reduce the amount of branches and complexity for the code and output some non-dogshit functional facsimile. basically a unix philosophy program generator given a program"

> "run a branch detector on this researchers class using python code. then, rewrite all of it in pytorch functional in a single file, incrementally, using torch.close to make sure you haven't made any mistakes"

**The tool would be:**
```bash
# Input: Any Python code with classes/branches
$ functional-extract model.py --entry forward --verify

# Output: Single functional equivalent
# - Traces actual execution
# - Removes dead branches
# - Flattens class hierarchy
# - Verifies with test inputs
```

**Could be RL-trained to:**
- Minimize LOC while preserving behavior
- Maximize `torch.compile` compatibility
- Prefer explicit over implicit

## Why This Matters for nano-inference

1. **Performance:** Flat functional code compiles better
2. **Hackability:** 150 lines vs 5000 lines
3. **TP Integration:** Easier to insert `all_reduce` when you can see the whole forward pass
4. **Debugging:** No abstraction layers hiding what's happening

## Implementation Path

### Phase 1: Manual ✅ DONE
Working implementation in `tools/functional_extractor/`:
- `qwen_functional.py:1-320` - Qwen2.5-0.5B forward pass, numerically identical to HF
- `tools.py` - `read_module_source()`, `get_weight_info()`, `capture_intermediate()`
- `verify.py` - GPU verification via bifrost/broker
- `test_qwen.py`, `debug_qwen.py` - Test and debug scripts

Key bugs found during manual implementation:
- RMSNorm: must multiply weight AFTER casting back to input dtype (`qwen_functional.py:64`)
- Attention: must use `F.scaled_dot_product_attention` to match HF's SDPA (`qwen_functional.py:223-230`)

### Phase 2: Semi-Automated (Next)
- Build Environment with tools as reward signals
- `verify_component()` as the critical tool - tells agent if code is correct
- LLM iterates on functional code using verification feedback

### Phase 3: Fully Automated (Eventually)
- RL-trained model that:
  - Reads class-based code
  - Outputs minimal functional equivalent
  - Self-verifies with `torch.allclose`

## Related Work

- `torch.fx` - Traces PyTorch models to graph IR
- `torch.compile` - JIT compiles models (but keeps abstraction)
- `torch.export` - Exports to flat representation
- Grok Fast - Shows ~150 line functional Llama is possible

## Notes

This is a tool idea, not blocking for nano-inference. Capturing here for future reference.

The key insight: **You don't need to understand the code - just trace it and replicate the tensor ops.**
