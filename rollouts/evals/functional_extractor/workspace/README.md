# Functional Model Extraction

Your task is to convert a HuggingFace model to minimal, functional PyTorch code.

## CRITICAL: How Success Is Measured

Your task is ONLY complete when you run:
```bash
python /workspace/scripts/verify.py /workspace/functional.py
```
And it prints **"PASS (max_diff=...)"**

This is the ONLY valid verification. Do NOT create your own tests. Do NOT assume success without seeing PASS.

## Goal

Write a single Python file `functional.py` that:
1. Contains a `forward(input_ids, weights)` function
2. Produces numerically identical output to the HuggingFace model
3. Uses only `torch` and `torch.nn.functional` (no classes, no HF imports)
4. Is as short as possible while remaining readable

## Model Info

See `model_info.json` for:
- Model name (e.g., "Qwen/Qwen2.5-0.5B")
- Architecture config (hidden_size, num_layers, etc.)
- Weight names and shapes

## Available Scripts

All scripts are in `/opt/eval_scripts/` (also symlinked at `/workspace/scripts/`).

```bash
# OFFICIAL VERIFICATION - run this to check if you succeeded
python /opt/eval_scripts/verify.py /workspace/functional.py
# Output: "PASS (max_diff=1.23e-6)" or "FAIL (max_diff=0.123)"

# Inspect HF model source code
python /opt/eval_scripts/inspect_model.py model.layers.0.self_attn
# Output: Source code of the attention module

# Capture layer activations
python /opt/eval_scripts/capture.py 0 --save layer0.pt
# Output: Saves HF model's layer 0 output to layer0.pt

# Compare your implementation at a specific layer
python /opt/eval_scripts/compare.py 0 /workspace/functional.py
# Output: Diff between HF and your implementation at layer 0

# Find where your implementation diverges from HF (debugging tool)
python /opt/eval_scripts/find_divergence.py /workspace/functional.py
# Output: Shows exactly which layer first diverges

# Trace the HF forward pass (advanced - for understanding execution flow)
python /opt/eval_scripts/trace_forward.py --verbose
# Output: Weights accessed, layer shapes, coverage info
```

## Reference

See `reference/` for an example of what good functional code looks like.

## Strategy

1. Start by reading `model_info.json` to understand the architecture
2. Use `scripts/inspect_model.py` to read the HF source for each component
3. Write `functional.py` incrementally:
   - Start with embeddings
   - Add one transformer layer
   - Verify with `scripts/compare.py 0 functional.py`
   - Add remaining layers
4. Run `python /workspace/scripts/verify.py /workspace/functional.py`
5. If it says FAIL, use `scripts/compare.py` to find where divergence starts
6. Keep iterating until verify.py says PASS

## Success Criteria

You are done when `python /workspace/scripts/verify.py /workspace/functional.py` outputs:
```
PASS (max_diff=...)
```

## Tips

- RMSNorm: Cast to float32 for variance, then back to input dtype before multiplying weight
- RoPE: Make sure you're computing frequencies the same way as HF
- Attention: Use `F.scaled_dot_product_attention` with `is_causal=True` for causal masking
- GQA: Remember to repeat KV heads to match query head count
