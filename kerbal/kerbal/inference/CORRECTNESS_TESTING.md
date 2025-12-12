# Inference Engine Correctness Testing

How to verify a custom inference engine produces correct outputs.

## Reference Implementation

**HuggingFace Transformers is the ground truth**, not vLLM or SGLang.

Both vLLM and SGLang verify against Transformers. They can produce different outputs from each other due to:
- Different attention backends (FlashInfer vs FlashAttention)
- Different numerical precision optimizations
- Different default sampling parameters (e.g., `top_k`: Transformers=50, vLLM=-1)

## Testing Levels

### Level 1: Exact Match (Greedy Decoding)

With `temperature=0` and identical sampling params, output tokens should exactly match Transformers.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# Greedy decoding (temperature=0 equivalent)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,  # Greedy
        top_k=50,         # Match Transformers default
        top_p=1.0,
    )

reference_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Reference: {reference_text}")
```

Your engine's output should exactly match `reference_text`.

### Level 2: Logprobs Similarity

When exact match isn't feasible (due to numerical precision), verify logprobs are close:

```python
def check_logprobs_close(
    your_logprobs: list[float],
    reference_logprobs: list[float],
    your_top_tokens: list[list[int]],
    reference_top_tokens: list[list[int]],
    rtol: float = 1e-2,
    top_k: int = 5,
) -> bool:
    """
    Verify logprobs similarity following vLLM's approach.

    Checks:
    1. Your top token is in reference's top-k
    2. Reference's top token is in your top-k
    3. Logprob values are within relative tolerance
    """
    for i, (your_lp, ref_lp) in enumerate(zip(your_logprobs, reference_logprobs)):
        # Check top token overlap
        your_top = set(your_top_tokens[i][:top_k])
        ref_top = set(reference_top_tokens[i][:top_k])

        if your_top_tokens[i][0] not in ref_top:
            print(f"Token {i}: Your top token not in reference top-{top_k}")
            return False

        if reference_top_tokens[i][0] not in your_top:
            print(f"Token {i}: Reference top token not in your top-{top_k}")
            return False

        # Check logprob values (skip None for first token)
        if your_lp is not None and ref_lp is not None:
            if abs(your_lp - ref_lp) > abs(ref_lp * rtol):
                print(f"Token {i}: Logprob diff too large: {your_lp} vs {ref_lp}")
                return False

    return True
```

### Level 3: Output Quality (Relaxed)

For heavily optimized engines, at minimum verify:
- Output is valid text (not garbage)
- Output is semantically reasonable
- No crashes or hangs

## Getting Reference Logprobs from Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_reference_logprobs(model_name: str, prompt: str, max_tokens: int = 10):
    """Get token-by-token logprobs from Transformers."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Generate with output scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Extract logprobs
    generated_ids = outputs.sequences[0, input_len:]
    scores = outputs.scores  # List of (vocab_size,) tensors

    logprobs = []
    top_tokens = []

    for i, (token_id, score) in enumerate(zip(generated_ids, scores)):
        log_probs = torch.log_softmax(score[0], dim=-1)

        # Get logprob of generated token
        token_logprob = log_probs[token_id].item()
        logprobs.append(token_logprob)

        # Get top-k tokens
        top_k_values, top_k_indices = torch.topk(log_probs, k=10)
        top_tokens.append(top_k_indices.tolist())

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "text": generated_text,
        "tokens": generated_ids.tolist(),
        "logprobs": logprobs,
        "top_tokens": top_tokens,
    }

# Usage
ref = get_reference_logprobs("Qwen/Qwen2.5-0.5B-Instruct", "The capital of France is")
print(f"Text: {ref['text']}")
print(f"Logprobs: {ref['logprobs']}")
```

## Testing Your Engine

```python
def test_engine_correctness(
    your_engine,
    model_name: str,
    prompts: list[str],
    max_tokens: int = 10,
):
    """Test your engine against Transformers reference."""
    results = []

    for prompt in prompts:
        # Get reference
        ref = get_reference_logprobs(model_name, prompt, max_tokens)

        # Get your output
        your_output = your_engine.generate(prompt, max_tokens=max_tokens)

        # Compare
        exact_match = (your_output["text"] == ref["text"])
        logprobs_close = check_logprobs_close(
            your_output["logprobs"],
            ref["logprobs"],
            your_output["top_tokens"],
            ref["top_tokens"],
        )

        results.append({
            "prompt": prompt,
            "exact_match": exact_match,
            "logprobs_close": logprobs_close,
            "your_text": your_output["text"],
            "ref_text": ref["text"],
        })

        status = "EXACT" if exact_match else ("CLOSE" if logprobs_close else "FAIL")
        print(f"[{status}] {prompt[:30]}...")

    return results

# Standard test prompts
TEST_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
    "The quick brown fox",
    "import torch\n",
]
```

## Common Pitfalls

1. **Default `top_k`**: Transformers defaults to 50, vLLM to -1 (unlimited). Always set explicitly.

2. **Padding**: Make sure padding is handled identically to Transformers.

3. **BOS/EOS tokens**: Some models auto-add these, others don't. Match Transformers behavior.

4. **Attention mask**: Ensure your attention mask matches Transformers exactly.

5. **Float precision**: FP16 vs BF16 vs FP32 can cause divergence. Test with FP32 first.

6. **KV cache numerical drift**: Long sequences may accumulate numerical errors. Test various lengths.

## Debugging Divergence

When outputs differ:

1. **Check first divergent token**: Print logprobs at that position from both
2. **Compare top-k at divergence point**: See if correct token is in your top-k
3. **Check attention patterns**: Visualize attention weights if possible
4. **Reduce precision**: Test with FP32 to rule out precision issues
5. **Test without optimizations**: Disable FlashAttention, KV cache, etc.

## CI Integration

For CI, run level 1 (exact match) on a small set of prompts:

```bash
pytest tests/test_correctness.py -k "exact_match" --model Qwen/Qwen2.5-0.5B-Instruct
```

For nightly, run level 2 (logprobs similarity) on larger prompt set:

```bash
pytest tests/test_correctness.py -k "logprobs" --model Qwen/Qwen2.5-7B-Instruct
```
