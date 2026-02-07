"""KV cache mode - for vLLM-style inference with prefill + decode."""
import torch

def run(model, tokenizer):
    """Prefill then decode with KV cache."""
    # Prefill phase
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    with torch.no_grad():
        output = model(input_ids, use_cache=True)
        past_kv = output.past_key_values

        # Decode phase - multiple steps to exercise cache logic
        for _ in range(5):
            next_token = output.logits[:, -1:].argmax(dim=-1)
            output = model(
                next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = output.past_key_values
