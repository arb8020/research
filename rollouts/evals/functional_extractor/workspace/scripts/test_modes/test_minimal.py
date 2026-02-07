"""Minimal inference mode - no KV cache, no training."""
import torch

def run(model, tokenizer):
    """Simple forward pass, explicitly no caching."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    with torch.no_grad():
        output = model(input_ids, use_cache=False)

    # Just verify we got logits
    assert output.logits.shape[0] == 1
