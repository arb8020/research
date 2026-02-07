"""Training mode - forward + backward with loss."""
import torch

def run(model, tokenizer):
    """Forward with labels, compute loss, backward."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    # Labels are shifted input_ids (standard causal LM)
    labels = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])

    # Need gradients for backward
    model.train()

    output = model(input_ids, labels=labels)
    loss = output.loss

    # Backward pass
    loss.backward()

    # Verify gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            break  # Just check one
