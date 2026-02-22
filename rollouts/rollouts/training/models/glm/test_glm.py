"""Test GLM model implementation.

Run with:
    cd /Users/chiraagbalu/research/rollouts
    python -m rollouts.training.models.glm.test_glm
"""

from __future__ import annotations

import torch


def test_glm_debug_model() -> None:
    """Test GLM debug model forward pass."""
    from .args import GLM_DEBUG
    from .model import GLMModel

    print("Testing GLM debug model...")
    print(f"Config: {GLM_DEBUG}")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GLMModel(GLM_DEBUG)
    model.init_weights()
    model = model.to(device)

    print(f"Model created on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, GLM_DEBUG.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, GLM_DEBUG.vocab_size), (
        f"Expected {(batch_size, seq_len, GLM_DEBUG.vocab_size)}, got {logits.shape}"
    )

    print("Forward pass: OK")

    # Test backward pass
    model.train()
    input_ids = torch.randint(0, GLM_DEBUG.vocab_size, (batch_size, seq_len), device=device)
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()

    print(f"Backward pass: OK (loss={loss.item():.4f})")

    # Check gradients exist
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients computed"
    print("Gradients: OK")

    print("\nAll tests passed!")


def test_torchtitan_registration() -> None:
    """Test that GLM is registered with torchtitan."""
    # Import GLM module to trigger registration
    from torchtitan.protocols.train_spec import get_train_spec

    from rollouts.training.models import glm  # noqa: F401

    print("Testing torchtitan registration...")

    spec = get_train_spec("glm")
    print(f"TrainSpec: {spec}")
    print(f"Model class: {spec.model_cls}")
    print(f"Available sizes: {list(spec.model_args.keys())}")

    assert "4.7-flash" in spec.model_args, "GLM-4.7-Flash not registered"
    assert "5" in spec.model_args, "GLM-5 not registered"

    print("\nTorchtitan registration: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("GLM Model Tests")
    print("=" * 60)

    test_glm_debug_model()
    print()

    try:
        test_torchtitan_registration()
    except ImportError as e:
        print(f"Skipping torchtitan registration test (not installed): {e}")
