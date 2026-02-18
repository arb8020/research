"""Tests for Muon optimizer.

Key numerical properties to verify:
1. Polar Express produces orthogonal matrices
2. Muon updates have correct scaling
3. Weight decay is decoupled (AdamW-style)
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from rollouts.pretrain.optim import Muon, polar_express


class TestPolarExpress:
    """Tests for Polar Express orthogonalization."""

    @given(
        rows=st.integers(16, 128),
        cols=st.integers(16, 128),
    )
    @settings(max_examples=20)
    def test_output_is_approximately_orthogonal(self, rows: int, cols: int) -> None:
        """Polar Express should produce approximately orthogonal matrices.

        For tall matrices (M > N): X.T @ X ≈ I
        For wide matrices (M < N): X @ X.T ≈ I
        """
        g = torch.randn(rows, cols)
        out = polar_express(g, ns_steps=5)

        # Convert to float32 for stable comparison (polar_express outputs bf16)
        out = out.float()

        if rows > cols:
            # Tall: X.T @ X should be identity
            product = out.T @ out
            identity = torch.eye(cols, dtype=product.dtype)
        else:
            # Wide: X @ X.T should be identity
            product = out @ out.T
            identity = torch.eye(rows, dtype=product.dtype)

        # Allow some tolerance (bf16 and iterative approximation can be noisy)
        # The key property is that it's much more orthogonal than random
        diff = (product - identity).abs().max()
        assert diff < 0.3, f"Not orthogonal: max diff = {diff:.4f}"

    def test_preserves_shape(self) -> None:
        """Output shape must match input shape."""
        for shape in [(64, 32), (32, 64), (64, 64)]:
            g = torch.randn(*shape)
            out = polar_express(g)
            assert out.shape == g.shape

    def test_no_nan_on_zero_input(self) -> None:
        """Should handle zero input gracefully (division by norm)."""
        g = torch.zeros(32, 32)
        out = polar_express(g)
        assert not torch.isnan(out).any()


class TestMuon:
    """Tests for Muon optimizer behavior."""

    def test_step_updates_params(self) -> None:
        """Muon step should actually update parameters."""
        param = torch.nn.Parameter(torch.randn(64, 32))
        original = param.clone()

        optimizer = Muon([param], lr=0.02)

        # Fake gradient
        param.grad = torch.randn_like(param)
        optimizer.step()

        assert not torch.equal(param, original), "Param not updated"

    def test_weight_decay_shrinks_params(self) -> None:
        """Decoupled weight decay should shrink parameters toward zero."""
        param = torch.nn.Parameter(torch.ones(64, 32) * 10.0)
        original_norm = param.norm().item()

        optimizer = Muon([param], lr=0.02, weight_decay=0.1)

        # Zero gradient - only weight decay acts
        param.grad = torch.zeros_like(param)
        optimizer.step()

        new_norm = param.norm().item()
        assert new_norm < original_norm, "Weight decay didn't shrink params"

    def test_momentum_accumulates(self) -> None:
        """Momentum buffer should accumulate across steps."""
        param = torch.nn.Parameter(torch.randn(64, 32))
        optimizer = Muon([param], lr=0.02, momentum=0.95)

        # Same gradient multiple times
        for _ in range(5):
            param.grad = torch.ones_like(param)
            optimizer.step()

        # Check momentum buffer exists and has grown
        state = optimizer.state[param]
        assert "momentum_buffer" in state
        # With momentum=0.95 and 5 steps: converges to ~0.2 which is correct
        # lerp with (1-0.95)=0.05 from 0 toward 1
        assert state["momentum_buffer"].abs().mean() > 0.1

    @given(rows=st.integers(32, 128), cols=st.integers(32, 128))
    @settings(max_examples=10)
    def test_no_nan_gradients(self, rows: int, cols: int) -> None:
        """Muon should not produce NaN in parameter updates."""
        param = torch.nn.Parameter(torch.randn(rows, cols))
        optimizer = Muon([param], lr=0.02)

        param.grad = torch.randn_like(param)
        optimizer.step()

        assert not torch.isnan(param).any(), "NaN in parameters after step"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
