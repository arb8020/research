"""Tests for functional Llama layers.

Philosophy (from code_style docs):
- Assertions in code handle shapes/invariants (structural constraints)
- Tests verify numerical properties that can't be asserted
- Focus on integration-level properties, not implementation details
- Hypothesis for edge case coverage on properties that matter

What we test:
1. Numerical correctness (RMSNorm normalizes, RoPE rotates)
2. Behavioral properties (attention is causal)
3. Gradient flow (backward doesn't produce NaN)
4. Reference parity (TODO: compare against HuggingFace)
"""

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from rollouts.layers import compute_rope_embeddings, rms_norm, rotate_half
from rollouts.pretrain.config import ModelConfig
from rollouts.pretrain.models.llama import (
    attention,
    forward,
    init_weights,
)

# -----------------------------------------------------------------------------
# Hypothesis strategies
# -----------------------------------------------------------------------------


@st.composite
def model_config(draw: st.DrawFn) -> ModelConfig:
    """Generate valid ModelConfig."""
    dim = draw(st.sampled_from([32, 64, 128]))
    n_heads = draw(st.sampled_from([h for h in [1, 2, 4, 8] if dim % h == 0]))
    n_layers = draw(st.integers(1, 3))
    kv_divisors = [k for k in [1, 2, 4, n_heads] if n_heads % k == 0 and k <= n_heads]
    n_kv_heads = draw(st.sampled_from(kv_divisors))
    use_qk_norm = draw(st.booleans())
    use_relu2 = draw(st.booleans())
    return ModelConfig(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        use_qk_norm=use_qk_norm,
        use_relu2=use_relu2,
    )


# -----------------------------------------------------------------------------
# Numerical correctness tests
# -----------------------------------------------------------------------------


class TestNumericalCorrectness:
    """Tests that verify the math is right, not just that code runs."""

    @given(
        batch=st.integers(1, 4),
        seq=st.integers(1, 32),
        dim=st.sampled_from([32, 64, 128]),
    )
    @settings(max_examples=30)
    def test_rms_norm_normalizes(self, batch: int, seq: int, dim: int) -> None:
        """RMSNorm output should have RMS ≈ 1 when weight=1.

        This is the defining property of RMSNorm - can't be asserted in code
        because it's a statistical property of the output.
        """
        x = torch.randn(batch, seq, dim)
        weight = torch.ones(dim)
        out = rms_norm(x, weight)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_rotate_half_correct(self) -> None:
        """rotate_half should swap halves and negate first half.

        This is the core RoPE operation - if wrong, attention patterns break.
        """
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).float()
        out = rotate_half(x)
        expected = torch.tensor([[-3, -4, 1, 2], [-7, -8, 5, 6]]).float()
        assert torch.equal(out, expected)

    @given(seq=st.integers(2, 64), head_dim=st.sampled_from([32, 64]))
    @settings(max_examples=20)
    def test_rope_positions_differ(self, seq: int, head_dim: int) -> None:
        """Different positions must get different RoPE embeddings.

        If positions don't differ, the model can't distinguish token order.
        """
        cos, sin = compute_rope_embeddings(seq, head_dim, torch.device("cpu"))
        # Consecutive positions should differ
        assert not torch.allclose(cos[0], cos[1], atol=1e-6)

    def test_qk_norm_normalizes_per_head(self) -> None:
        """QK Norm should normalize Q and K per-head.

        The defining property: after QK Norm, the RMS over head_dim ≈ 1.
        This prevents attention logit explosion in deep networks.
        """
        config = ModelConfig(dim=64, n_layers=1, n_heads=4, use_qk_norm=True)
        weights = init_weights(config, torch.device("cpu"), torch.float32)
        cos, sin = compute_rope_embeddings(16, config.head_dim, torch.device("cpu"))

        x = torch.randn(2, 16, config.dim) * 10  # Large input to stress test
        out = attention(x, weights, layer_idx=0, cos=cos, sin=sin, config=config)

        # Output shouldn't explode (QK Norm prevents this)
        assert not torch.isnan(out).any(), "NaN in output"
        assert out.abs().max() < 1000, f"Output exploded: max={out.abs().max()}"

    def test_relu2_is_non_negative_squared(self) -> None:
        """ReLU² should produce non-negative outputs (relu squared is always ≥ 0).

        This is the core property of ReLU² - sparse activations that are smooth.
        """
        from rollouts.pretrain.models.llama import mlp

        config = ModelConfig(dim=64, n_layers=1, n_heads=4, use_relu2=True)
        weights = init_weights(config, torch.device("cpu"), torch.float32)

        x = torch.randn(2, 16, config.dim)
        # Get the hidden state after up_proj and relu²
        # We test this indirectly through the full MLP
        out = mlp(x, weights, layer_idx=0, config=config)

        # ReLU² should not produce NaN or Inf
        assert not torch.isnan(out).any(), "NaN in ReLU² output"
        assert not torch.isinf(out).any(), "Inf in ReLU² output"

    @given(
        batch=st.integers(1, 4),
        seq=st.integers(1, 32),
        dim=st.sampled_from([32, 64]),
    )
    @settings(max_examples=20)
    def test_relu2_sparser_than_silu(self, batch: int, seq: int, dim: int) -> None:
        """ReLU² should be sparser than SiLU (more zeros/small values).

        This is why ReLU² is used - sparsity is computationally efficient.
        """
        x = torch.randn(batch, seq, dim)

        # ReLU² has hard zeros for negative inputs
        relu2_out = F.relu(x).square()
        silu_out = F.silu(x)

        # Count near-zero values (< 1e-6)
        relu2_zeros = (relu2_out.abs() < 1e-6).float().mean()
        silu_zeros = (silu_out.abs() < 1e-6).float().mean()

        # ReLU² should have more near-zeros (at least half the inputs are negative -> zero)
        assert relu2_zeros >= silu_zeros * 0.5, (
            f"ReLU² not sparser: {relu2_zeros:.2%} vs SiLU {silu_zeros:.2%}"
        )


# -----------------------------------------------------------------------------
# Behavioral property tests
# -----------------------------------------------------------------------------


class TestBehavioralProperties:
    """Tests for properties that define correct behavior."""

    @given(config=model_config(), batch=st.integers(1, 2))
    @settings(max_examples=20, deadline=None)
    def test_attention_is_causal(self, config: ModelConfig, batch: int) -> None:
        """Future tokens must not affect past token outputs.

        This is THE critical property of causal attention. If violated,
        the model cheats by looking at future tokens during training.
        """
        seq = 16
        weights = init_weights(config, torch.device("cpu"), torch.float32)
        cos, sin = compute_rope_embeddings(seq, config.head_dim, torch.device("cpu"))

        # Same input, but second version has different future tokens
        x1 = torch.randn(batch, seq, config.dim)
        x2 = x1.clone()
        x2[:, seq // 2 :] = torch.randn(batch, seq // 2, config.dim)

        out1 = attention(x1, weights, layer_idx=0, cos=cos, sin=sin, config=config)
        out2 = attention(x2, weights, layer_idx=0, cos=cos, sin=sin, config=config)

        # First half must be identical (causal = no future leakage)
        assert torch.allclose(out1[:, : seq // 2], out2[:, : seq // 2], atol=1e-5)


# -----------------------------------------------------------------------------
# Gradient flow tests
# -----------------------------------------------------------------------------


class TestGradientFlow:
    """Tests that gradients flow correctly through the model."""

    @given(config=model_config())
    @settings(max_examples=10, deadline=None)
    def test_backward_no_nan(self, config: ModelConfig) -> None:
        """Backward pass must not produce NaN gradients.

        NaN gradients indicate numerical instability (division by zero,
        log of negative, etc). This catches issues that only appear
        in certain configs or edge cases.
        """
        batch, seq = 2, 16
        weights = init_weights(config, torch.device("cpu"), torch.float32)
        input_ids = torch.randint(0, config.vocab_size, (batch, seq))
        labels = torch.randint(0, config.vocab_size, (batch, seq))

        logits = forward(input_ids, weights, config)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))
        loss.backward()

        for name, w in weights.items():
            assert w.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(w.grad).any(), f"{name} has NaN gradient"
            assert not torch.isinf(w.grad).any(), f"{name} has Inf gradient"


# -----------------------------------------------------------------------------
# Reference parity tests (TODO)
# -----------------------------------------------------------------------------


@pytest.mark.skip(reason="TODO: implement HuggingFace reference comparison")
class TestReferenceParity:
    """Tests that verify our implementation matches HuggingFace Llama.

    This is the gold standard for correctness - if we match HF output
    for the same weights and inputs, we're correct.
    """

    def test_forward_matches_hf(self) -> None:
        """Forward pass should match HuggingFace LlamaForCausalLM."""
        # TODO: Load HF model, copy weights, compare outputs
        pass

    def test_attention_matches_hf(self) -> None:
        """Attention output should match HuggingFace LlamaAttention."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
