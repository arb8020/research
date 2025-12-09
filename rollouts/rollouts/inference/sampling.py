"""Pure functions for sampling.

No state, no side effects. Logits in, tokens out.
"""

import torch
import torch.nn.functional as F


def sample_with_logprobs(
    logits: torch.Tensor,  # [batch, vocab]
    temperatures: torch.Tensor,  # [batch]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure function: sample tokens and compute their logprobs.

    Args:
        logits: Raw model output [batch, vocab_size]
        temperatures: Per-sequence temperature [batch]

    Returns:
        tokens: Sampled token IDs [batch]
        logprobs: Log probability of each sampled token [batch]
    """
    assert logits.dim() == 2, f"Expected 2D logits, got {logits.dim()}D"
    assert temperatures.dim() == 1, f"Expected 1D temperatures, got {temperatures.dim()}D"
    assert logits.size(0) == temperatures.size(0), "Batch size mismatch"

    batch_size = logits.size(0)

    # Handle temperature=0 (greedy) separately
    greedy_mask = temperatures == 0
    temps_safe = temperatures.clone()
    temps_safe[greedy_mask] = 1.0  # Avoid division by zero

    # Scale by temperature
    scaled_logits = logits / temps_safe.unsqueeze(-1)

    # Compute log probabilities (more numerically stable than softmax -> log)
    log_probs = F.log_softmax(scaled_logits, dim=-1)

    # Sample from distribution
    probs = log_probs.exp()
    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # For greedy, override with argmax
    if greedy_mask.any():
        greedy_tokens = logits.argmax(dim=-1)
        sampled_tokens = torch.where(greedy_mask, greedy_tokens, sampled_tokens)

    # Get logprob of sampled token
    sampled_logprobs = log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

    assert sampled_tokens.shape == (batch_size,)
    assert sampled_logprobs.shape == (batch_size,)

    return sampled_tokens, sampled_logprobs


def sample_greedy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure function: greedy sampling (temperature=0).

    Args:
        logits: Raw model output [batch, vocab_size]

    Returns:
        tokens: Argmax token IDs [batch]
        logprobs: Log probability of each token [batch]
    """
    assert logits.dim() == 2

    log_probs = F.log_softmax(logits, dim=-1)
    tokens = logits.argmax(dim=-1)
    logprobs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    return tokens, logprobs


def compute_logprobs_for_tokens(
    logits: torch.Tensor,  # [batch, seq_len, vocab]
    tokens: torch.Tensor,  # [batch, seq_len]
) -> torch.Tensor:  # [batch, seq_len]
    """Pure function: compute logprobs for given tokens.

    Useful for computing reference logprobs or verifying generation.

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        tokens: Token IDs to compute logprobs for [batch, seq_len]

    Returns:
        logprobs: Log probability of each token [batch, seq_len]
    """
    assert logits.dim() == 3
    assert tokens.dim() == 2
    assert logits.size(0) == tokens.size(0)
    assert logits.size(1) == tokens.size(1)

    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    return token_logprobs
