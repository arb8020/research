"""RL loss functions (pure).

GRPO (Group Relative Policy Optimization) - simplest RL loss.
Implements the loss function described in the GRPO paper.
"""

import torch
import torch.nn.functional as F


def grpo_loss(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch, seq_len]
    loss_mask: torch.Tensor,  # [batch, seq_len]
    advantages: torch.Tensor,  # [batch]
) -> torch.Tensor:
    """Compute GRPO loss (simplified policy gradient).

    GRPO = Group Relative Policy Optimization
    Loss = -mean(log_prob(sequence) * advantage)

    This is a simplified policy gradient where:
    - We maximize log probability of sequences with positive advantage
    - We minimize log probability of sequences with negative advantage

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        loss_mask: Token-level loss weights [batch, seq_len] (0.0 = ignore, 1.0 = train)
        advantages: Advantage estimates per sample [batch]

    Returns:
        Scalar loss

    Example:
        >>> logits = model(input_ids)  # [4, 32, 50000]
        >>> advantages = compute_advantages(rewards, baseline=0.5)  # [4]
        >>> loss = grpo_loss(logits, labels, loss_mask, advantages)
        >>> loss.backward()
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

    # Get log probs for target tokens
    batch_size, seq_len, vocab_size = logits.shape
    target_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),  # [batch, seq_len, 1]
    ).squeeze(-1)  # [batch, seq_len]

    # Apply loss mask and sum over sequence
    masked_log_probs = target_log_probs * loss_mask  # [batch, seq_len]

    # Average log prob per sequence (over non-masked tokens)
    seq_log_probs = masked_log_probs.sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)  # [batch]

    # GRPO: policy gradient weighted by advantages
    # Loss = -E[log_prob * advantage]
    # Negative because we want to maximize, but optimizer minimizes
    loss = -(seq_log_probs * advantages).mean()

    return loss


def ppo_loss(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch, seq_len]
    loss_mask: torch.Tensor,  # [batch, seq_len]
    advantages: torch.Tensor,  # [batch]
    old_log_probs: torch.Tensor,  # [batch]
    clip_range: float = 0.2,
) -> torch.Tensor:
    """Compute PPO clipped loss.

    PPO = Proximal Policy Optimization
    Uses clipping to prevent too-large policy updates.

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        loss_mask: Token-level loss weights [batch, seq_len]
        advantages: Advantage estimates per sample [batch]
        old_log_probs: Log probs from old policy [batch]
        clip_range: PPO clip range (typically 0.2)

    Returns:
        Scalar loss

    Note:
        This is a placeholder for future PPO implementation.
        For now, use GRPO which is simpler and doesn't require old_log_probs.
    """
    # Compute current log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Get log probs for target tokens
    batch_size, seq_len, vocab_size = logits.shape
    target_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Apply loss mask and average over sequence
    masked_log_probs = target_log_probs * loss_mask
    seq_log_probs = masked_log_probs.sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)

    # Compute ratio: π_θ / π_θ_old
    ratio = torch.exp(seq_log_probs - old_log_probs)

    # PPO clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages

    # Take minimum (pessimistic bound)
    loss = -torch.min(surr1, surr2).mean()

    return loss
