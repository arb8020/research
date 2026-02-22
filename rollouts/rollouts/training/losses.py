"""Loss functions for pretraining, SFT, and RL.

All loss functions return (loss, metrics) where:
- loss: Scalar tensor for backprop
- metrics: Dict of floats for logging

This module provides:
- pretrain_loss: Cross-entropy for pretraining
- sft_loss: Cross-entropy with masking for SFT
- grpo_loss: Simple GRPO (on-policy, no clipping)
- grpo_loss_clipped: Slime/Miles style with PPO clipping
- grpo_loss_masked: Prime-RL style with importance ratio masking
- compute_group_advantages: GRPO group-wise advantage normalization
"""

import torch
import torch.nn.functional as F

# Type alias for loss function return type
LossOutput = tuple[torch.Tensor, dict[str, float]]


# ============================================================================
# Advantage computation (GRPO and OPD)
# ============================================================================


def compute_group_advantages(
    rewards: list[float],
    group_indices: list[int],
    normalize_std: bool = False,
) -> torch.Tensor:
    """Compute GRPO-style group-normalized advantages.

    For each sample, advantage = reward - mean(rewards in same group).
    This is the "G" (Group) in GRPO - normalization within prompt groups.

    Args:
        rewards: List of rewards (one per sample)
        group_indices: List of group indices (samples with same index are from same prompt)
        normalize_std: Also divide by group std (like Miles grpo_std_normalization)

    Returns:
        Tensor of advantages [batch_size]

    Example:
        >>> rewards = [1.0, 0.0, 1.0, 1.0]  # 2 groups of 2 samples each
        >>> group_indices = [0, 0, 1, 1]
        >>> advantages = compute_group_advantages(rewards, group_indices)
        >>> # Group 0: mean=0.5, advantages=[0.5, -0.5]
        >>> # Group 1: mean=1.0, advantages=[0.0, 0.0]
    """
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    group_indices_tensor = torch.tensor(group_indices)

    # Get unique groups
    unique_groups = torch.unique(group_indices_tensor)

    advantages = torch.zeros_like(rewards_tensor)

    for group_idx in unique_groups:
        mask = group_indices_tensor == group_idx
        group_rewards = rewards_tensor[mask]

        # Compute group mean
        group_mean = group_rewards.mean()
        group_advantages = group_rewards - group_mean

        # Optional: normalize by std (Miles pattern)
        if normalize_std and len(group_rewards) > 1:
            group_std = group_rewards.std()
            if group_std > 1e-6:
                group_advantages = group_advantages / group_std

        advantages[mask] = group_advantages

    return advantages


def compute_opd_advantages(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute On-Policy Distillation per-token advantages.

    Following Miles: advantage = teacher_logprob - student_logprob

    This is a per-token advantage (dense signal) unlike GRPO which uses
    per-sequence advantages (sparse signal).

    Args:
        student_logprobs: Student's per-token log probs [batch, seq_len]
        teacher_logprobs: Teacher's per-token log probs [batch, seq_len]
        loss_mask: Which tokens to include [batch, seq_len]

    Returns:
        Per-token advantages [batch, seq_len]

    Example:
        >>> student_lp = torch.tensor([[-1.0, -2.0, -1.5]])
        >>> teacher_lp = torch.tensor([[-0.5, -1.0, -1.5]])
        >>> mask = torch.tensor([[0.0, 1.0, 1.0]])
        >>> adv = compute_opd_advantages(student_lp, teacher_lp, mask)
        >>> # adv = [[0.0, 1.0, 0.0]]  # teacher - student, masked
    """
    # Per-token advantage: encourage student to match teacher
    advantages = (teacher_logprobs - student_logprobs) * loss_mask
    return advantages


# ============================================================================
# Helper functions
# ============================================================================


def compute_entropy(logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute entropy of the policy distribution.

    Higher entropy = more exploration/randomness.
    Useful for detecting policy collapse.

    Args:
        logits: [batch, seq_len, vocab_size]
        mask: [batch, seq_len] optional mask

    Returns:
        Scalar mean entropy
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]

    if mask is not None:
        entropy = (entropy * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        entropy = entropy.mean()

    return entropy


def get_per_token_logprobs(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch, seq_len]
) -> torch.Tensor:
    """Extract log probabilities for target tokens.

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        labels: Target token ids [batch, seq_len]

    Returns:
        Log probabilities [batch, seq_len]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def get_seq_logprobs(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch, seq_len]
    mask: torch.Tensor,  # [batch, seq_len]
) -> torch.Tensor:
    """Compute sequence-level log probabilities (mean over tokens).

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        labels: Target token ids [batch, seq_len]
        mask: Which tokens to include [batch, seq_len]

    Returns:
        Sequence log probabilities [batch]
    """
    token_logprobs = get_per_token_logprobs(logits, labels)
    masked_logprobs = token_logprobs * mask
    return masked_logprobs.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


# ============================================================================
# Pretraining / SFT losses
# ============================================================================


def pretrain_loss(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    batch: dict[str, torch.Tensor],
) -> LossOutput:
    """Cross-entropy loss for pretraining.

    Simple next-token prediction on all tokens.

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        batch: Must contain "labels" [batch, seq_len]
               Labels with -100 are ignored (standard PyTorch convention)

    Returns:
        (loss, metrics) tuple
    """
    labels = batch["labels"]

    # Flatten for cross_entropy
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    # Compute perplexity for logging
    with torch.no_grad():
        perplexity = torch.exp(loss).item()

    metrics = {
        "loss": loss.item(),
        "perplexity": perplexity,
    }

    return loss, metrics


def sft_loss(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    batch: dict[str, torch.Tensor],
) -> LossOutput:
    """Cross-entropy loss for supervised fine-tuning.

    Like pretrain_loss but with explicit loss_mask for conversation masking
    (e.g., only train on assistant responses, not user messages).

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        batch: Must contain:
            - "labels" [batch, seq_len]: Target tokens (-100 = ignore)
            - "loss_mask" [batch, seq_len]: Optional, 1.0 = train, 0.0 = ignore

    Returns:
        (loss, metrics) tuple
    """
    labels = batch["labels"]
    loss_mask = batch.get("loss_mask")

    if loss_mask is None:
        # Fall back to pretrain_loss behavior
        return pretrain_loss(logits, batch)

    # Compute per-token cross entropy
    # Shift for next-token prediction if needed (depends on how data is prepared)
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)

    # Apply mask and compute mean
    masked_loss = ce_loss * loss_mask
    num_tokens = loss_mask.sum().clamp(min=1.0)
    loss = masked_loss.sum() / num_tokens

    # Compute perplexity for logging
    with torch.no_grad():
        perplexity = torch.exp(loss).item()

    metrics = {
        "loss": loss.item(),
        "perplexity": perplexity,
        "num_tokens": num_tokens.item(),
    }

    return loss, metrics


# ============================================================================
# RL losses
# ============================================================================


def grpo_loss(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    batch: dict[str, torch.Tensor],
) -> LossOutput:
    """Simple GRPO loss (on-policy, no clipping).

    GRPO = Group Relative Policy Optimization
    Loss = -mean(log_prob(sequence) * advantage)

    This is vanilla policy gradient with group-relative advantages.
    Simple and works well for on-policy training where rollouts are fresh.

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        batch: Must contain:
            - "labels" [batch, seq_len]: Target tokens
            - "loss_mask" [batch, seq_len]: Which tokens to train on
            - "advantages" [batch]: Per-sequence advantages

    Returns:
        (loss, metrics) tuple
    """
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    advantages = batch["advantages"]

    # Compute sequence-level log probs
    seq_logprobs = get_seq_logprobs(logits, labels, loss_mask)

    # Policy gradient: maximize log_prob * advantage
    # Negate because optimizer minimizes
    pg_loss = -(seq_logprobs * advantages).mean()

    # Compute entropy for logging (detects policy collapse)
    with torch.no_grad():
        entropy = compute_entropy(logits, loss_mask).item()
        avg_logprob = seq_logprobs.mean().item()
        avg_advantage = advantages.mean().item()

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy,
        "avg_logprob": avg_logprob,
        "avg_advantage": avg_advantage,
    }

    return pg_loss, metrics


def grpo_loss_clipped(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    batch: dict[str, torch.Tensor],
    clip_range: float = 0.2,
    entropy_coef: float = 0.0,
) -> LossOutput:
    """GRPO with PPO-style clipping (Slime/Miles pattern).

    Uses importance sampling ratio with clipping to prevent
    large policy updates when training on slightly stale rollouts.

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        batch: Must contain:
            - "labels" [batch, seq_len]: Target tokens
            - "loss_mask" [batch, seq_len]: Which tokens to train on
            - "advantages" [batch]: Per-sequence advantages
            - "old_logprobs" [batch]: Log probs from rollout policy
        clip_range: PPO clip range (default 0.2)
        entropy_coef: Entropy bonus coefficient (default 0.0)

    Returns:
        (loss, metrics) tuple
    """
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    advantages = batch["advantages"]
    old_logprobs = batch["old_logprobs"]

    # Compute current sequence-level log probs
    seq_logprobs = get_seq_logprobs(logits, labels, loss_mask)

    # Importance sampling ratio: π_θ / π_θ_old
    log_ratio = seq_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    # PPO clipped objective
    pg_loss1 = -ratio * advantages
    pg_loss2 = -torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # pessimistic bound

    # Optional entropy bonus
    entropy = compute_entropy(logits, loss_mask)
    loss = pg_loss - entropy_coef * entropy

    # Compute metrics
    with torch.no_grad():
        # Fraction of samples where clipping was active
        clipped = (ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)
        clipfrac = clipped.float().mean().item()

        # Approximate KL divergence
        approx_kl = ((ratio - 1) - log_ratio).mean().item()

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy.item(),
        "clipfrac": clipfrac,
        "approx_kl": approx_kl,
        "avg_ratio": ratio.mean().item(),
        "avg_logprob": seq_logprobs.mean().item(),
        "avg_advantage": advantages.mean().item(),
    }

    return loss, metrics


def grpo_loss_masked(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    batch: dict[str, torch.Tensor],
    ratio_low: float = 0.1,
    ratio_high: float = 10.0,
    kl_coef: float = 0.01,
) -> LossOutput:
    """GRPO with importance ratio masking (Prime-RL pattern).

    Instead of clipping, masks out samples where the importance ratio
    is too extreme (policy has drifted too far from rollout policy).

    Loss formula (following Prime-RL):
        coeff = ratio * (advantages - kl_coef * log_ratio)
        loss = -(coeff.detach() * seq_logprobs)[keep_mask].mean()

    The key insight is that we detach the coefficient so gradients flow
    only through seq_logprobs. This is the standard REINFORCE formula:
        ∇_θ J = E[∇_θ log π(a|s) * A]

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        batch: Must contain:
            - "labels" [batch, seq_len]: Target tokens
            - "loss_mask" [batch, seq_len]: Which tokens to train on
            - "advantages" [batch]: Per-sequence advantages
            - "old_logprobs" [batch]: Log probs from rollout policy
        ratio_low: Mask if ratio < this (default 0.1)
        ratio_high: Mask if ratio > this (default 10.0)
        kl_coef: KL penalty coefficient (default 0.01)

    Returns:
        (loss, metrics) tuple
    """
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    advantages = batch["advantages"]
    old_logprobs = batch["old_logprobs"]

    # Compute current sequence-level log probs
    seq_logprobs = get_seq_logprobs(logits, labels, loss_mask)

    # Importance sampling ratio
    log_ratio = seq_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    # Mask out extreme ratios (Prime-RL pattern)
    is_masked_low = ratio < ratio_low
    is_masked_high = ratio > ratio_high
    keep_mask = ~(is_masked_low | is_masked_high)

    # Policy gradient loss (Prime-RL formula)
    # coeff = ratio * (advantages - kl_coef * log_ratio)
    # Detach coeff so gradient flows only through seq_logprobs
    coeff = ratio * (advantages - kl_coef * log_ratio)

    if keep_mask.sum() > 0:
        # Standard REINFORCE: loss = -E[log_prob * advantage_coeff]
        pg_loss = -(coeff.detach() * seq_logprobs)[keep_mask].sum() / keep_mask.sum()
    else:
        # All samples masked - use small loss to avoid NaN
        pg_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = pg_loss

    # Compute metrics
    with torch.no_grad():
        entropy = compute_entropy(logits, loss_mask).item()

        # Mismatch KL: how much has policy drifted
        mismatch_kl = (torch.exp(log_ratio) - log_ratio - 1).mean().item()

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy,
        "masked_frac": (~keep_mask).float().mean().item(),
        "masked_low_frac": is_masked_low.float().mean().item(),
        "masked_high_frac": is_masked_high.float().mean().item(),
        "mismatch_kl": mismatch_kl,
        "avg_ratio": ratio.mean().item(),
        "avg_logprob": seq_logprobs.mean().item(),
        "avg_advantage": advantages.mean().item(),
        "avg_coeff": coeff.mean().item(),
    }

    return loss, metrics


def opd_loss(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    batch: dict[str, torch.Tensor],
) -> LossOutput:
    """On-Policy Distillation loss with per-token advantages.

    Unlike GRPO which uses per-sequence advantages from sparse rewards,
    OPD uses per-token advantages from teacher model log probs:
        advantage[t] = teacher_logprob[t] - student_logprob[t]

    This provides dense feedback (O(N) bits per episode) vs sparse
    (O(1) bit per episode).

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        batch: Must contain:
            - "labels" [batch, seq_len]: Target tokens
            - "loss_mask" [batch, seq_len]: Which tokens to train on
            - "teacher_logprobs" [batch, seq_len]: Teacher's per-token log probs
            - "advantages" [batch, seq_len]: Optional pre-computed per-token advantages
                If not provided, computed from teacher_logprobs

    Returns:
        (loss, metrics) tuple
    """
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    teacher_logprobs = batch["teacher_logprobs"]

    # Compute student log probs
    student_logprobs = get_per_token_logprobs(logits, labels)

    # Compute per-token advantages if not provided
    if "advantages" in batch:
        advantages = batch["advantages"]
    else:
        advantages = compute_opd_advantages(student_logprobs, teacher_logprobs, loss_mask)

    # Policy gradient with per-token advantages
    # Loss = -sum(log_prob * advantage) / sum(mask)
    # We detach advantages so gradients flow only through log_prob
    masked_pg = student_logprobs * advantages.detach() * loss_mask
    num_tokens = loss_mask.sum().clamp(min=1.0)
    pg_loss = -masked_pg.sum() / num_tokens

    # Compute metrics
    with torch.no_grad():
        entropy = compute_entropy(logits, loss_mask).item()
        avg_student_lp = (student_logprobs * loss_mask).sum().item() / num_tokens.item()
        avg_teacher_lp = (teacher_logprobs * loss_mask).sum().item() / num_tokens.item()
        avg_advantage = (advantages * loss_mask).sum().item() / num_tokens.item()

        # KL divergence: how far is student from teacher
        kl_div = (
            (student_logprobs - teacher_logprobs) * loss_mask
        ).sum().item() / num_tokens.item()

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy,
        "avg_student_logprob": avg_student_lp,
        "avg_teacher_logprob": avg_teacher_lp,
        "avg_advantage": avg_advantage,
        "kl_div": kl_div,
        "num_tokens": num_tokens.item(),
    }

    return pg_loss, metrics


# ============================================================================
# Legacy API (for backwards compatibility)
# ============================================================================


def ppo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """Legacy PPO loss (deprecated, use grpo_loss_clipped instead).

    Kept for backwards compatibility. Returns only loss tensor.
    """
    batch = {
        "labels": labels,
        "loss_mask": loss_mask,
        "advantages": advantages,
        "old_logprobs": old_log_probs,
    }
    loss, _ = grpo_loss_clipped(logits, batch, clip_range=clip_range)
    return loss
