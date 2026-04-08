"""On-Policy Distillation (OPD) support.

OPD trains a student model to match a teacher model's token-level predictions
on student-generated rollouts. This provides dense feedback (O(N) bits per episode)
compared to sparse RL rewards (O(1) bit per episode).

Key insight from Miles implementation:
- Teacher runs as a separate SGLang server
- After student generates rollout, query teacher for log probs on those tokens
- Use (teacher_logprob - student_logprob) as per-token advantages
- Feed into standard policy gradient loss

References:
- https://thinkingmachines.ai/blog/on-policy-distillation/
- /tmp/miles/examples/on_policy_distillation/on_policy_distillation.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .types import RowAttempt

logger = logging.getLogger(__name__)


async def compute_teacher_logprobs(
    teacher_url: str,
    tokens: list[int],
    timeout: float = 120.0,
) -> list[float]:
    """Compute teacher model's log probabilities for a token sequence.

    Queries the teacher SGLang server with max_new_tokens=0 to score
    the existing tokens without generating new ones.

    Args:
        teacher_url: Teacher SGLang server URL (e.g., "http://localhost:30001")
        tokens: Full token sequence (prompt + response)
        timeout: Request timeout in seconds

    Returns:
        List of per-token log probabilities (length = len(tokens) - 1,
        since first token has no log prob)

    Example:
        >>> tokens = [1, 2, 3, 4, 5]  # prompt + response
        >>> logprobs = await compute_teacher_logprobs("http://localhost:30001", tokens)
        >>> assert len(logprobs) == 4  # one less than tokens
    """
    # Build payload for scoring (no generation)
    payload = {
        "input_ids": tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,  # No generation, only scoring
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,  # Start from position 0
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        response = await client.post(f"{teacher_url}/generate", json=payload)
        response.raise_for_status()
        data = response.json()

    # Extract log probs from response
    # SGLang returns input_token_logprobs as [(logprob, token_id), ...]
    # First token has no logprob (it's the BOS or first input token)
    meta_info = data.get("meta_info", {})
    input_token_logprobs = meta_info.get("input_token_logprobs", [])

    # Skip first token (no logprob for it)
    logprobs = [item[0] for item in input_token_logprobs[1:]]

    return logprobs


async def compute_teacher_logprobs_batch(
    teacher_url: str,
    samples: list[RowAttempt],
    timeout: float = 120.0,
) -> None:
    """Compute teacher log probs for a batch of attempts in parallel.

    Modifies attempts in place, setting ``attempt.teacher_log_probs`` on the
    attached training sample.

    Args:
        teacher_url: Teacher SGLang server URL
        samples: List of attempt rows with tokens populated
        timeout: Request timeout per attempt

    Side effects:
        Sets ``attempt.teacher_log_probs`` for each attempt
    """
    import trio

    async def compute_for_sample(sample: RowAttempt) -> None:
        if not sample.tokens:
            logger.warning(
                f"Attempt {sample.id} has no tokens, skipping teacher logprob computation"
            )
            return

        try:
            teacher_logprobs = await compute_teacher_logprobs(
                teacher_url, sample.tokens, timeout=timeout
            )
            # Align with tokens: prepend 0.0 for first token (no logprob)
            # This matches the rollout_log_probs convention
            sample.teacher_log_probs = [0.0] + teacher_logprobs
        except Exception as e:
            logger.exception(f"Failed to compute teacher logprobs for attempt {sample.id}: {e}")
            sample.teacher_log_probs = None

    async with trio.open_nursery() as nursery:
        for sample in samples:
            nursery.start_soon(compute_for_sample, sample)


def compute_opd_advantages(
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    loss_mask: list[float],
) -> list[float]:
    """Compute OPD per-token advantages.

    Following Miles: advantage = teacher_logprob - student_logprob

    This encourages the student to increase probability where the teacher
    is confident.

    Args:
        student_logprobs: Student's per-token log probs
        teacher_logprobs: Teacher's per-token log probs
        loss_mask: Which tokens to include (1.0 = include, 0.0 = exclude)

    Returns:
        Per-token advantages (same length as inputs)
    """
    assert len(student_logprobs) == len(teacher_logprobs) == len(loss_mask), (
        f"Length mismatch: student={len(student_logprobs)}, "
        f"teacher={len(teacher_logprobs)}, mask={len(loss_mask)}"
    )

    advantages = []
    for s_lp, t_lp, mask in zip(student_logprobs, teacher_logprobs, loss_mask, strict=False):
        if mask > 0:
            # Teacher - student: positive means student should increase prob
            advantages.append(t_lp - s_lp)
        else:
            advantages.append(0.0)

    return advantages
