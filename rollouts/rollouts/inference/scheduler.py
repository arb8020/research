"""Pure function for scheduling.

Decides which sequences to prefill/decode each step.
No state, no side effects.
"""

from ..inference.types import (
    SchedulerConfig,
    SchedulerOutput,
    Sequence,
    SequenceStatus,
)


def schedule(
    waiting: list[Sequence],
    running: list[Sequence],
    num_free_blocks: int,
    config: SchedulerConfig,
) -> SchedulerOutput:
    """Pure function: decide what to run this step.

    Priority:
    1. Continue running sequences (decode)
    2. Start new sequences (prefill)

    Args:
        waiting: Sequences waiting to start
        running: Sequences currently generating
        num_free_blocks: Available KV cache blocks
        config: Scheduler limits

    Returns:
        SchedulerOutput with seq_ids to prefill/decode/preempt
    """
    assert config.max_batch_size > 0
    assert config.block_size > 0

    prefill_seqs: list[int] = []
    decode_seqs: list[int] = []
    preempted_seqs: list[int] = []

    blocks_available = num_free_blocks
    batch_tokens = 0

    # 1. Decode running sequences first (they're already allocated)
    for seq in running:
        assert seq.status == SequenceStatus.RUNNING, f"Expected RUNNING, got {seq.status}"

        if len(decode_seqs) >= config.max_batch_size:
            # Can't fit more in batch - preempt
            preempted_seqs.append(seq.seq_id)
            continue

        decode_seqs.append(seq.seq_id)
        batch_tokens += 1  # decode = 1 token per sequence

    # 2. Prefill waiting sequences with remaining capacity
    for seq in waiting:
        assert seq.status == SequenceStatus.WAITING, f"Expected WAITING, got {seq.status}"

        # Check batch size limit
        if len(prefill_seqs) + len(decode_seqs) >= config.max_batch_size:
            break

        # Check token budget
        seq_tokens = len(seq.token_ids)
        if batch_tokens + seq_tokens > config.max_tokens_per_batch:
            break

        # Check block availability (for KV cache)
        blocks_needed = (seq_tokens + config.block_size - 1) // config.block_size
        if blocks_needed > blocks_available:
            break

        prefill_seqs.append(seq.seq_id)
        blocks_available -= blocks_needed
        batch_tokens += seq_tokens

    return SchedulerOutput(
        prefill_seqs=tuple(prefill_seqs),
        decode_seqs=tuple(decode_seqs),
        preempted_seqs=tuple(preempted_seqs),
    )


def can_schedule_any(
    waiting: list[Sequence],
    running: list[Sequence],
    num_free_blocks: int,
    config: SchedulerConfig,
) -> bool:
    """Quick check if any work can be done.

    Useful for deciding whether to sleep or poll.
    """
    # Can always decode if there are running sequences
    if running:
        return True

    # Can prefill if there's a waiting sequence that fits
    if not waiting:
        return False

    first_seq = waiting[0]
    blocks_needed = (len(first_seq.token_ids) + config.block_size - 1) // config.block_size

    return blocks_needed <= num_free_blocks
