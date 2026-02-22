"""Overlap scheduling for hiding CPU latency.

The key insight is that while the GPU runs batch N, the CPU can:
1. Process results from batch N-1
2. Prepare inputs for batch N+1

This requires two CUDA streams:
- scheduler_stream: CPU-side work (memory allocation, scheduling)
- engine_stream: GPU compute (model forward)

Timeline:
    CPU: [schedule N] [process N-1, schedule N+1] [process N, schedule N+2] ...
    GPU:              [forward N]                  [forward N+1]            ...

State lives in caller-provided dict to keep this module stateless.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .attention.backend import AttentionMetadata
from .core import Batch


@dataclass
class ForwardInput:
    """Input prepared for a forward pass."""

    batch: Batch
    attn_metadata: AttentionMetadata


@dataclass
class ForwardOutput:
    """Output from a forward pass."""

    next_tokens_gpu: Tensor  # On GPU
    next_tokens_cpu: Tensor  # Async copied to CPU
    copy_done: torch.cuda.Event  # Signals when CPU copy is done
    batch: Batch  # The batch that was processed
    logprobs: list[float | None]  # Per-token logprobs (for state update)


def create_forward_output(
    next_tokens_gpu: Tensor,
    stream: torch.cuda.Stream,
    batch: Batch,
    logprobs: list[float | None],
) -> ForwardOutput:
    """Create ForwardOutput with async CPU copy.

    Args:
        next_tokens_gpu: Next tokens on GPU
        stream: Stream to record copy event on
        batch: The batch that was processed
        logprobs: Per-token logprobs

    Returns:
        ForwardOutput with async CPU copy in progress
    """
    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)

    copy_done = torch.cuda.Event()
    copy_done.record(stream)

    return ForwardOutput(
        next_tokens_gpu=next_tokens_gpu,
        next_tokens_cpu=next_tokens_cpu,
        copy_done=copy_done,
        batch=batch,
        logprobs=logprobs,
    )


def get_or_create_streams(
    state: dict,
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """Get or create overlap streams from state dict.

    Args:
        state: Caller-provided state dict
        device: CUDA device

    Returns:
        (scheduler_stream, engine_stream)
    """
    if "scheduler_stream" not in state:
        state["scheduler_stream"] = torch.cuda.Stream(device=device)
        state["engine_stream"] = torch.cuda.Stream(device=device)

    return state["scheduler_stream"], state["engine_stream"]


def overlap_step(
    state: dict,
    device: torch.device,
    last_output: ForwardOutput | None,
    forward_fn: Callable[[ForwardInput], ForwardOutput],
    schedule_fn: Callable[[], ForwardInput | None],
    process_fn: Callable[[ForwardOutput], None],
) -> ForwardOutput | None:
    """Run one overlap step.

    This function:
    1. Schedules the next batch (on scheduler_stream)
    2. Waits for engine to be ready
    3. Launches forward (on engine_stream)
    4. Processes the previous output (on scheduler_stream)

    Args:
        state: Caller-provided state dict for streams
        device: CUDA device
        last_output: Output from previous step (to process)
        forward_fn: Runs model forward, returns output
        schedule_fn: Schedules next batch, returns input or None
        process_fn: Processes forward output

    Returns:
        Output from this step's forward (or None if nothing to do)
    """
    scheduler_stream, engine_stream = get_or_create_streams(state, device)

    with torch.cuda.stream(scheduler_stream):
        forward_input = schedule_fn()

    if forward_input is None and last_output is None:
        return None

    current_output = None
    if forward_input is not None:
        with torch.cuda.stream(engine_stream):
            engine_stream.wait_stream(scheduler_stream)
            current_output = forward_fn(forward_input)

    if last_output is not None:
        with torch.cuda.stream(scheduler_stream):
            last_output.copy_done.synchronize()
            process_fn(last_output)

    return current_output


def run_overlap_loop(
    state: dict,
    device: torch.device,
    forward_fn: Callable[[ForwardInput], ForwardOutput],
    schedule_fn: Callable[[], ForwardInput | None],
    process_fn: Callable[[ForwardOutput], None],
) -> None:
    """Run the overlap loop until no more work.

    Processes all batches with maximum overlap.

    Args:
        state: Caller-provided state dict for streams
        device: CUDA device
        forward_fn: Runs model forward, returns output
        schedule_fn: Schedules next batch, returns input or None
        process_fn: Processes forward output
    """
    last_output = None

    while True:
        current_output = overlap_step(
            state, device, last_output, forward_fn, schedule_fn, process_fn
        )

        if current_output is None and last_output is None:
            break

        last_output = current_output

    if last_output is not None:
        last_output.copy_done.synchronize()
        process_fn(last_output)
