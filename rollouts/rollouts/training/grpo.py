"""GRPO Training Loop.

Shared training infrastructure for GRPO (Group Relative Policy Optimization).
Each task provides prompts, score_fn, and environment_cls - this module handles
the rest: SGLang server, training backend, rollout generation, gradient updates.

Usage:
    from rollouts.training.grpo import GRPOConfig, grpo_train

    config = GRPOConfig(model_name="Qwen/Qwen3-0.6B", num_steps=100)
    prompts = [{"messages": [...], "answer": "42"}, ...]

    def my_score_fn(sample):
        return Score(metrics=(Metric("correct", 1.0 if correct else 0.0, weight=1.0),))

    results = grpo_train(
        config=config,
        prompts=prompts,
        score_fn=my_score_fn,
        environment_cls=BasicEnvironment,
    )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from rollouts.dtypes import Environment, Score
    from rollouts.training.types import Sample

# ──────────────────────── Config ─────────────────────────────────────────────


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO training.

    Groups related settings into a flat, explicit config.
    """

    # Model
    model_name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"

    # Inference server
    inference_backend: str = "sglang"  # "sglang" or "vllm"
    inference_port: int = 30000
    inference_cuda_device_ids: tuple[int, ...] = (0,)
    mem_fraction: float = 0.7

    # Trainer
    trainer_cuda_device_ids: tuple[int, ...] = (0,)
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_minibatches: int = 8

    # Rollout generation
    batch_size: int = 8  # Unique prompts per step
    n_samples_per_prompt: int = 8  # Completions per prompt (the "G" in GRPO)
    max_seq_len: int = 1024
    max_tokens: int = 512
    temperature: float = 0.8
    max_turns: int = 1  # For multi-turn environments

    # TI/TO (Tokens-In/Tokens-Out) - avoids retokenization collapse
    # When True, uses token-level generation via /generate endpoint
    # and stores rollout logprobs for off-policy correction
    use_tito: bool = False

    # Trajectory strategy for multi-turn rollouts
    # - "interleaved": Full conversation as one sequence (efficient, prefix sharing)
    # - "branching": Each assistant turn is a separate sample (safer, mirrors deployment)
    trajectory_strategy: str = "interleaved"  # Literal["interleaved", "branching"]

    # Checkpoint loading (for SFT → RL pipeline)
    # If set, loads weights from this checkpoint before training.
    # Can be:
    #   - Path to HuggingFace-format directory (with config.json)
    #   - Path to pytorch checkpoint directory (with pytorch_model.bin)
    checkpoint_path: str | None = None

    # Training loop
    num_steps: int = 100
    log_every: int = 1
    checkpoint_every: int = 20  # Save to disk (for recovery/resuming)
    sync_weights_every: int = 1  # Sync to inference engine (for on-policy vs off-policy)

    # Output
    output_dir: str = "results/rl"
    experiment_name: str = "grpo"

    def save(self, path: Path | str) -> None:
        """Save config to JSON."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ──────────────────────── Training Function ──────────────────────────────────


def grpo_train(
    config: GRPOConfig,
    prompts: list[dict[str, Any]],
    score_fn: Callable[[Sample], Score],
    environment_cls: type[Environment],
    metadata_key: str | None = None,
) -> dict[str, Any]:
    """Run GRPO training.

    Args:
        config: Training configuration
        prompts: List of prompt dicts, each containing:
            - "messages": List of chat messages [{"role": "...", "content": "..."}]
            - Any metadata needed by score_fn (e.g., "answer", "expected_sorted")
        score_fn: Function (Sample) -> Score that computes reward
        environment_cls: Environment class (BasicEnvironment for no tools,
            CalculatorEnvironment for calculator, etc.)
        metadata_key: If set, extract this key from prompt dict to pass as metadata.
            If None, passes all non-"messages" keys as metadata.

    Returns:
        Dict with "metrics_history" list of per-step metrics

    Example:
        >>> from rollouts.training.grpo import GRPOConfig, grpo_train
        >>> from rollouts.environments.no_tools import BasicEnvironment
        >>>
        >>> config = GRPOConfig(model_name="Qwen/Qwen3-0.6B", num_steps=10)
        >>> prompts = [
        ...     {"messages": [{"role": "user", "content": "2+2=?"}], "answer": "4"},
        ... ]
        >>> results = grpo_train(config, prompts, my_score_fn, BasicEnvironment)
    """
    return trio.run(_grpo_train_async, config, prompts, score_fn, environment_cls, metadata_key)


# ──────────────────────── Training Helpers ────────────────────────────────────


def _setup_output_dir(config: GRPOConfig) -> tuple[Path, str]:
    """Setup output directory and run name.

    Returns:
        Tuple of (output_dir, run_name)
    """
    import os
    from datetime import datetime, timezone

    run_name = os.environ.get("ROLLOUTS_RUN_NAME")
    if run_name:
        output_dir = Path(config.output_dir) / run_name
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_name = f"{config.experiment_name}_{timestamp}"
        output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, run_name


def _create_inference_engine(
    config: GRPOConfig, output_dir: Path
) -> Any:  # SGLangEngine | VLLMEngine
    """Create and configure inference engine."""
    from rollouts.training.weight_sync import SGLangEngine, VLLMEngine

    if config.inference_backend == "sglang":
        return SGLangEngine(
            model_name=config.model_name,
            port=config.inference_port,
            cuda_device_ids=config.inference_cuda_device_ids,
            output_dir=output_dir,
            dtype=config.dtype,
            mem_fraction=config.mem_fraction,
        )
    elif config.inference_backend == "vllm":
        return VLLMEngine(
            model_name=config.model_name,
            port=config.inference_port,
            cuda_device_ids=config.inference_cuda_device_ids,
            output_dir=output_dir,
            dtype=config.dtype,
            gpu_memory_utilization=config.mem_fraction,
        )
    else:
        msg = f"Unknown inference backend: {config.inference_backend}"
        raise ValueError(msg)


def _setup_training_backend(
    config: GRPOConfig, output_dir: Path, inference_engine: Any
) -> tuple[Any, Any, Any]:  # (backend, tokenizer, endpoint)
    """Setup training backend, tokenizer, and endpoint.

    Returns:
        Tuple of (backend, tokenizer, endpoint)
    """
    from transformers import AutoTokenizer

    from rollouts.dtypes import Endpoint
    from rollouts.training.backends.pytorch_factory import create_pytorch_backend
    from rollouts.training.losses import grpo_loss

    gpu_rank = config.trainer_cuda_device_ids[0]
    backend = create_pytorch_backend(
        model_name=config.model_name,
        checkpoint_dir=output_dir,
        device_type="cuda",
        dtype=config.dtype,
        gpu_rank=gpu_rank,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        loss_fn=lambda logits, batch: grpo_loss(logits, batch),
        num_minibatches=config.num_minibatches,
        max_grad_norm=config.max_grad_norm,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    endpoint = Endpoint(
        provider="openai",
        model=config.model_name,
        api_base=inference_engine.api_base,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    return backend, tokenizer, endpoint


def _create_generate_fn(
    config: GRPOConfig,
    endpoint: Any,
    tokenizer: Any,
    environment_cls: type[Environment],
    metadata_key: str | None,
    logger: logging.Logger,
) -> Callable:
    """Create the generate function for rollout generation."""
    if config.use_tito:
        return _create_tito_generate_fn(config, endpoint, tokenizer, metadata_key, logger)
    return _create_agent_generate_fn(
        config, endpoint, tokenizer, environment_cls, metadata_key, logger
    )


def _create_tito_generate_fn(
    config: GRPOConfig,
    endpoint: Any,
    tokenizer: Any,
    metadata_key: str | None,
    logger: logging.Logger,
) -> Callable:
    """Create TI/TO (token-level) generate function."""
    from rollouts.inference.backends import compute_suffix_ids
    from rollouts.providers import rollout_sglang_token_level, rollout_vllm_token_level

    suffix_ids = compute_suffix_ids(tokenizer)
    tito_provider = (
        rollout_sglang_token_level
        if config.inference_backend == "sglang"
        else rollout_vllm_token_level
    )

    async def generate_fn(batch_prompts: list[dict], **kwargs: Any) -> list:
        from rollouts.dtypes import Actor, Message, Trajectory

        results = []
        for prompt_data in batch_prompts:
            messages = prompt_data["messages"]
            if metadata_key:
                metadata = {metadata_key: prompt_data.get(metadata_key)}
            else:
                metadata = {k: v for k, v in prompt_data.items() if k != "messages"}

            try:
                initial_messages = [Message(role=m["role"], content=m["content"]) for m in messages]
                trajectory = Trajectory(messages=initial_messages)
                actor = Actor(trajectory=trajectory, endpoint=endpoint)

                async def noop_chunk(chunk: object) -> None:
                    pass

                updated_actor = await tito_provider(
                    actor, noop_chunk, tokenizer=tokenizer, suffix_ids=suffix_ids
                )

                samples = _trajectory_to_samples_tito(
                    trajectory=updated_actor.trajectory,
                    tokenizer=tokenizer,
                    strategy=config.trajectory_strategy,
                    metadata=metadata,
                )
                results.extend(samples)
            except Exception as e:
                logger.warning(f"TI/TO rollout failed: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        return results

    return generate_fn


def _create_agent_generate_fn(
    config: GRPOConfig,
    endpoint: Any,
    tokenizer: Any,
    environment_cls: type[Environment],
    metadata_key: str | None,
    logger: logging.Logger,
) -> Callable:
    """Create standard agent rollout generate function."""
    from rollouts.training.agent_integration import agent_rollout_to_sample

    async def generate_fn(batch_prompts: list[dict], **kwargs: Any) -> list:
        results = []
        for prompt_data in batch_prompts:
            messages = prompt_data["messages"]
            if metadata_key:
                metadata = {metadata_key: prompt_data.get(metadata_key)}
            else:
                metadata = {k: v for k, v in prompt_data.items() if k != "messages"}

            try:
                sample = await agent_rollout_to_sample(
                    prompt=messages,
                    environment_cls=environment_cls,
                    endpoint=endpoint,
                    tokenizer=tokenizer,
                    max_turns=config.max_turns,
                    metadata=metadata,
                )
                results.append(sample)
            except Exception as e:
                logger.warning(f"Rollout failed: {e}")

        return results

    return generate_fn


async def _process_training_step(
    step: int,
    batch: Any,
    config: GRPOConfig,
    backend: Any,
    tokenizer: Any,
    device: str,
    output_dir: Path,
    metrics_logger: Any,
    inference_engine: Any,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    """Process a single training step.

    Returns:
        Step metrics dict, or None if step was skipped
    """
    import json

    import torch

    from rollouts.training.losses import compute_group_advantages

    if not batch.tokens:
        logger.warning("No successful rollouts, skipping step")
        return None

    # Save rollouts to JSONL
    rollouts_file = output_dir / "rollouts.jsonl"
    with open(rollouts_file, "a") as f:  # noqa: ASYNC230
        for sample in batch.samples:
            record = {
                "step": step + 1,
                "prompt": sample.prompt,
                "response": sample.response,
                "reward": sample.reward,
                "status": sample.status.value,
                "group_index": sample.group_index,
                "turns": sample.metadata.get("turns"),
                "stop_reason": sample.metadata.get("stop_reason"),
                "messages": sample.metadata.get("messages"),
                "metadata": {
                    k: v
                    for k, v in sample.metadata.items()
                    if k not in ("turns", "stop_reason", "messages")
                },
            }
            f.write(json.dumps(record) + "\n")
            logger.info("rollout", extra=record)

    # Compute advantages
    rewards = batch.rewards
    group_indices = batch.group_indices
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    num_groups = len(set(group_indices)) if group_indices else len(rewards)
    logger.info(f"Reward: {mean_reward:.3f} ({len(rewards)} samples, {num_groups} groups)")

    if group_indices and len(set(group_indices)) > 1:
        advantages = compute_group_advantages(rewards, group_indices).to(device)
    else:
        advantages = torch.tensor([r - mean_reward for r in rewards], device=device)

    # Prepare batch tensors
    training_batch = _prepare_training_batch(batch, config, tokenizer, advantages, device)

    # Training step
    fb_future = backend.forward_backward(training_batch)
    fb_metrics = await fb_future.result()

    optim_future = backend.optim_step()
    optim_metrics = await optim_future.result()

    accumulated_metrics = {**fb_metrics, **optim_metrics}
    pg_loss = accumulated_metrics.get("pg_loss", 0.0)
    entropy = accumulated_metrics.get("entropy", 0.0)

    step_metrics = {
        "mean_reward": mean_reward,
        "num_samples": len(rewards),
        "num_groups": num_groups,
        **accumulated_metrics,
    }

    metrics_logger.log(step_metrics, step=step + 1)
    logger.info("metrics", extra={"step": step + 1, **step_metrics})

    if (step + 1) % config.log_every == 0:
        logger.info(
            f"Step {step + 1}: reward={mean_reward:.3f} | "
            f"pg_loss={pg_loss:.4f} | entropy={entropy:.2f}"
        )

    # Checkpoint (save to disk for recovery)
    should_checkpoint = (step + 1) % config.checkpoint_every == 0
    should_sync = (step + 1) % config.sync_weights_every == 0

    if should_checkpoint:
        ckpt_dir = await backend.save_checkpoint(step + 1, accumulated_metrics)
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    # Sync weights to inference engine (for on-policy training)
    # If we just checkpointed, use that. Otherwise save a temp checkpoint to RAM disk.
    if should_sync:
        if should_checkpoint:
            # Reuse the checkpoint we just saved
            sync_dir = ckpt_dir
        else:
            # Save temp checkpoint to RAM disk (/dev/shm) for fast I/O
            # This avoids slow disk writes when syncing every step
            from rollouts.training.weight_sync import get_fast_sync_dir

            fast_dir = get_fast_sync_dir()
            sync_dir = await backend.save_checkpoint_to_path(
                fast_dir / "sync_latest", accumulated_metrics
            )
        logger.info(f"Syncing weights to {inference_engine.name}...")
        await inference_engine.update_weights_from_checkpoint(str(sync_dir))
        logger.info("Weight sync complete")

    return step_metrics


def _prepare_training_batch(
    batch: Any,
    config: GRPOConfig,
    tokenizer: Any,
    advantages: Any,
    device: str,
) -> dict[str, Any]:
    """Prepare tensors for training step."""
    import torch

    max_len = min(max(len(t) for t in batch.tokens), config.max_seq_len)

    batch_tokens = []
    batch_loss_masks = []
    batch_rollout_logprobs = []
    has_rollout_logprobs = batch.rollout_log_probs is not None

    for i, (toks, mask) in enumerate(zip(batch.tokens, batch.loss_masks, strict=True)):
        toks_truncated = list(toks[:max_len])
        mask_truncated = list(mask[:max_len])
        pad_len = max_len - len(toks_truncated)
        toks_padded = toks_truncated + [tokenizer.pad_token_id or 0] * pad_len
        mask_padded = mask_truncated + [0.0] * pad_len
        batch_tokens.append(toks_padded)
        batch_loss_masks.append(mask_padded)

        if has_rollout_logprobs:
            rlp = list(batch.rollout_log_probs[i][:max_len])
            rlp_padded = rlp + [0.0] * (max_len - len(rlp))
            batch_rollout_logprobs.append(rlp_padded)

    input_ids = torch.tensor(batch_tokens, device=device)
    labels = input_ids.clone()
    loss_mask = torch.tensor(batch_loss_masks, device=device)

    training_batch = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "advantages": advantages,
    }

    if has_rollout_logprobs:
        rollout_logprobs_tensor = torch.tensor(batch_rollout_logprobs, device=device)
        seq_rollout_logprobs = (rollout_logprobs_tensor * loss_mask).sum(dim=1) / loss_mask.sum(
            dim=1
        ).clamp(min=1.0)
        training_batch["old_logprobs"] = seq_rollout_logprobs

    return training_batch


async def _grpo_train_async(
    config: GRPOConfig,
    prompts: list[dict[str, Any]],
    score_fn: Callable[[Sample], Score],
    environment_cls: type[Environment],
    metadata_key: str | None = None,
) -> dict[str, Any]:
    """Async GRPO training implementation."""
    import os

    from rollouts._logging import setup_logging
    from rollouts.training.datasets.data_buffer import DataBuffer
    from rollouts.training.metrics import JSONLLogger
    from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
    from rollouts.training.types import RolloutConfig

    # Setup logging
    use_json_logs = os.environ.get("ROLLOUTS_JSON_LOGS", "").lower() == "true"
    setup_logging(
        level="INFO",
        use_json=use_json_logs,
        use_color=not use_json_logs,
        logger_levels={"httpx": "WARNING", "httpcore": "WARNING"},
    )
    logger = logging.getLogger(__name__)

    # Setup output directory
    output_dir, run_name = _setup_output_dir(config)

    logger.info("=" * 60)
    logger.info(f"GRPO Training: {run_name}")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Backend: {config.inference_backend}")
    logger.info(f"Steps: {config.num_steps}")
    logger.info(f"Batch: {config.batch_size} prompts x {config.n_samples_per_prompt} samples")
    logger.info(f"Output: {output_dir}")

    config.save(output_dir / "config.json")
    metrics_logger = JSONLLogger(output_dir)

    # Launch inference engine
    inference_engine = _create_inference_engine(config, output_dir)
    gpu_str = ",".join(str(g) for g in config.inference_cuda_device_ids)
    logger.info(f"Launching {inference_engine.name} on GPU {gpu_str}...")

    inference_engine.launch()
    inference_engine.start_log_tailer()

    try:
        await inference_engine.wait_until_ready()
        logger.info(f"{inference_engine.name} ready")

        # Setup training backend
        backend, tokenizer, endpoint = _setup_training_backend(config, output_dir, inference_engine)
        device = f"cuda:{config.trainer_cuda_device_ids[0]}"

        # Load checkpoint if provided (for SFT → RL pipeline)
        if config.checkpoint_path:
            ckpt_path = Path(config.checkpoint_path)
            if (ckpt_path / "pytorch_model.bin").exists():
                # Our checkpoint format
                logger.info(f"Loading checkpoint from {ckpt_path}")
                await backend.load_checkpoint(ckpt_path)
                logger.info("Checkpoint loaded successfully")
            elif (ckpt_path / "config.json").exists():
                # HuggingFace format - already loaded via model_name
                logger.info(f"Using HuggingFace checkpoint: {ckpt_path}")
            else:
                raise ValueError(
                    f"Invalid checkpoint path: {ckpt_path} (no pytorch_model.bin or config.json)"
                )

        # Setup data and rollout generation
        logger.info(f"Dataset: {len(prompts)} prompts")
        data_buffer = DataBuffer(prompts=prompts)
        generate_fn = _create_generate_fn(
            config, endpoint, tokenizer, environment_cls, metadata_key, logger
        )

        rollout_config = RolloutConfig(
            batch_size=config.batch_size,
            n_samples_per_prompt=config.n_samples_per_prompt,
            over_sampling_factor=1.0,
            generate_fn=generate_fn,
            score_fn=score_fn,
        )

        # Training loop
        metrics_history = []

        async with AsyncRolloutManager(data_buffer, rollout_config) as rollout_manager:
            for step in range(config.num_steps):
                logger.info(f"\n--- Step {step + 1}/{config.num_steps} ---")

                batch = await rollout_manager.generate_batch(score_fn=score_fn)
                step_metrics = await _process_training_step(
                    step,
                    batch,
                    config,
                    backend,
                    tokenizer,
                    device,
                    output_dir,
                    metrics_logger,
                    inference_engine,
                    logger,
                )

                if step_metrics:
                    metrics_history.append({"step": step + 1, **step_metrics})

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)

        if metrics_history:
            first_reward = metrics_history[0]["mean_reward"]
            last_reward = metrics_history[-1]["mean_reward"]
            first_loss = metrics_history[0].get("pg_loss", 0.0)
            last_loss = metrics_history[-1].get("pg_loss", 0.0)
            logger.info(f"First: reward={first_reward:.3f}, pg_loss={first_loss:.4f}")
            logger.info(f"Last:  reward={last_reward:.3f}, pg_loss={last_loss:.4f}")

        metrics_logger.finish()
        return {"metrics_history": metrics_history}

    finally:
        logger.info(f"Shutting down {inference_engine.name}...")
        inference_engine.shutdown()
        logger.info(f"Logs: {inference_engine.log_path}")


# ──────────────────────── TI/TO Helpers ───────────────────────────────────────


def _trajectory_to_samples_tito(
    trajectory: Any,
    tokenizer: Any,
    strategy: str = "interleaved",
    metadata: dict[str, Any] | None = None,
) -> list[Sample]:
    """Convert TI/TO trajectory to training sample(s) based on strategy.

    This function is specialized for TI/TO mode where:
    - token_ids are stored directly in Choice (no retokenization needed)
    - logprobs are stored in Logprobs.content as per-token Logprob objects

    Args:
        trajectory: Trajectory with completions containing token_ids and logprobs
        tokenizer: HuggingFace tokenizer
        strategy: "interleaved" (one sample) or "branching" (one per assistant turn)
        metadata: Optional metadata

    Returns:
        List of Samples with tokens, loss_mask, and rollout_log_probs
    """
    assert strategy in ("interleaved", "branching"), f"Unknown strategy: {strategy}"

    if strategy == "interleaved":
        return [_trajectory_to_sample_tito_interleaved(trajectory, tokenizer, metadata)]
    else:
        return _trajectory_to_samples_tito_branching(trajectory, tokenizer, metadata)


def _trajectory_to_sample_tito_interleaved(
    trajectory: Any,
    tokenizer: Any,
    metadata: dict[str, Any] | None = None,
) -> Sample:
    """Convert TI/TO trajectory to single sample (interleaved strategy)."""
    from rollouts.training.types import Sample, Status

    assert trajectory is not None
    assert tokenizer is not None
    assert len(trajectory.messages) > 0

    # Extract prompt (messages before first assistant)
    prompt_messages = []
    for msg in trajectory.messages:
        if msg.role == "assistant":
            break
        prompt_messages.append(msg)

    prompt = tokenizer.apply_chat_template(
        [{"role": m.role, "content": _get_message_content(m)} for m in prompt_messages],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Extract tokens and logprobs from completions
    all_tokens: list[int] = []
    all_logprobs: list[float] = []
    loss_mask: list[float] = []

    # First, tokenize the prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    all_tokens.extend(prompt_ids)
    loss_mask.extend([0.0] * len(prompt_ids))  # Don't train on prompt
    all_logprobs.extend([0.0] * len(prompt_ids))  # Placeholder for prompt tokens

    # Extract tokens and logprobs from each completion
    for completion in trajectory.completions:
        if not completion.choices:
            continue

        choice = completion.choices[0]

        # Use stored token_ids
        if choice.token_ids:
            token_ids = list(choice.token_ids)
            all_tokens.extend(token_ids)
            loss_mask.extend([1.0] * len(token_ids))  # Train on completion tokens

            # Extract logprobs from Logprobs.content
            if choice.logprobs and choice.logprobs.content:
                for logprob_item in choice.logprobs.content:
                    all_logprobs.append(logprob_item.logprob)
            else:
                # No logprobs stored, use placeholder
                all_logprobs.extend([0.0] * len(token_ids))

    return Sample(
        prompt=prompt,
        tokens=all_tokens,
        loss_mask=loss_mask,
        rollout_log_probs=all_logprobs,
        reward=0.0,  # Will be computed by score_fn
        metadata=metadata or {},
        status=Status.COMPLETED,
    )


def _trajectory_to_samples_tito_branching(
    trajectory: Any,
    tokenizer: Any,
    metadata: dict[str, Any] | None = None,
) -> list[Sample]:
    """Convert TI/TO trajectory to samples using branching strategy.

    Each assistant turn becomes a separate sample:
    - Input: tokenized history up to (but not including) that assistant turn
    - Output: that assistant turn's token_ids (from TI/TO)
    - Loss mask: 0 for input, 1 for output

    This mirrors deployed usage exactly - each generation is independent.
    """
    from rollouts.training.types import Sample, Status

    assert trajectory is not None
    assert tokenizer is not None

    samples = []
    completion_idx = 0

    for msg_idx, msg in enumerate(trajectory.messages):
        if msg.role != "assistant":
            continue

        # Get completion for this assistant turn
        if completion_idx >= len(trajectory.completions):
            break
        completion = trajectory.completions[completion_idx]
        completion_idx += 1

        if not completion.choices:
            continue
        choice = completion.choices[0]
        if not choice.token_ids:
            continue

        # Input = all messages before this assistant turn
        input_messages = trajectory.messages[:msg_idx]
        if input_messages:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": m.role, "content": _get_message_content(m)} for m in input_messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        else:
            prompt_text = ""
            input_ids = []

        # Output tokens from stored token_ids (TI/TO)
        output_ids = list(choice.token_ids)

        # Extract logprobs if available
        if choice.logprobs and choice.logprobs.content:
            output_logprobs = [lp.logprob for lp in choice.logprobs.content]
        else:
            output_logprobs = [0.0] * len(output_ids)

        # Full sequence
        tokens = input_ids + output_ids
        loss_mask = [0.0] * len(input_ids) + [1.0] * len(output_ids)
        all_logprobs = [0.0] * len(input_ids) + output_logprobs

        # Build metadata for this turn
        turn_metadata = metadata.copy() if metadata else {}
        turn_metadata["turn_index"] = msg_idx

        sample = Sample(
            prompt=prompt_text,
            tokens=tokens,
            loss_mask=loss_mask,
            rollout_log_probs=all_logprobs,
            reward=0.0,  # Will be computed by score_fn
            metadata=turn_metadata,
            status=Status.COMPLETED,
        )

        samples.append(sample)

    return samples


def _get_message_content(msg: Any) -> str:
    """Extract text content from a Message."""
    from rollouts.dtypes import TextContent, ThinkingContent

    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingContent):
                text_parts.append(block.thinking)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    text_parts.append(block.get("thinking", ""))
        return "".join(text_parts)
    return str(content) if content else ""
