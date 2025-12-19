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

    # Training loop
    num_steps: int = 100
    log_every: int = 1
    checkpoint_every: int = 20

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


async def _grpo_train_async(
    config: GRPOConfig,
    prompts: list[dict[str, Any]],
    score_fn: Callable[[Sample], Score],
    environment_cls: type[Environment],
    metadata_key: str | None = None,
) -> dict[str, Any]:
    """Async GRPO training implementation."""
    import os
    from datetime import datetime, timezone

    import torch
    from transformers import AutoTokenizer

    from rollouts._logging import setup_logging
    from rollouts.dtypes import Endpoint
    from rollouts.training.agent_integration import agent_rollout_to_sample
    from rollouts.training.backends.pytorch_factory import create_pytorch_backend
    from rollouts.training.datasets.data_buffer import DataBuffer
    from rollouts.training.losses import compute_group_advantages, grpo_loss
    from rollouts.training.metrics import JSONLLogger
    from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
    from rollouts.training.types import RolloutConfig
    from rollouts.training.weight_sync import SGLangEngine, VLLMEngine

    # Setup logging (JSON format when TUI is active)
    use_json_logs = os.environ.get("ROLLOUTS_JSON_LOGS", "").lower() == "true"
    setup_logging(
        level="INFO",
        use_json=use_json_logs,
        use_color=not use_json_logs,
        logger_levels={"httpx": "WARNING", "httpcore": "WARNING"},
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    # Use ROLLOUTS_RUN_NAME if provided (from run_remote), otherwise generate timestamp
    run_name = os.environ.get("ROLLOUTS_RUN_NAME")
    if run_name:
        # Remote run - run_remote already created the directory
        output_dir = Path(config.output_dir) / run_name
    else:
        # Local run - generate timestamped name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_name = f"{config.experiment_name}_{timestamp}"
        output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"GRPO Training: {run_name}")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Backend: {config.inference_backend}")
    logger.info(f"Steps: {config.num_steps}")
    logger.info(f"Batch: {config.batch_size} prompts x {config.n_samples_per_prompt} samples")
    logger.info(f"Output: {output_dir}")

    config.save(output_dir / "config.json")

    # Initialize metrics logger (structured JSONL for TUI/analysis)
    metrics_logger = JSONLLogger(output_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Launch inference server (SGLang or vLLM)
    # ─────────────────────────────────────────────────────────────────────────
    if config.inference_backend == "sglang":
        inference_engine = SGLangEngine(
            model_name=config.model_name,
            port=config.inference_port,
            cuda_device_ids=config.inference_cuda_device_ids,
            output_dir=output_dir,
            dtype=config.dtype,
            mem_fraction=config.mem_fraction,
        )
    elif config.inference_backend == "vllm":
        inference_engine = VLLMEngine(
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

    gpu_str = ",".join(str(g) for g in config.inference_cuda_device_ids)
    logger.info(f"Launching {inference_engine.name} on GPU {gpu_str}...")

    inference_engine.launch()
    inference_engine.start_log_tailer()  # Route logs via Python logging

    try:
        await inference_engine.wait_until_ready()
        logger.info(f"{inference_engine.name} ready")

        # ─────────────────────────────────────────────────────────────────────
        # 2. Setup training backend
        # ─────────────────────────────────────────────────────────────────────
        gpu_rank = config.trainer_cuda_device_ids[0]
        device = f"cuda:{gpu_rank}"
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

        # Create endpoint for agent rollouts
        endpoint = Endpoint(
            provider="openai",
            model=config.model_name,
            api_base=inference_engine.api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # ─────────────────────────────────────────────────────────────────────
        # 3. Setup data and rollout generation
        # ─────────────────────────────────────────────────────────────────────
        logger.info(f"Dataset: {len(prompts)} prompts")
        data_buffer = DataBuffer(prompts=prompts)

        # Generation function - TI/TO or standard agent rollouts
        if config.use_tito:
            # TI/TO mode: Use token-level providers directly
            # This avoids retokenization issues that cause RL training collapse
            from rollouts.inference.backends import compute_suffix_ids
            from rollouts.providers import rollout_sglang_token_level, rollout_vllm_token_level

            suffix_ids = compute_suffix_ids(tokenizer)
            tito_provider = (
                rollout_sglang_token_level
                if config.inference_backend == "sglang"
                else rollout_vllm_token_level
            )

            async def generate_fn(batch_prompts: list[dict], **kwargs: Any) -> list:
                """Generate samples using TI/TO (token-level) providers."""
                from rollouts.dtypes import Actor, Message, Trajectory
                from rollouts.training.types import Sample, Status

                results = []
                for prompt_data in batch_prompts:
                    messages = prompt_data["messages"]
                    if metadata_key:
                        metadata = {metadata_key: prompt_data.get(metadata_key)}
                    else:
                        metadata = {k: v for k, v in prompt_data.items() if k != "messages"}

                    try:
                        # Build initial trajectory from messages
                        initial_messages = [
                            Message(role=m["role"], content=m["content"]) for m in messages
                        ]
                        trajectory = Trajectory(messages=initial_messages)
                        actor = Actor(trajectory=trajectory, endpoint=endpoint)

                        # Single-turn TI/TO rollout (multi-turn support via agent loop later)
                        async def noop_chunk(chunk):
                            pass

                        updated_actor = await tito_provider(
                            actor,
                            noop_chunk,
                            tokenizer=tokenizer,
                            suffix_ids=suffix_ids,
                        )

                        # Extract sample(s) from trajectory based on strategy
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
        else:
            # Standard mode: Use unified agent infrastructure
            async def generate_fn(batch_prompts: list[dict], **kwargs: Any) -> list:
                """Generate samples using agent_rollout_to_sample."""
                results = []
                for prompt_data in batch_prompts:
                    messages = prompt_data["messages"]

                    # Extract metadata (everything except "messages")
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

        rollout_config = RolloutConfig(
            batch_size=config.batch_size,
            n_samples_per_prompt=config.n_samples_per_prompt,
            over_sampling_factor=1.0,
            generate_fn=generate_fn,
            score_fn=score_fn,
        )

        # ─────────────────────────────────────────────────────────────────────
        # 4. Training loop
        # ─────────────────────────────────────────────────────────────────────
        import json

        metrics_history = []
        rollouts_file = output_dir / "rollouts.jsonl"

        async with AsyncRolloutManager(data_buffer, rollout_config) as rollout_manager:
            for step in range(config.num_steps):
                logger.info(f"\n--- Step {step + 1}/{config.num_steps} ---")

                # Generate rollouts
                batch = await rollout_manager.generate_batch(score_fn=score_fn)

                if not batch.tokens:
                    logger.warning("No successful rollouts, skipping step")
                    continue

                # Save rollouts to JSONL (for debugging/analysis)
                # Rich format matching run_eval.py output
                with open(rollouts_file, "a") as f:  # noqa: ASYNC230
                    for sample in batch.samples:
                        record = {
                            "step": step + 1,
                            "prompt": sample.prompt,
                            "response": sample.response,
                            "reward": sample.reward,
                            "status": sample.status.value,
                            "group_index": sample.group_index,
                            # Extract agent execution info from metadata
                            "turns": sample.metadata.get("turns"),
                            "stop_reason": sample.metadata.get("stop_reason"),
                            "messages": sample.metadata.get("messages"),
                            # Keep remaining metadata (ground_truth, etc.)
                            "metadata": {
                                k: v
                                for k, v in sample.metadata.items()
                                if k not in ("turns", "stop_reason", "messages")
                            },
                        }
                        f.write(json.dumps(record) + "\n")
                        # Also emit to log stream for TUI (remote runs can't access file)
                        logger.info("rollout", extra=record)

                # Compute group-wise advantages (the "G" in GRPO)
                rewards = batch.rewards
                group_indices = batch.group_indices

                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                num_groups = len(set(group_indices)) if group_indices else len(rewards)
                logger.info(
                    f"Reward: {mean_reward:.3f} ({len(rewards)} samples, {num_groups} groups)"
                )

                if group_indices and len(set(group_indices)) > 1:
                    advantages = compute_group_advantages(rewards, group_indices).to(device)
                else:
                    advantages = torch.tensor([r - mean_reward for r in rewards], device=device)

                # Prepare batch tensors
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

                    # Handle rollout logprobs for TI/TO off-policy correction
                    if has_rollout_logprobs:
                        rlp = list(batch.rollout_log_probs[i][:max_len])
                        rlp_padded = rlp + [0.0] * (max_len - len(rlp))
                        batch_rollout_logprobs.append(rlp_padded)

                input_ids = torch.tensor(batch_tokens, device=device)
                labels = input_ids.clone()
                loss_mask = torch.tensor(batch_loss_masks, device=device)

                # Training step (Tinker pattern: backend handles minibatching internally)
                training_batch = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "loss_mask": loss_mask,
                    "advantages": advantages,
                }

                # Add old_logprobs for TI/TO off-policy correction (sequence-level)
                if has_rollout_logprobs:
                    # Compute sequence-level logprobs: sum of per-token logprobs where mask > 0
                    rollout_logprobs_tensor = torch.tensor(batch_rollout_logprobs, device=device)
                    # Sequence-level = sum of token logprobs (mean would also work)
                    seq_rollout_logprobs = (rollout_logprobs_tensor * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)
                    training_batch["old_logprobs"] = seq_rollout_logprobs

                # forward_backward handles gradient accumulation via trainer_config
                fb_future = backend.forward_backward(training_batch)
                fb_metrics = await fb_future.result()

                # optim_step clips gradients and updates weights
                optim_future = backend.optim_step()
                optim_metrics = await optim_future.result()

                # Merge metrics from both calls
                accumulated_metrics = {**fb_metrics, **optim_metrics}

                pg_loss = accumulated_metrics.get("pg_loss", 0.0)
                entropy = accumulated_metrics.get("entropy", 0.0)

                # Log metrics
                step_metrics = {
                    "mean_reward": mean_reward,
                    "num_samples": len(rewards),
                    "num_groups": num_groups,
                    **accumulated_metrics,
                }
                metrics_history.append({"step": step + 1, **step_metrics})

                # Write to structured metrics.jsonl (for TUI/analysis)
                metrics_logger.log(step_metrics, step=step + 1)

                # Emit structured metrics for TUI (extra fields become top-level in JSONL)
                # This allows the TUI to detect and plot metrics
                logger.info(
                    "metrics",
                    extra={"step": step + 1, **step_metrics},
                )

                if (step + 1) % config.log_every == 0:
                    logger.info(
                        f"Step {step + 1}: reward={mean_reward:.3f} | "
                        f"pg_loss={pg_loss:.4f} | entropy={entropy:.2f}"
                    )

                # Checkpoint and sync weights
                if (step + 1) % config.checkpoint_every == 0:
                    ckpt_dir = await backend.save_checkpoint(step + 1, accumulated_metrics)
                    logger.info(f"Saved checkpoint: {ckpt_dir}")

                    logger.info(f"Syncing weights to {inference_engine.name}...")
                    await inference_engine.update_weights_from_checkpoint(str(ckpt_dir))
                    logger.info("Weight sync complete")

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

        # Finalize metrics logger
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
) -> list["Sample"]:
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
) -> "Sample":
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

    # Extract response text
    response_messages = trajectory.messages[len(prompt_messages) :]
    response = (
        tokenizer.apply_chat_template(
            [{"role": m.role, "content": _get_message_content(m)} for m in response_messages],
            tokenize=False,
            add_generation_prompt=False,
        )
        if response_messages
        else ""
    )

    return Sample(
        prompt=prompt,
        response=response,
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
) -> list["Sample"]:
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
            response=_get_message_content(msg),
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
