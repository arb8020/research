"""GRPO Training Loop.

Shared training infrastructure for GRPO (Group Relative Policy Optimization).
Each task provides prompts, score_fn, and environment_cls - this module handles
the rest: SGLang server, training backend, rollout generation, gradient updates.

Usage:
    from ..training.grpo import GRPOConfig, grpo_train

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from ..dtypes import Environment, Score
    from ..training.types import Sample

# ──────────────────────── Sub-Configs (re-exported from shared) ───────────────

from ..training.configs import (  # noqa: E402
    CheckpointConfig,
    InferenceConfig,
    ModelConfig,
    OutputConfig,
    RolloutConfig,
    TrainerConfig,
)


def GRPOOutputConfig(  # noqa: N802 — factory, not a class
    experiment_name: str = "grpo",
    output_dir: str = "results/rl",
) -> OutputConfig:
    """Create OutputConfig with GRPO defaults (output_dir="results/rl")."""
    return OutputConfig(output_dir=output_dir, experiment_name=experiment_name)


# ──────────────────────── Composed Config ────────────────────────────────────


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO training.

    Composed from sub-configs that separate concerns cleanly.
    Use replace() for inheritance (see docs/code_style/experiment_config.md).

    Example:
        config = GRPOConfig(
            model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
            trainer=TrainerConfig(lr=1e-5),
            rollout=RolloutConfig(batch_size=4, n_samples_per_prompt=4),
        )

        # Derive a variant:
        fast = replace(config, trainer=replace(config.trainer, lr=1e-4))
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(output_dir="results/rl", experiment_name="grpo")
    )

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
        >>> from ..training.grpo import GRPOConfig, grpo_train
        >>> from ..environments.no_tools import BasicEnvironment
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

    Priority:
    1. ROLLOUTS_OUTPUT_DIR env var (set by runner — single source of truth)
    2. ROLLOUTS_RUN_NAME env var + config.output.output_dir (legacy)
    3. Generate from config (local execution)

    Returns:
        Tuple of (output_dir, run_name)
    """
    import os
    from datetime import datetime, timezone

    explicit_dir = os.environ.get("ROLLOUTS_OUTPUT_DIR")
    if explicit_dir:
        output_dir = Path(explicit_dir)
        run_name = output_dir.name
    elif run_name_env := os.environ.get("ROLLOUTS_RUN_NAME"):
        run_name = run_name_env
        output_dir = Path(config.output.output_dir) / run_name
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_name = f"{config.output.experiment_name}_{timestamp}"
        output_dir = Path(config.output.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, run_name


def _create_inference_engine(
    config: GRPOConfig, output_dir: Path
) -> Any:  # SGLangEngine | VLLMEngine
    """Create and configure inference engine."""
    from ..training.weight_sync import SGLangEngine, VLLMEngine

    if config.inference.backend == "sglang":
        return SGLangEngine(
            model_name=config.model.name,
            port=config.inference.port,
            cuda_device_ids=config.inference.cuda_device_ids,
            output_dir=output_dir,
            dtype=config.model.dtype,
            mem_fraction=config.inference.mem_fraction,
        )
    elif config.inference.backend == "vllm":
        return VLLMEngine(
            model_name=config.model.name,
            port=config.inference.port,
            cuda_device_ids=config.inference.cuda_device_ids,
            output_dir=output_dir,
            dtype=config.model.dtype,
            gpu_memory_utilization=config.inference.mem_fraction,
        )
    else:
        msg = f"Unknown inference backend: {config.inference.backend}"
        raise ValueError(msg)


def _make_loss_fn(
    trainer: TrainerConfig,
    vanilla_fn: Callable,
    clipped_fn: Callable,
    masked_fn: Callable,
) -> Callable:
    """Build the loss function from trainer config."""
    if trainer.loss_type == "vanilla":
        return vanilla_fn
    if trainer.loss_type == "clipped":
        return clipped_fn
    if trainer.loss_type == "masked":
        lo, hi = trainer.mask_ratio_low, trainer.mask_ratio_high

        def _masked(logits: Any, batch: Any) -> Any:
            return masked_fn(logits, batch, ratio_low=lo, ratio_high=hi)

        return _masked
    raise ValueError(
        f"Unknown loss_type: {trainer.loss_type!r}. Use 'vanilla', 'clipped', or 'masked'."
    )


def _setup_training_backend(
    config: GRPOConfig, output_dir: Path, inference_engine: Any
) -> tuple[Any, Any, Any]:  # (backend, tokenizer, endpoint)
    """Setup training backend, tokenizer, and endpoint.

    Returns:
        Tuple of (backend, tokenizer, endpoint)
    """
    # TODO: Remove HF transformers dependency. Use tokenizers library directly
    # or load tokenizer.json with custom wrapper.
    from transformers import AutoTokenizer

    from ..dtypes import Endpoint
    from ..training.backends.pytorch_factory import create_pytorch_backend
    from ..training.losses import grpo_loss, grpo_loss_clipped, grpo_loss_masked

    # Select loss function based on config
    loss_fn = _make_loss_fn(config.trainer, grpo_loss, grpo_loss_clipped, grpo_loss_masked)

    gpu_rank = config.trainer.cuda_device_ids[0]
    backend = create_pytorch_backend(
        model_name=config.model.name,
        checkpoint_dir=output_dir,
        device_type="cuda",
        dtype=config.model.dtype,
        gpu_rank=gpu_rank,
        learning_rate=config.trainer.lr,
        weight_decay=config.trainer.weight_decay,
        loss_fn=loss_fn,
        num_minibatches=config.trainer.num_minibatches,
        max_grad_norm=config.trainer.max_grad_norm,
        use_lora=config.model.use_lora,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    endpoint = Endpoint(
        model=f"openai/{config.model.name}",
        base_url=inference_engine.base_url,
        api_format="openai-completions",
        temperature=config.rollout.temperature,
        max_tokens=config.rollout.max_tokens,
        extra_params=config.rollout.extra_params or None,
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
    """Create the generate function for rollout generation.

    Uses the agent loop with standard providers. Providers now always request
    logprobs and populate Choice.token_ids, so token data flows through
    the normal agent path.

    For multi-turn agent rollouts (max_turns > 1), the agent loop handles
    tool execution. For single-turn (max_turns = 1), it's just one LLM call.
    """
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
    from ..inference.backends import compute_suffix_ids
    from ..providers import rollout_sglang_token_level, rollout_vllm_token_level

    suffix_ids = compute_suffix_ids(tokenizer)
    tito_provider = (
        rollout_sglang_token_level
        if config.inference.backend == "sglang"
        else rollout_vllm_token_level
    )

    async def generate_fn(batch_prompts: list[dict], **kwargs: Any) -> list:
        from ..dtypes import Actor, Message, Trajectory

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
                    strategy=config.rollout.trajectory_strategy,
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
    from ..training.agent_integration import agent_rollout_to_sample

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
                    max_turns=config.rollout.max_turns,
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

    from ..training.losses import compute_group_advantages

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

    if (step + 1) % config.checkpoint.log_every == 0:
        logger.info(
            f"Step {step + 1}: reward={mean_reward:.3f} | "
            f"pg_loss={pg_loss:.4f} | entropy={entropy:.2f}"
        )

    # Checkpoint (save to disk for recovery)
    should_checkpoint = (step + 1) % config.checkpoint.checkpoint_every == 0
    should_sync = (step + 1) % config.checkpoint.sync_weights_every == 0

    if should_checkpoint:
        ckpt_dir = await backend.save_checkpoint(step + 1, accumulated_metrics)
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    # Sync weights to inference engine (for on-policy training)
    if should_sync:
        if config.checkpoint.weight_sync_mode == "nccl":
            # NCCL in-flight sync: GPU-to-GPU broadcast (PipelineRL-style)
            logger.info(f"Syncing weights via NCCL to {inference_engine.name}...")
            await backend.sync_weights_nccl()
            logger.info("NCCL weight sync complete")
        else:
            # Disk-based sync: save to /dev/shm, reload (default)
            from ..training.weight_sync import get_fast_sync_dir

            fast_dir = get_fast_sync_dir()
            sync_dir = await backend.save_weights_for_sampler(fast_dir / "sync_latest")
            logger.info(f"Syncing weights to {inference_engine.name}...")
            await inference_engine.update_weights_from_checkpoint(str(sync_dir))
            logger.info("Weight sync complete")

    return step_metrics


async def _process_training_step_no_sync(
    step: int,
    batch: Any,
    config: GRPOConfig,
    backend: Any,
    tokenizer: Any,
    device: str,
    output_dir: Path,
    metrics_logger: Any,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    """Process a training step without weight sync (for true_pipeline mode).

    Same as _process_training_step but without the weight sync logic.
    Weight sync is handled separately by PipelineWeightSyncManager.

    Returns:
        Step metrics dict, or None if step was skipped
    """
    import json

    import torch

    from ..training.losses import compute_group_advantages

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
                "weight_version": sample.weight_version,  # Track staleness
                "turns": sample.metadata.get("turns"),
                "stop_reason": sample.metadata.get("stop_reason"),
                "messages": sample.metadata.get("messages"),
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

    if (step + 1) % config.checkpoint.log_every == 0:
        logger.info(
            f"Step {step + 1}: reward={mean_reward:.3f} | "
            f"pg_loss={pg_loss:.4f} | entropy={entropy:.2f}"
        )

    # Checkpoint (save to disk for recovery)
    should_checkpoint = (step + 1) % config.checkpoint.checkpoint_every == 0
    if should_checkpoint:
        ckpt_dir = await backend.save_checkpoint(step + 1, accumulated_metrics)
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    # Note: NO weight sync here - handled by PipelineWeightSyncManager

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

    max_len = min(max(len(t) for t in batch.tokens), config.rollout.max_seq_len)

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

    from .._logging import setup_logging
    from ..training.datasets.data_buffer import DataBuffer
    from ..training.metrics import JSONLLogger
    from ..training.rollout_gen.async_rollout_manager import AsyncRolloutManager
    from ..training.types import RolloutConfig

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
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Backend: {config.inference.backend}")
    logger.info(f"Steps: {config.checkpoint.num_steps}")
    logger.info(
        f"Batch: {config.rollout.batch_size} prompts x {config.rollout.n_samples_per_prompt} samples"
    )
    logger.info(f"Output: {output_dir}")

    config.save(output_dir / "config.json")
    metrics_logger = JSONLLogger(output_dir)

    # Launch inference engine
    inference_engine = _create_inference_engine(config, output_dir)
    gpu_str = ",".join(str(g) for g in config.inference.cuda_device_ids)
    logger.info(f"Launching {inference_engine.name} on GPU {gpu_str}...")

    inference_engine.launch()
    inference_engine.start_log_tailer()

    try:
        await inference_engine.wait_until_ready()
        logger.info(f"{inference_engine.name} ready")

        # Setup training backend
        backend, tokenizer, endpoint = _setup_training_backend(config, output_dir, inference_engine)
        device = f"cuda:{config.trainer.cuda_device_ids[0]}"

        # Load checkpoint if provided (for SFT → RL pipeline)
        if config.model.checkpoint_path:
            ckpt_path = Path(config.model.checkpoint_path)
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

        # VRAM preflight: dry-run one forward+backward at worst-case seq_len
        if not config.trainer.skip_vram_check:
            from ..training.vram import preflight_vram_check

            preflight_vram_check(backend, config, device)
        else:
            logger.info("VRAM preflight check skipped (skip_vram_check=True)")

        # Initialize NCCL weight sync if enabled (PipelineRL-style in-flight updates)
        if config.checkpoint.weight_sync_mode == "nccl":
            logger.info("Initializing NCCL weight sync...")
            await backend.init_nccl_weight_sync(
                inference_endpoints=[inference_engine.base_url],
                master_port=config.checkpoint.nccl_master_port,
            )
            logger.info("NCCL weight sync initialized")

        # Setup data and rollout generation
        logger.info(f"Dataset: {len(prompts)} prompts")
        data_buffer = DataBuffer(prompts=prompts)
        generate_fn = _create_generate_fn(
            config, endpoint, tokenizer, environment_cls, metadata_key, logger
        )

        rollout_config = RolloutConfig(
            batch_size=config.rollout.batch_size,
            n_samples_per_prompt=config.rollout.n_samples_per_prompt,
            over_sampling_factor=1.0,
            generate_fn=generate_fn,
            score_fn=score_fn,
        )

        # Training loop
        metrics_history = []

        if config.checkpoint.pipeline_mode == "true_pipeline":
            # True PipelineRL: both sampling AND weight sync are non-blocking
            # Inference never stops, training never waits
            from ..training.rollout_gen.pipelined_rollout_manager import PipelinedRolloutManager
            from ..training.weight_sync import PipelineWeightSyncManager

            pipelined_manager = PipelinedRolloutManager(
                data_buffer=data_buffer,
                config=rollout_config,
                max_lag=config.checkpoint.max_lag,
                queue_size=config.checkpoint.pipeline_queue_size,
            )

            weight_sync_manager = PipelineWeightSyncManager(
                inference_endpoints=[inference_engine.base_url],
                max_lag=config.checkpoint.max_lag,
                nccl_master_port=config.checkpoint.nccl_master_port,
            )

            logger.info(
                f"Using TRUE PipelineRL mode (max_lag={config.checkpoint.max_lag}, "
                f"queue_size={config.checkpoint.pipeline_queue_size})"
            )
            logger.info("  - Background sampling: ON (inference never stops)")
            logger.info("  - Non-blocking weight sync: ON (training never waits)")

            try:
                # Initialize NCCL for non-blocking weight sync
                await weight_sync_manager.init_nccl_group()

                async with pipelined_manager:
                    async with trio.open_nursery() as nursery:
                        # Start background sampling
                        await pipelined_manager.start_sampling(
                            nursery=nursery,
                            initial_weight_version=weight_sync_manager.current_version,
                        )

                        for step in range(config.checkpoint.num_steps):
                            logger.info(f"\n--- Step {step + 1}/{config.checkpoint.num_steps} ---")

                            # Get batch (filters stale samples based on weight version)
                            batch = await pipelined_manager.get_batch(
                                current_weight_version=weight_sync_manager.current_version,
                                score_fn=score_fn,
                            )

                            # Training step (uses compute_grpo_loss, backward, optimizer)
                            step_metrics = await _process_training_step_no_sync(
                                step,
                                batch,
                                config,
                                backend,
                                tokenizer,
                                device,
                                output_dir,
                                metrics_logger,
                                logger,
                            )

                            if step_metrics:
                                metrics_history.append({"step": step + 1, **step_metrics})

                            # Non-blocking weight sync - spawns background task
                            # Training continues immediately, doesn't wait!
                            should_sync = (step + 1) % config.checkpoint.sync_weights_every == 0
                            if should_sync:
                                logger.debug(f"Spawning async weight sync (v={weight_sync_manager.current_version + 1})")
                                await weight_sync_manager.broadcast_weights_async(backend.model, nursery)

                            # Update version in rollout manager
                            pipelined_manager.update_weight_version(weight_sync_manager.current_version)

                        # Log pipeline stats
                        stats = pipelined_manager.stats()
                        logger.info(
                            f"Pipeline stats: generated={stats['samples_generated']}, "
                            f"discarded_stale={stats['samples_discarded_stale']} "
                            f"({stats['discard_rate']:.1f}%)"
                        )
            finally:
                await weight_sync_manager.cleanup()

        elif config.checkpoint.pipeline_mode == "async":
            # Async sampling but blocking weight sync
            # Sampling runs in background, but training waits for weight sync
            from ..training.rollout_gen.pipelined_rollout_manager import PipelinedRolloutManager

            pipelined_manager = PipelinedRolloutManager(
                data_buffer=data_buffer,
                config=rollout_config,
                max_lag=config.checkpoint.max_lag,
                queue_size=config.checkpoint.pipeline_queue_size,
            )

            logger.info(
                f"Using async pipeline (max_lag={config.checkpoint.max_lag}, "
                f"queue_size={config.checkpoint.pipeline_queue_size})"
            )

            async with pipelined_manager:
                async with trio.open_nursery() as nursery:
                    # Start background sampling
                    await pipelined_manager.start_sampling(
                        nursery=nursery,
                        initial_weight_version=backend.weight_version,
                    )

                    for step in range(config.checkpoint.num_steps):
                        logger.info(f"\n--- Step {step + 1}/{config.checkpoint.num_steps} ---")

                        # Get batch (filters stale samples based on weight version)
                        batch = await pipelined_manager.get_batch(
                            current_weight_version=backend.weight_version,
                            score_fn=score_fn,
                        )

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

                        # Update weight version in manager (tells sampler weights changed)
                        pipelined_manager.update_weight_version(backend.weight_version)

                    # Log pipeline stats
                    stats = pipelined_manager.stats()
                    logger.info(
                        f"Pipeline stats: generated={stats['samples_generated']}, "
                        f"discarded_stale={stats['samples_discarded_stale']} "
                        f"({stats['discard_rate']:.1f}%)"
                    )

        else:
            # Synchronous training (default)
            # Generate batch, train, sync weights, repeat
            async with AsyncRolloutManager(data_buffer, rollout_config) as rollout_manager:
                for step in range(config.checkpoint.num_steps):
                    logger.info(f"\n--- Step {step + 1}/{config.checkpoint.num_steps} ---")

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
        # Cleanup NCCL weight sync if it was initialized
        if config.checkpoint.weight_sync_mode == "nccl":
            try:
                await backend.cleanup_nccl_weight_sync()
            except NameError:
                pass  # backend not yet created
            except Exception as e:
                logger.warning(f"NCCL cleanup failed: {e}")

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
    from ..training.types import Sample, Status

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
    from ..training.types import Sample, Status

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
    from ..dtypes import TextContent, ThinkingContent

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
