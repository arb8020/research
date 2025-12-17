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
    inference_gpu_ids: tuple[int, ...] = (0,)
    mem_fraction: float = 0.7

    # Trainer
    trainer_gpu_ids: tuple[int, ...] = (0,)
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
    from datetime import datetime, timezone

    import torch

    from rollouts._logging import setup_logging
    from rollouts.dtypes import Endpoint
    from rollouts.training.agent_integration import agent_rollout_to_sample
    from rollouts.training.backends.pytorch import PyTorchTrainingBackend
    from rollouts.training.datasets.data_buffer import DataBuffer
    from rollouts.training.losses import compute_group_advantages, grpo_loss
    from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
    from rollouts.training.types import RolloutConfig
    from rollouts.training.weight_sync import SGLangEngine, VLLMEngine

    # Setup logging
    setup_logging(
        level="INFO",
        use_color=True,
        logger_levels={"httpx": "WARNING", "httpcore": "WARNING"},
    )
    logger = logging.getLogger(__name__)

    # Create timestamped output directory
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

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Launch inference server (SGLang or vLLM)
    # ─────────────────────────────────────────────────────────────────────────
    if config.inference_backend == "sglang":
        inference_engine = SGLangEngine(
            model_name=config.model_name,
            port=config.inference_port,
            gpu_ids=config.inference_gpu_ids,
            output_dir=output_dir,
            dtype=config.dtype,
            mem_fraction=config.mem_fraction,
        )
    elif config.inference_backend == "vllm":
        inference_engine = VLLMEngine(
            model_name=config.model_name,
            port=config.inference_port,
            gpu_ids=config.inference_gpu_ids,
            output_dir=output_dir,
            dtype=config.dtype,
            gpu_memory_utilization=config.mem_fraction,
        )
    else:
        msg = f"Unknown inference backend: {config.inference_backend}"
        raise ValueError(msg)

    gpu_str = ",".join(str(g) for g in config.inference_gpu_ids)
    logger.info(f"Launching {inference_engine.name} on GPU {gpu_str}...")

    inference_proc = inference_engine.launch()
    inference_engine.start_log_tailer()  # Emit JSONL to stdout for TUI

    try:
        await inference_engine.wait_until_ready()
        logger.info(f"{inference_engine.name} ready")

        # ─────────────────────────────────────────────────────────────────────
        # 2. Setup training backend
        # ─────────────────────────────────────────────────────────────────────
        device = f"cuda:{config.trainer_gpu_ids[0]}"
        backend = PyTorchTrainingBackend.from_pretrained(
            model_name=config.model_name,
            device=device,
            dtype=config.dtype,
            lr=config.lr,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            loss_fn=lambda logits, batch: grpo_loss(logits, batch),
        )
        tokenizer = backend.tokenizer

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

        # Generation function using unified agent infrastructure
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
                with open(rollouts_file, "a") as f:
                    for sample in batch.samples:
                        record = {
                            "step": step + 1,
                            "prompt": sample.prompt,
                            "response": sample.response,
                            "reward": sample.reward,
                            "metadata": sample.metadata,
                        }
                        f.write(json.dumps(record) + "\n")

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
                for toks, mask in zip(batch.tokens, batch.loss_masks, strict=True):
                    toks_truncated = list(toks[:max_len])
                    mask_truncated = list(mask[:max_len])
                    pad_len = max_len - len(toks_truncated)
                    toks_padded = toks_truncated + [tokenizer.pad_token_id or 0] * pad_len
                    mask_padded = mask_truncated + [0.0] * pad_len
                    batch_tokens.append(toks_padded)
                    batch_loss_masks.append(mask_padded)

                input_ids = torch.tensor(batch_tokens, device=device)
                labels = input_ids.clone()
                loss_mask = torch.tensor(batch_loss_masks, device=device)

                # Training step with gradient accumulation
                batch_size = input_ids.size(0)
                num_minibatches = min(config.num_minibatches, batch_size)
                minibatch_size = batch_size // num_minibatches

                accumulated_metrics: dict[str, float] = {}
                for mb_idx in range(num_minibatches):
                    start = mb_idx * minibatch_size
                    end = start + minibatch_size

                    mb_batch = {
                        "input_ids": input_ids[start:end],
                        "labels": labels[start:end],
                        "loss_mask": loss_mask[start:end],
                        "advantages": advantages[start:end],
                    }

                    is_last = mb_idx == num_minibatches - 1
                    mb_metrics = backend.forward_backward(
                        mb_batch,
                        accumulation_steps=num_minibatches,
                        step_optimizer=is_last,
                    )

                    for k, v in mb_metrics.items():
                        accumulated_metrics[k] = (
                            accumulated_metrics.get(k, 0.0) + v / num_minibatches
                        )

                pg_loss = accumulated_metrics.get("pg_loss", 0.0)
                entropy = accumulated_metrics.get("entropy", 0.0)

                # Log
                step_metrics = {
                    "step": step + 1,
                    "mean_reward": mean_reward,
                    "num_samples": len(rewards),
                    "num_groups": num_groups,
                    **accumulated_metrics,
                }
                metrics_history.append(step_metrics)

                if (step + 1) % config.log_every == 0:
                    logger.info(
                        f"Step {step + 1}: reward={mean_reward:.3f} | "
                        f"pg_loss={pg_loss:.4f} | entropy={entropy:.2f}"
                    )

                # Checkpoint and sync weights
                if (step + 1) % config.checkpoint_every == 0:
                    ckpt_dir = output_dir / f"checkpoint_{step + 1}"
                    backend.save_checkpoint(ckpt_dir)
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

        return {"metrics_history": metrics_history}

    finally:
        logger.info(f"Shutting down {inference_engine.name}...")
        inference_engine.shutdown(inference_proc)
        logger.info(f"Logs: {inference_engine.log_path}")
