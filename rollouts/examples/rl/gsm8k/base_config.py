"""GSM8K RL training config.

Single-turn math reasoning with GRPO.
Model outputs answer in \\boxed{} format.

Based on:
- Miles GSM8K recipe: n_samples_per_prompt=8, lr=1e-6, temp=0.8
- Prime-RL patterns: group-wise advantage normalization
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# ──────────────────────── Config Dataclasses ────────────────────────────────


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration for GSM8K."""

    source: Literal["hf", "jsonl", "parquet"] = "hf"

    # HuggingFace dataset
    hf_dataset: str = "openai/gsm8k"
    hf_subset: str = "main"
    hf_split: str = "train"

    # Field mapping
    prompt_key: str = "question"
    label_key: str = "answer"

    # Limits
    max_samples: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class InferenceConfig:
    """Inference server configuration."""

    provider: str = "sglang"
    port: int = 30000
    gpu_ids: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class TrainerConfig:
    """Training configuration."""

    gpu_ids: tuple[int, ...] = (0,)
    lr: float = 1e-6  # Miles uses 1e-6 for GSM8K
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    num_minibatches: int = 8


@dataclass(frozen=True)
class OrchestratorConfig:
    """Rollout orchestration configuration."""

    batch_size: int = 8  # Unique prompts per step
    rollouts_per_example: int = 8  # N samples per prompt (Miles uses 8)
    max_seq_len: int = 1024
    max_tokens: int = 512  # Max generation length
    temperature: float = 0.8  # Miles uses 0.8


@dataclass(frozen=True)
class RLConfig:
    """Top-level RL configuration for GSM8K."""

    # Nested configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    # Training loop
    num_steps: int = 100
    log_every: int = 1
    checkpoint_every: int = 20
    eval_every: int = 20

    # Output
    output_dir: str = "/tmp/rollouts_rl"
    experiment_name: str = "gsm8k_grpo"

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ──────────────────────── Dataset Loading ───────────────────────────────────


def load_gsm8k_samples(config: DatasetConfig) -> list[dict[str, Any]]:
    """Load GSM8K samples with answer extraction."""
    from datasets import load_dataset

    ds = load_dataset(config.hf_dataset, config.hf_subset, split=config.hf_split)

    samples = []
    for i, row in enumerate(ds):
        if config.max_samples and i >= config.max_samples:
            break

        # Extract final answer from solution (format: "#### 42")
        solution = row[config.label_key]
        match = re.search(r"####\s*([\d,.-]+)", solution)
        answer = match.group(1).replace(",", "") if match else ""

        samples.append({
            "prompt": row[config.prompt_key],
            "answer": answer,
            "full_solution": solution,
        })

    return samples


# ──────────────────────── System Prompt ─────────────────────────────────────

SYSTEM_PROMPT = """\
Solve the following math problem step by step.
Show your reasoning clearly, then put your final numerical answer in \\boxed{}.

Example format:
Step 1: ...
Step 2: ...
Therefore, the answer is \\boxed{42}
"""


# ──────────────────────── Score Function ────────────────────────────────────


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if match:
        answer = match.group(1).strip()
        answer = answer.replace(",", "").replace("$", "").strip()
        return answer
    return None


def normalize_answer(answer: str) -> float | None:
    """Normalize answer string to float."""
    if not answer:
        return None
    try:
        if "/" in answer:
            parts = answer.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        if "%" in answer:
            return float(answer.replace("%", "")) / 100
        return float(answer)
    except ValueError:
        return None


def gsm8k_score_fn(sample: Any) -> Any:
    """Score function for GSM8K.

    Returns Score with reward=1.0 if correct, 0.0 otherwise.
    """
    from rollouts.dtypes import Metric, Score

    ground_truth = sample.metadata.get("answer")
    if ground_truth is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    response = sample.response if hasattr(sample, "response") else ""
    predicted = extract_boxed_answer(response)

    if predicted is None:
        return Score(
            metrics=(
                Metric("correct", 0.0, weight=1.0),
                Metric("parse_failed", 1.0, weight=0.0),
            )
        )

    pred_val = normalize_answer(predicted)
    true_val = normalize_answer(ground_truth)

    if pred_val is None or true_val is None:
        return Score(
            metrics=(
                Metric("correct", 0.0, weight=1.0),
                Metric("parse_failed", 1.0, weight=0.0),
            )
        )

    is_correct = abs(pred_val - true_val) < 0.01
    return Score(
        metrics=(
            Metric("correct", 1.0 if is_correct else 0.0, weight=1.0),
        )
    )


# ──────────────────────── Training Loop ─────────────────────────────────────


def train(config: RLConfig) -> dict[str, Any]:
    """Run GSM8K RL training with GRPO."""
    import trio

    return trio.run(_train_async, config)


async def _train_async(config: RLConfig) -> dict[str, Any]:
    """Async training loop for GSM8K."""
    import subprocess
    import time

    import httpx
    import torch
    import trio

    from rollouts._logging import setup_logging
    from rollouts.training.backends.pytorch import PyTorchTrainingBackend
    from rollouts.training.datasets.data_buffer import DataBuffer
    from rollouts.training.losses import compute_group_advantages, grpo_loss
    from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
    from rollouts.training.types import RolloutConfig, Sample
    from rollouts.training.weight_sync import SGLangEngine

    # Setup logging
    setup_logging(
        level="INFO",
        use_color=True,
        logger_levels={"httpx": "WARNING", "httpcore": "WARNING"},
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"GSM8K RL Training: {config.experiment_name}")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Steps: {config.num_steps}")
    logger.info(f"Batch: {config.orchestrator.batch_size} prompts x {config.orchestrator.rollouts_per_example} samples")

    # Output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Launch inference server
    # ─────────────────────────────────────────────────────────────────────────
    inference_port = config.inference.port
    inference_gpu_ids = config.inference.gpu_ids
    gpu_str = ",".join(str(g) for g in inference_gpu_ids)

    sglang_log_file = output_dir / "sglang_server.log"
    sglang_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_str} "
        f"python -m sglang.launch_server "
        f"--model-path {config.model.name} "
        f"--port {inference_port} "
        f"--dtype {config.model.dtype} "
        f"--mem-fraction-static 0.7 "
        f">> {sglang_log_file} 2>&1"
    )

    logger.info(f"Launching SGLang on GPU {gpu_str}...")
    sglang_proc = subprocess.Popen(sglang_cmd, shell=True, start_new_session=True)

    # Wait for server
    async with httpx.AsyncClient(timeout=5.0) as client:
        for attempt in range(120):
            try:
                resp = await client.get(f"http://localhost:{inference_port}/health")
                if resp.status_code == 200:
                    logger.info(f"SGLang ready after {attempt + 1}s")
                    break
            except Exception:
                pass
            await trio.sleep(1.0)
        else:
            raise RuntimeError("SGLang failed to start")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Setup training backend
    # ─────────────────────────────────────────────────────────────────────────
    device = f"cuda:{config.trainer.gpu_ids[0]}"
    backend = PyTorchTrainingBackend.from_pretrained(
        model_name=config.model.name,
        device=device,
        dtype=config.model.dtype,
        lr=config.trainer.lr,
        weight_decay=config.trainer.weight_decay,
        max_grad_norm=config.trainer.max_grad_norm,
        loss_fn=lambda logits, batch: grpo_loss(logits, batch),
    )
    tokenizer = backend.tokenizer

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Setup data and rollout generation
    # ─────────────────────────────────────────────────────────────────────────
    samples = load_gsm8k_samples(config.dataset)
    logger.info(f"Dataset: {len(samples)} samples")

    # Create prompts with chat format
    prompts = []
    for s in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["prompt"]},
        ]
        prompts.append({"messages": messages, "answer": s["answer"]})

    data_buffer = DataBuffer(prompts=prompts)

    # Generation function
    async def generate_fn(
        batch_prompts: list[dict],
        **kwargs,
    ) -> list[Sample]:
        """Generate completions via SGLang."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            results = []
            for prompt_data in batch_prompts:
                messages = prompt_data["messages"]
                answer = prompt_data["answer"]

                resp = await client.post(
                    f"http://localhost:{inference_port}/v1/chat/completions",
                    json={
                        "model": config.model.name,
                        "messages": messages,
                        "max_tokens": config.orchestrator.max_tokens,
                        "temperature": config.orchestrator.temperature,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                response_text = data["choices"][0]["message"]["content"]

                # Tokenize full conversation for training
                full_messages = messages + [{"role": "assistant", "content": response_text}]
                encoded = tokenizer.apply_chat_template(
                    full_messages,
                    add_generation_prompt=False,
                    return_dict=True,
                )
                tokens = encoded["input_ids"]

                # Create loss mask (only train on assistant response)
                prompt_tokens = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_dict=True,
                )["input_ids"]
                prompt_len = len(prompt_tokens)
                loss_mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)

                results.append(
                    Sample(
                        prompt=prompt_data,
                        response=response_text,
                        tokens=tokens,
                        loss_mask=loss_mask,
                        metadata={"answer": answer},
                    )
                )
            return results

    rollout_config = RolloutConfig(
        batch_size=config.orchestrator.batch_size,
        n_samples_per_prompt=config.orchestrator.rollouts_per_example,
        over_sampling_factor=1.0,
        generate_fn=generate_fn,
        score_fn=gsm8k_score_fn,
    )

    inference_engine = SGLangEngine(base_url=f"http://localhost:{inference_port}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Training loop
    # ─────────────────────────────────────────────────────────────────────────
    metrics_history = []

    try:
        async with AsyncRolloutManager(data_buffer, rollout_config) as rollout_manager:
            for step in range(config.num_steps):
                logger.info(f"\n--- Step {step + 1}/{config.num_steps} ---")

                # 1. Generate rollouts
                batch = await rollout_manager.generate_batch(score_fn=gsm8k_score_fn)

                if not batch.tokens:
                    logger.warning("No successful rollouts, skipping step")
                    continue

                # 2. Compute group-wise advantages
                rewards = batch.rewards
                group_indices = batch.group_indices

                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                num_groups = len(set(group_indices)) if group_indices else len(rewards)
                logger.info(f"Reward: {mean_reward:.3f} ({len(rewards)} samples, {num_groups} groups)")

                if group_indices and len(set(group_indices)) > 1:
                    advantages = compute_group_advantages(rewards, group_indices).to(device)
                else:
                    baseline = mean_reward
                    advantages = torch.tensor([r - baseline for r in rewards], device=device)

                # 3. Prepare batch
                max_len = min(
                    max(len(t) for t in batch.tokens),
                    config.orchestrator.max_seq_len,
                )

                batch_tokens = []
                batch_loss_masks = []
                for tokens, loss_mask in zip(batch.tokens, batch.loss_masks, strict=True):
                    tokens = list(tokens[:max_len])
                    loss_mask = list(loss_mask[:max_len])
                    pad_len = max_len - len(tokens)
                    tokens = tokens + [tokenizer.pad_token_id or 0] * pad_len
                    loss_mask = loss_mask + [0.0] * pad_len
                    batch_tokens.append(tokens)
                    batch_loss_masks.append(loss_mask)

                input_ids = torch.tensor(batch_tokens, device=device)
                labels = input_ids.clone()
                loss_mask = torch.tensor(batch_loss_masks, device=device)

                # 4. Training step with gradient accumulation
                batch_size = input_ids.size(0)
                num_minibatches = min(config.trainer.num_minibatches, batch_size)
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
                        accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v / num_minibatches

                pg_loss = accumulated_metrics.get("pg_loss", 0.0)
                entropy = accumulated_metrics.get("entropy", 0.0)

                # 5. Log
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

                # 6. Checkpoint and sync weights
                if (step + 1) % config.checkpoint_every == 0:
                    ckpt_dir = output_dir / f"hf_checkpoint_{step + 1}"
                    backend.save_checkpoint(ckpt_dir)
                    logger.info(f"Saved checkpoint: {ckpt_dir}")

                    logger.info("Syncing weights to SGLang...")
                    await inference_engine.update_weights_from_disk(str(ckpt_dir))
                    logger.info("Weight sync complete")

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)

        if metrics_history:
            first_reward = metrics_history[0]["mean_reward"]
            last_reward = metrics_history[-1]["mean_reward"]
            first_loss = metrics_history[0]["pg_loss"]
            last_loss = metrics_history[-1]["pg_loss"]
            logger.info(f"First: reward={first_reward:.3f}, pg_loss={first_loss:.4f}")
            logger.info(f"Last:  reward={last_reward:.3f}, pg_loss={last_loss:.4f}")

        return {"metrics_history": metrics_history}

    finally:
        logger.info("Shutting down SGLang...")
        sglang_proc.terminate()
        logger.info(f"Logs: {sglang_log_file}")
