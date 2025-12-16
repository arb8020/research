"""Reverse Text RL training config.

Simple single-turn task: reverse a string.
Based on Prime-RL's reverse-text example.

Baseline Qwen3-0.6B: ~0.05 reward
After RL (20 steps): ~0.8 reward
"""

from __future__ import annotations

import json
import logging
import random
import string
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# ──────────────────────── Config Dataclasses ────────────────────────────────


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration for reverse text."""

    num_samples: int = 1000
    min_length: int = 10
    max_length: int = 50
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
    lr: float = 3e-6  # Prime-RL uses 3e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_minibatches: int = 8


@dataclass(frozen=True)
class OrchestratorConfig:
    """Rollout orchestration configuration."""

    batch_size: int = 8
    rollouts_per_example: int = 16  # Prime-RL uses 16
    max_seq_len: int = 256
    max_tokens: int = 128
    temperature: float = 0.7


@dataclass(frozen=True)
class RLConfig:
    """Top-level RL configuration for reverse text."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    num_steps: int = 20  # Prime-RL uses 20 steps
    log_every: int = 1
    checkpoint_every: int = 10

    output_dir: str = "/tmp/rollouts_rl"
    experiment_name: str = "reverse_text_grpo"

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ──────────────────────── Dataset Generation ────────────────────────────────


def generate_random_text(length: int, rng: random.Random) -> str:
    """Generate random text to reverse."""
    # Mix of words and characters
    words = ["hello", "world", "python", "code", "test", "data", "model", "train"]
    text = []
    while len(" ".join(text)) < length:
        if rng.random() < 0.7:
            text.append(rng.choice(words))
        else:
            text.append("".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8))))
    return " ".join(text)[:length]


def generate_samples(config: DatasetConfig) -> list[dict[str, str]]:
    """Generate reverse text samples."""
    rng = random.Random(config.seed)
    samples = []

    for _ in range(config.num_samples):
        length = rng.randint(config.min_length, config.max_length)
        text = generate_random_text(length, rng)
        reversed_text = text[::-1]

        samples.append({
            "text": text,
            "reversed": reversed_text,
        })

    return samples


# ──────────────────────── System Prompt ─────────────────────────────────────

SYSTEM_PROMPT = """\
You are a text reversal assistant. When given text, reverse it character by character.

Example:
Input: hello world
Output: dlrow olleh

Just output the reversed text, nothing else.
"""


# ──────────────────────── Score Function ────────────────────────────────────


def compute_similarity(predicted: str, expected: str) -> float:
    """Compute string similarity using SequenceMatcher."""
    return SequenceMatcher(None, predicted.strip(), expected.strip()).ratio()


def reverse_text_score_fn(sample: Any) -> Any:
    """Score function for reverse text.

    Returns similarity score between prediction and expected reversal.
    """
    from rollouts.dtypes import Metric, Score

    expected = sample.metadata.get("reversed", "")
    response = sample.response if hasattr(sample, "response") else ""

    # Clean up response (remove quotes, extra whitespace)
    response = response.strip().strip('"\'')

    similarity = compute_similarity(response, expected)

    return Score(
        metrics=(
            Metric("similarity", similarity, weight=1.0),
            Metric("exact_match", 1.0 if response == expected else 0.0, weight=0.0),
        )
    )


# ──────────────────────── Training Loop ─────────────────────────────────────


def train(config: RLConfig) -> dict[str, Any]:
    """Run reverse text RL training."""
    import trio

    return trio.run(_train_async, config)


async def _train_async(config: RLConfig) -> dict[str, Any]:
    """Async training loop."""
    import subprocess

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

    setup_logging(
        level="INFO",
        use_color=True,
        logger_levels={"httpx": "WARNING", "httpcore": "WARNING"},
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"Reverse Text RL: {config.experiment_name}")
    logger.info("=" * 60)

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")

    # Launch SGLang
    inference_port = config.inference.port
    gpu_str = ",".join(str(g) for g in config.inference.gpu_ids)
    sglang_log = output_dir / "sglang.log"

    sglang_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_str} "
        f"python -m sglang.launch_server "
        f"--model-path {config.model.name} "
        f"--port {inference_port} "
        f"--dtype {config.model.dtype} "
        f"--mem-fraction-static 0.7 "
        f">> {sglang_log} 2>&1"
    )

    logger.info(f"Launching SGLang on GPU {gpu_str}...")
    sglang_proc = subprocess.Popen(sglang_cmd, shell=True, start_new_session=True)

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

    # Setup backend
    device = f"cuda:{config.trainer.gpu_ids[0]}"
    backend = PyTorchTrainingBackend.from_pretrained(
        model_name=config.model.name,
        device=device,
        dtype=config.model.dtype,
        lr=config.trainer.lr,
        loss_fn=lambda logits, batch: grpo_loss(logits, batch),
    )
    tokenizer = backend.tokenizer

    # Generate samples
    samples = generate_samples(config.dataset)
    logger.info(f"Generated {len(samples)} samples")

    prompts = []
    for s in samples:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s["text"]},
        ]
        prompts.append({"messages": messages, "reversed": s["reversed"]})

    data_buffer = DataBuffer(prompts=prompts)

    async def generate_fn(batch_prompts: list[dict], **kwargs) -> list[Sample]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            results = []
            for prompt_data in batch_prompts:
                messages = prompt_data["messages"]
                reversed_text = prompt_data["reversed"]

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

                full_messages = messages + [{"role": "assistant", "content": response_text}]
                tokens = tokenizer.apply_chat_template(full_messages, add_generation_prompt=False)
                prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                prompt_len = len(prompt_tokens)
                loss_mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)

                results.append(
                    Sample(
                        prompt=prompt_data,
                        response=response_text,
                        tokens=tokens,
                        loss_mask=loss_mask,
                        metadata={"reversed": reversed_text},
                    )
                )
            return results

    rollout_config = RolloutConfig(
        batch_size=config.orchestrator.batch_size,
        n_samples_per_prompt=config.orchestrator.rollouts_per_example,
        generate_fn=generate_fn,
        score_fn=reverse_text_score_fn,
    )

    inference_engine = SGLangEngine(base_url=f"http://localhost:{inference_port}")
    metrics_history = []

    try:
        async with AsyncRolloutManager(data_buffer, rollout_config) as mgr:
            for step in range(config.num_steps):
                logger.info(f"\n--- Step {step + 1}/{config.num_steps} ---")

                batch = await mgr.generate_batch(score_fn=reverse_text_score_fn)
                if not batch.tokens:
                    continue

                rewards = batch.rewards
                group_indices = batch.group_indices
                mean_reward = sum(rewards) / len(rewards)
                num_groups = len(set(group_indices)) if group_indices else len(rewards)

                logger.info(f"Reward: {mean_reward:.3f} ({len(rewards)} samples, {num_groups} groups)")

                if group_indices and len(set(group_indices)) > 1:
                    advantages = compute_group_advantages(rewards, group_indices).to(device)
                else:
                    advantages = torch.tensor([r - mean_reward for r in rewards], device=device)

                max_len = min(max(len(t) for t in batch.tokens), config.orchestrator.max_seq_len)
                batch_tokens = []
                batch_masks = []
                for tokens, mask in zip(batch.tokens, batch.loss_masks):
                    t = list(tokens[:max_len])
                    m = list(mask[:max_len])
                    pad = max_len - len(t)
                    t += [tokenizer.pad_token_id or 0] * pad
                    m += [0.0] * pad
                    batch_tokens.append(t)
                    batch_masks.append(m)

                input_ids = torch.tensor(batch_tokens, device=device)
                loss_mask = torch.tensor(batch_masks, device=device)

                num_mb = min(config.trainer.num_minibatches, input_ids.size(0))
                mb_size = input_ids.size(0) // num_mb
                acc_metrics: dict[str, float] = {}

                for i in range(num_mb):
                    s, e = i * mb_size, (i + 1) * mb_size
                    mb = {
                        "input_ids": input_ids[s:e],
                        "labels": input_ids[s:e].clone(),
                        "loss_mask": loss_mask[s:e],
                        "advantages": advantages[s:e],
                    }
                    m = backend.forward_backward(mb, num_mb, step_optimizer=(i == num_mb - 1))
                    for k, v in m.items():
                        acc_metrics[k] = acc_metrics.get(k, 0.0) + v / num_mb

                metrics_history.append({"step": step + 1, "mean_reward": mean_reward, **acc_metrics})
                logger.info(f"pg_loss={acc_metrics.get('pg_loss', 0):.4f}")

                if (step + 1) % config.checkpoint_every == 0:
                    ckpt = output_dir / f"checkpoint_{step + 1}"
                    backend.save_checkpoint(ckpt)
                    await inference_engine.update_weights_from_disk(str(ckpt))
                    logger.info(f"Checkpoint: {ckpt}")

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        if metrics_history:
            logger.info(f"First reward: {metrics_history[0]['mean_reward']:.3f}")
            logger.info(f"Last reward:  {metrics_history[-1]['mean_reward']:.3f}")

        return {"metrics_history": metrics_history}

    finally:
        sglang_proc.terminate()
