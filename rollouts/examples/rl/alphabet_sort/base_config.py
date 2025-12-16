"""Alphabet Sort RL training config.

Multi-turn task: sort names alphabetically across turns.
Based on Prime-RL's alphabet-sort example.

Baseline Qwen3-4B: ~0.26 reward
After RL (100 steps): ~0.81 reward
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# ──────────────────────── Config Dataclasses ────────────────────────────────


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration."""

    num_samples: int = 500
    min_turns: int = 3
    max_turns: int = 3
    min_names_per_turn: int = 1
    max_names_per_turn: int = 4
    sort_by: str = "last"  # "first" or "last" name
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
    lr: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_minibatches: int = 4


@dataclass(frozen=True)
class OrchestratorConfig:
    """Rollout orchestration configuration."""

    batch_size: int = 4
    rollouts_per_example: int = 8
    max_seq_len: int = 1024
    max_tokens: int = 256
    temperature: float = 0.7


@dataclass(frozen=True)
class RLConfig:
    """Top-level RL configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    num_steps: int = 100
    log_every: int = 1
    checkpoint_every: int = 20

    output_dir: str = "/tmp/rollouts_rl"
    experiment_name: str = "alphabet_sort_grpo"

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ──────────────────────── Name Generation ───────────────────────────────────

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
]

LAST_NAMES = [
    "Adams", "Brown", "Clark", "Davis", "Evans", "Fisher", "Garcia", "Hill",
    "Irwin", "Jones", "King", "Lopez", "Miller", "Nelson", "Ortiz", "Parker",
    "Quinn", "Roberts", "Smith", "Taylor", "Upton", "Vance", "Wilson", "Young",
]


def generate_name(rng: random.Random) -> str:
    """Generate a random full name (no space, concatenated)."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    return f"{first}{last}"


def get_sort_key(name: str, sort_by: str) -> str:
    """Extract sort key from concatenated name."""
    # Find where last name starts (uppercase after lowercase)
    for i in range(1, len(name)):
        if name[i].isupper():
            first_name = name[:i]
            last_name = name[i:]
            break
    else:
        first_name = name
        last_name = ""

    return last_name.lower() if sort_by == "last" else first_name.lower()


def generate_episode(config: DatasetConfig, rng: random.Random) -> dict[str, Any]:
    """Generate a multi-turn alphabet sort episode."""
    sort_by = rng.choice(["first", "last"]) if config.sort_by == "random" else config.sort_by
    num_turns = rng.randint(config.min_turns, config.max_turns)

    turns = []
    all_names = []

    for turn_idx in range(num_turns):
        num_names = rng.randint(config.min_names_per_turn, config.max_names_per_turn)
        new_names = [generate_name(rng) for _ in range(num_names)]
        all_names.extend(new_names)

        # Expected sorted list after this turn
        sorted_names = sorted(all_names, key=lambda n: get_sort_key(n, sort_by))

        turns.append({
            "new_names": new_names,
            "expected_sorted": sorted_names,
            "all_names_so_far": list(all_names),
        })

    return {
        "sort_by": sort_by,
        "turns": turns,
        "final_sorted": sorted(all_names, key=lambda n: get_sort_key(n, sort_by)),
    }


def generate_samples(config: DatasetConfig) -> list[dict[str, Any]]:
    """Generate alphabet sort episodes."""
    rng = random.Random(config.seed)
    return [generate_episode(config, rng) for _ in range(config.num_samples)]


# ──────────────────────── System Prompt ─────────────────────────────────────

SYSTEM_PROMPT = """\
You are sorting names alphabetically. Each turn, you'll receive new names to add to your sorted list.

Rules:
1. Sort by {sort_by} name (names are written as FirstLast with no space)
2. Output the complete sorted list after each turn
3. Mark new names with "// new name!" comment
4. Format: Put sorted names in <combined_alphabetical_sorted> tags

Example for sorting by LAST name:
Turn 1: AliceSmith
<alphabetical_sorted>
AliceSmith
</alphabetical_sorted>

Turn 2: BobJones
<combined_alphabetical_sorted>
BobJones // new name!
AliceSmith
</combined_alphabetical_sorted>
"""


# ──────────────────────── Score Function ────────────────────────────────────


def extract_sorted_list(response: str) -> list[str]:
    """Extract names from response."""
    import re

    # Try to find names in tags
    match = re.search(r"<(?:combined_)?alphabetical_sorted>(.*?)</", response, re.DOTALL)
    if match:
        content = match.group(1)
    else:
        content = response

    # Extract names (CamelCase patterns)
    names = re.findall(r"([A-Z][a-z]+[A-Z][a-z]+)", content)
    return names


def compute_list_similarity(predicted: list[str], expected: list[str]) -> float:
    """Compute similarity between two lists."""
    if not expected:
        return 1.0 if not predicted else 0.0

    # Exact match bonus
    if predicted == expected:
        return 1.0

    # Sequence similarity
    pred_str = " ".join(predicted)
    exp_str = " ".join(expected)
    return SequenceMatcher(None, pred_str, exp_str).ratio()


def alphabet_sort_score_fn(sample: Any) -> Any:
    """Score function for alphabet sort."""
    from rollouts.dtypes import Metric, Score

    expected = sample.metadata.get("expected_sorted", [])
    response = sample.response if hasattr(sample, "response") else ""

    predicted = extract_sorted_list(response)
    similarity = compute_list_similarity(predicted, expected)

    # Apply power scaling like Prime-RL (similarity^8)
    reward = similarity ** 8

    return Score(
        metrics=(
            Metric("reward", reward, weight=1.0),
            Metric("similarity", similarity, weight=0.0),
            Metric("exact_match", 1.0 if predicted == expected else 0.0, weight=0.0),
        )
    )


# ──────────────────────── Training Loop ─────────────────────────────────────


def train(config: RLConfig) -> dict[str, Any]:
    """Run alphabet sort RL training."""
    import trio

    return trio.run(_train_async, config)


async def _train_async(config: RLConfig) -> dict[str, Any]:
    """Async training loop with multi-turn rollouts."""
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
    logger.info(f"Alphabet Sort RL: {config.experiment_name}")
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

    # Generate episodes
    episodes = generate_samples(config.dataset)
    logger.info(f"Generated {len(episodes)} episodes")

    # Flatten to per-turn samples for training
    prompts = []
    for ep in episodes:
        sort_by = ep["sort_by"]
        system = SYSTEM_PROMPT.format(sort_by=sort_by.upper())
        messages = [{"role": "system", "content": system}]

        for turn in ep["turns"]:
            new_names = ", ".join(turn["new_names"])
            user_msg = f"Add these names: {new_names}"
            messages.append({"role": "user", "content": user_msg})

            prompts.append({
                "messages": list(messages),  # Copy
                "expected_sorted": turn["expected_sorted"],
                "sort_by": sort_by,
            })

            # Add placeholder for assistant response (will be generated)
            messages.append({"role": "assistant", "content": "[TO BE GENERATED]"})

    logger.info(f"Flattened to {len(prompts)} turn samples")
    data_buffer = DataBuffer(prompts=prompts)

    async def generate_fn(batch_prompts: list[dict], **kwargs) -> list[Sample]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            results = []
            for prompt_data in batch_prompts:
                messages = prompt_data["messages"]
                expected = prompt_data["expected_sorted"]

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

                full_msgs = messages + [{"role": "assistant", "content": response_text}]
                tokens = tokenizer.apply_chat_template(full_msgs, add_generation_prompt=False)
                prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                prompt_len = len(prompt_tokens)
                loss_mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)

                results.append(
                    Sample(
                        prompt=prompt_data,
                        response=response_text,
                        tokens=tokens,
                        loss_mask=loss_mask,
                        metadata={"expected_sorted": expected},
                    )
                )
            return results

    rollout_config = RolloutConfig(
        batch_size=config.orchestrator.batch_size,
        n_samples_per_prompt=config.orchestrator.rollouts_per_example,
        generate_fn=generate_fn,
        score_fn=alphabet_sort_score_fn,
    )

    inference_engine = SGLangEngine(base_url=f"http://localhost:{inference_port}")
    metrics_history = []

    try:
        async with AsyncRolloutManager(data_buffer, rollout_config) as mgr:
            for step in range(config.num_steps):
                logger.info(f"\n--- Step {step + 1}/{config.num_steps} ---")

                batch = await mgr.generate_batch(score_fn=alphabet_sort_score_fn)
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
                batch_tokens, batch_masks = [], []
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
