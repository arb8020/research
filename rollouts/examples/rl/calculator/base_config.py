"""Base RL config and training logic for Calculator environment.

Experiment files import from here and override config values.
Following experiment_config.md: Pythonic + Hierarchical + Serializable.

Directory: examples/rl/calculator/
- grpo_01_01.py: Base GRPO config (parent is self)
- grpo_high_lr_02_01.py: Derived from 01, higher learning rate
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass


# ──────────────────────── Config Dataclasses ────────────────────────────────


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset configuration for RL training data.

    Supports multiple sources via the `source` field:
    - "hf": HuggingFace datasets
    - "jsonl": Local JSONL file
    - "parquet": Local Parquet file
    - "builtin": Built-in dataset (e.g., calculator tasks)
    """

    source: Literal["hf", "jsonl", "parquet", "builtin"] = "builtin"

    # For HuggingFace datasets
    hf_dataset: str | None = None
    hf_subset: str | None = None
    hf_split: str = "train"

    # For local files
    path: str | None = None

    # Field mapping
    prompt_key: str = "prompt"
    label_key: str | None = None

    # Limits
    max_samples: int | None = None

    # Shuffling
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""

    name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"


@dataclass(frozen=True)
class InferenceConfig:
    """Inference server configuration."""

    provider: str = "sglang"  # sglang or vllm
    port: int = 30000
    gpu_ids: tuple[int, ...] = (0,)


@dataclass(frozen=True)
class TrainerConfig:
    """Training configuration."""

    gpu_ids: tuple[int, ...] = (0,)  # Same GPU for small models
    lr: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    # Gradient accumulation (Tinker naming: num_minibatches)
    # With batch_size=8 * rollouts_per_example=4 = 32 sequences,
    # split into 8 minibatches = 4 sequences per forward/backward
    num_minibatches: int = 8


@dataclass(frozen=True)
class OrchestratorConfig:
    """Rollout orchestration configuration."""

    batch_size: int = 8
    rollouts_per_example: int = 4
    max_seq_len: int = 2048
    max_turns: int = 10


@dataclass(frozen=True)
class RLConfig:
    """Top-level RL configuration.

    Following experiment_config.md pattern:
    - Hierarchical dataclasses
    - Frozen for immutability
    - Serializable to JSON
    """

    # Nested configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    # Training loop
    num_steps: int = 10
    log_every: int = 1
    checkpoint_every: int = 5

    # Output
    output_dir: str = "/tmp/rollouts_rl"
    experiment_name: str = "calculator_grpo"

    def save(self, path: Path | str) -> None:
        """Save config to JSON for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path | str) -> RLConfig:
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            dataset=DatasetConfig(**data.get("dataset", {})),
            model=ModelConfig(**data["model"]),
            inference=InferenceConfig(**data["inference"]),
            trainer=TrainerConfig(**data["trainer"]),
            orchestrator=OrchestratorConfig(**data["orchestrator"]),
            num_steps=data["num_steps"],
            log_every=data["log_every"],
            checkpoint_every=data["checkpoint_every"],
            output_dir=data["output_dir"],
            experiment_name=data["experiment_name"],
        )


# ──────────────────────── Dataset ───────────────────────────────────────────


def load_samples_from_config(config: DatasetConfig) -> list[dict[str, Any]]:
    """Load samples based on DatasetConfig.

    Returns list of dicts with prompt and metadata for RL.
    """
    from rollouts.training.datasets.data_buffer import (
        load_samples_from_hf,
        load_samples_from_jsonl,
        load_samples_from_parquet,
    )

    if config.source == "builtin":
        return load_calculator_prompts(config.max_samples)
    elif config.source == "hf":
        assert config.hf_dataset, "hf_dataset required for hf source"
        samples = load_samples_from_hf(
            dataset_name=config.hf_dataset,
            subset=config.hf_subset,
            split=config.hf_split,
            prompt_key=config.prompt_key,
            label_key=config.label_key,
            limit=config.max_samples,
        )
        # Convert Sample objects to dicts for RL
        return [{"prompt": s.prompt, "ground_truth": s.metadata.get("label")} for s in samples]
    elif config.source == "jsonl":
        assert config.path, "path required for jsonl source"
        samples = load_samples_from_jsonl(
            path=Path(config.path),
            prompt_key=config.prompt_key,
            label_key=config.label_key,
            limit=config.max_samples,
        )
        return [{"prompt": s.prompt, "ground_truth": s.metadata.get("label")} for s in samples]
    elif config.source == "parquet":
        assert config.path, "path required for parquet source"
        samples = load_samples_from_parquet(
            path=config.path,
            prompt_key=config.prompt_key,
            label_key=config.label_key,
            limit=config.max_samples,
        )
        return [{"prompt": s.prompt, "ground_truth": s.metadata.get("label")} for s in samples]
    else:
        raise ValueError(f"Unknown source: {config.source}")


def load_calculator_prompts(max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load calculator task prompts.

    Returns list of dicts with:
        - prompt: str (the task)
        - ground_truth: float (expected answer)
    """
    # Simple arithmetic tasks for testing
    tasks = [
        {"prompt": "What is 5 + 3?", "ground_truth": 8.0},
        {"prompt": "What is 12 - 7?", "ground_truth": 5.0},
        {"prompt": "What is 6 * 4?", "ground_truth": 24.0},
        {"prompt": "What is 20 / 5?", "ground_truth": 4.0},
        {"prompt": "What is 15 + 7 - 3?", "ground_truth": 19.0},
        {"prompt": "What is 8 * 3 + 2?", "ground_truth": 26.0},
        {"prompt": "What is 100 / 4 - 10?", "ground_truth": 15.0},
        {"prompt": "What is 7 + 8 * 2?", "ground_truth": 23.0},  # Tests order of ops
        {"prompt": "What is 50 - 25 + 10?", "ground_truth": 35.0},
        {"prompt": "What is 9 * 9?", "ground_truth": 81.0},
        {"prompt": "What is 144 / 12?", "ground_truth": 12.0},
        {"prompt": "What is 33 + 67?", "ground_truth": 100.0},
    ]

    if max_samples is not None:
        tasks = tasks[:max_samples]

    return tasks


# ──────────────────────── Reward Function ───────────────────────────────────


def calculator_score_fn(sample) -> Score:
    """Score function for calculator tasks.

    Compares final_result from complete_task tool to ground_truth.
    Returns Score with reward 1.0 if correct, 0.0 otherwise.

    Score uses metrics with weights - weighted average becomes .reward
    """
    from rollouts.dtypes import Metric, Score

    ground_truth = sample.metadata.get("ground_truth")
    if ground_truth is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Extract final_result from trajectory
    # Look for complete_task tool call result
    final_result = None
    response = sample.response

    # Simple heuristic: look for "Final result: X" in response
    import re

    match = re.search(r"Final result:\s*([\d.-]+)", response)
    if match:
        try:
            final_result = float(match.group(1))
        except ValueError:
            pass

    if final_result is None:
        return Score(metrics=(Metric("correct", 0.0, weight=1.0),))

    # Check if correct (with tolerance for floating point)
    is_correct = abs(final_result - ground_truth) < 0.01
    reward = 1.0 if is_correct else 0.0

    return Score(
        metrics=(
            Metric("correct", reward, weight=1.0),
            Metric("final_result", final_result, weight=0.0),  # tracked only
            Metric("ground_truth", ground_truth, weight=0.0),  # tracked only
        )
    )


# ──────────────────────── Training Loop ─────────────────────────────────────


async def _train_async(config: RLConfig) -> list[dict[str, Any]]:
    """Async RL training implementation.

    Full loop:
    1. Launch inference server (SGLang)
    2. Load training model
    3. For each step:
       - Generate rollouts via inference server
       - Compute rewards
       - GRPO loss + backward
       - (TODO: sync weights to inference server)
    4. Shutdown inference server
    """
    import subprocess
    import time

    import requests
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rollouts._logging import setup_logging
    from rollouts.dtypes import Endpoint
    from rollouts.environments.calculator import CalculatorEnvironment
    from rollouts.training.agent_integration import agent_rollout_to_sample
    from rollouts.training.datasets.data_buffer import DataBuffer
    from rollouts.training.losses import grpo_loss
    from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
    from rollouts.training.types import RolloutConfig
    from rollouts.training.weight_sync import SGLangEngine, sync_weights_to_engines

    # Setup logging (suppress noisy HTTP client logs)
    # Use JSON format for TUI parsing, fall back to color for direct terminal use
    use_json_logs = os.getenv("ROLLOUTS_JSON_LOGS", "").lower() == "true"
    setup_logging(
        level="INFO",
        use_json=use_json_logs,
        use_color=not use_json_logs,
        logger_levels={
            "httpx": "WARNING",
            "httpcore": "WARNING",
        },
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"RL Training: {config.experiment_name}")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Steps: {config.num_steps}")
    logger.info(f"Batch size: {config.orchestrator.batch_size}")
    logger.info(f"Rollouts per example: {config.orchestrator.rollouts_per_example}")

    # Output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")
    logger.info(f"Output: {output_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Launch inference server (SGLang) in tmux for debugging
    # Pattern from ~/wafer_stuff/clicker - launch in tmux with log file
    # ─────────────────────────────────────────────────────────────────────────
    inference_gpu_ids = config.inference.gpu_ids
    inference_port = config.inference.port
    tmux_session = "sglang-rl"
    sglang_log = output_dir / "sglang_server.log"

    logger.info(f"Launching SGLang on GPUs {inference_gpu_ids}, port {inference_port}...")
    logger.info(f"  Log file: {sglang_log}")
    logger.info(f"  Tmux session: {tmux_session}")

    # Build SGLang command
    sglang_cmd_parts = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model",
        config.model.name,
        "--host",
        "0.0.0.0",
        "--port",
        str(inference_port),
        "--trust-remote-code",
        "--tool-call-parser",
        "qwen25",  # Enable function calling for Qwen2.5
    ]

    # Multi-GPU: add tensor parallel size if more than 1 GPU
    if len(inference_gpu_ids) > 1:
        sglang_cmd_parts.extend(["--tp-size", str(len(inference_gpu_ids))])

    sglang_cmd = " ".join(sglang_cmd_parts)

    # Kill existing tmux session if present
    subprocess.run(["tmux", "kill-session", "-t", tmux_session], capture_output=True)

    # Kill any orphaned processes using our GPUs
    all_gpu_ids = set(inference_gpu_ids) | set(config.trainer.gpu_ids)
    for gpu_id in all_gpu_ids:
        subprocess.run(
            f"nvidia-smi --id={gpu_id} --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
            shell=True,
            capture_output=True,
        )

    # Build full command with environment and logging
    cuda_devices = ",".join(str(g) for g in inference_gpu_ids)
    full_cmd = f"CUDA_VISIBLE_DEVICES={cuda_devices} {sglang_cmd} 2>&1 | tee {sglang_log}"

    # Launch in tmux (allows inspection if something goes wrong)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", tmux_session, full_cmd],
        check=True,
    )
    logger.info(f"  Started SGLang in tmux session '{tmux_session}'")
    logger.info(f"  To attach: tmux attach -t {tmux_session}")

    # Wait for server to be ready
    health_url = f"http://localhost:{inference_port}/health"
    server_ready = False
    for attempt in range(120):  # 2 minutes timeout
        # Check if tmux session is still alive
        result = subprocess.run(
            ["tmux", "has-session", "-t", tmux_session],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.error("SGLang server crashed during startup!")
            logger.error(f"Check logs: tail -50 {sglang_log}")
            raise RuntimeError("SGLang server crashed - check logs")

        try:
            resp = requests.get(health_url, timeout=1)
            if resp.status_code == 200:
                server_ready = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        if attempt % 10 == 0:
            logger.info(f"  Waiting for SGLang... ({attempt}s)")

    if not server_ready:
        subprocess.run(["tmux", "kill-session", "-t", tmux_session], capture_output=True)
        logger.error(f"SGLang failed to start. Check logs: {sglang_log}")
        raise RuntimeError("SGLang server failed to start within 2 minutes")

    logger.info(f"SGLang ready at http://localhost:{inference_port}")

    # Start tailing SGLang log in background (emits JSONL for TUI)
    import threading

    def tail_sglang_log():
        """Tail SGLang log and emit as JSONL."""
        try:
            with open(sglang_log) as f:
                # Start from end of file (don't replay startup logs)
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        line = line.strip()
                        if line:
                            print(json.dumps({"logger": "sglang", "message": line}), flush=True)
                    else:
                        time.sleep(0.1)
        except Exception:
            pass  # File closed or thread killed

    sglang_tailer = threading.Thread(target=tail_sglang_log, daemon=True)
    sglang_tailer.start()

    try:
        # ─────────────────────────────────────────────────────────────────────
        # 2. Load training model
        # ─────────────────────────────────────────────────────────────────────
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading training model...")
        trainer_gpu_ids = config.trainer.gpu_ids
        # TODO: For multi-GPU training, use FSDP. For now, use first GPU.
        if len(trainer_gpu_ids) > 1:
            logger.warning(
                f"Multi-GPU training ({trainer_gpu_ids}) requested but FSDP not implemented. "
                f"Using single GPU: {trainer_gpu_ids[0]}"
            )
        device = f"cuda:{trainer_gpu_ids[0]}"
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=getattr(torch, config.model.dtype),
            device_map=device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
        )
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {param_count / 1e6:.1f}M params on {device}")

        # Create endpoint for inference
        endpoint = Endpoint(
            provider="openai",  # SGLang exposes OpenAI-compatible API
            model=config.model.name,
            api_base=f"http://localhost:{inference_port}/v1",
        )

        # Load dataset
        tasks = load_samples_from_config(config.dataset)
        logger.info(f"Loaded {len(tasks)} tasks")

        # ─────────────────────────────────────────────────────────────────────
        # 3. Setup AsyncRolloutManager
        # ─────────────────────────────────────────────────────────────────────

        # Create generate_fn that wraps agent_rollout_to_sample
        async def generate_fn(prompts: list[str | dict]) -> list:
            """Generate samples for a batch of prompts."""

            results = []
            for prompt_data in prompts:
                # Handle both str and dict prompts
                if isinstance(prompt_data, dict):
                    prompt = prompt_data["prompt"]
                    metadata = {"ground_truth": prompt_data.get("ground_truth")}
                else:
                    prompt = prompt_data
                    metadata = {}

                try:
                    sample = await agent_rollout_to_sample(
                        prompt=prompt,
                        environment_cls=CalculatorEnvironment,
                        endpoint=endpoint,
                        tokenizer=tokenizer,
                        max_turns=config.orchestrator.max_turns,
                        metadata=metadata,
                    )
                    results.append(sample)
                except Exception as e:
                    logger.warning(
                        f"Rollout failed for prompt '{prompt[:50]}...': {e}", exc_info=True
                    )

            return results

        # Setup DataBuffer with task prompts (includes ground_truth metadata)
        data_buffer = DataBuffer(prompts=tasks)

        # Setup RolloutConfig
        rollout_config = RolloutConfig(
            batch_size=config.orchestrator.batch_size,
            n_samples_per_prompt=config.orchestrator.rollouts_per_example,
            over_sampling_factor=1.2,  # Generate 20% extra, keep best
            generate_fn=generate_fn,
            score_fn=calculator_score_fn,
        )

        # Setup inference engine for weight sync
        inference_engine = SGLangEngine(base_url=f"http://localhost:{inference_port}")

        # ─────────────────────────────────────────────────────────────────────
        # 4. Training loop with AsyncRolloutManager
        # ─────────────────────────────────────────────────────────────────────
        metrics_history = []
        sync_every = config.checkpoint_every  # Sync weights when we checkpoint

        async with AsyncRolloutManager(data_buffer, rollout_config) as rollout_manager:
            for step in range(config.num_steps):
                logger.info(f"\n--- Step {step + 1}/{config.num_steps} ---")

                # 1. Generate rollouts via AsyncRolloutManager
                logger.info(
                    f"Generating batch (size={config.orchestrator.batch_size}, "
                    f"rollouts_per_example={config.orchestrator.rollouts_per_example})..."
                )
                batch = await rollout_manager.generate_batch(score_fn=calculator_score_fn)

                if not batch.tokens:
                    logger.warning("No successful rollouts, skipping step")
                    continue

                # 2. Extract rewards (already computed by score_fn in generate_batch)
                rewards = batch.rewards
                group_indices = batch.group_indices

                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                num_groups = len(set(group_indices)) if group_indices else len(rewards)
                logger.info(
                    f"Mean reward: {mean_reward:.3f} ({len(rewards)} samples, {num_groups} groups)"
                )

                # 3. Compute advantages (GRPO: group-normalized)
                # Each sample's advantage = reward - mean(rewards in same group)
                # This is the "G" (Group) in GRPO
                from rollouts.training.losses import compute_group_advantages

                if group_indices and len(set(group_indices)) > 1:
                    # Use group-wise normalization (proper GRPO)
                    advantages = compute_group_advantages(rewards, group_indices).to(device)
                else:
                    # Fallback to batch-wise normalization (single group or no groups)
                    baseline = mean_reward
                    advantages = torch.tensor([r - baseline for r in rewards], device=device)

                # 4. Prepare batch for training
                # Pad sequences to same length
                max_len = min(
                    max(len(toks) for toks in batch.tokens),
                    config.orchestrator.max_seq_len,
                )

                batch_tokens = []
                batch_loss_masks = []
                for tokens, loss_mask in zip(batch.tokens, batch.loss_masks, strict=True):
                    tokens = list(tokens[:max_len])
                    loss_mask = list(loss_mask[:max_len])

                    # Pad to max_len
                    pad_len = max_len - len(tokens)
                    tokens = tokens + [tokenizer.pad_token_id] * pad_len
                    loss_mask = loss_mask + [0.0] * pad_len

                    batch_tokens.append(tokens)
                    batch_loss_masks.append(loss_mask)

                # Convert to tensors
                input_ids = torch.tensor(batch_tokens, device=device)  # [batch, seq_len]
                labels = input_ids.clone()  # For causal LM, labels = input_ids
                loss_mask = torch.tensor(batch_loss_masks, device=device)  # [batch, seq_len]

                # 5. Forward pass + GRPO loss with gradient accumulation
                # Split batch into minibatches to avoid OOM (Tinker/Slime pattern)
                model.train()
                optimizer.zero_grad()

                batch_size = input_ids.shape[0]
                num_minibatches = config.trainer.num_minibatches
                micro_batch_size = batch_size // num_minibatches

                total_loss = 0.0
                accumulated_metrics: dict[str, float] = {}
                for i in range(num_minibatches):
                    start_idx = i * micro_batch_size
                    end_idx = start_idx + micro_batch_size

                    # Slice micro-batch
                    mb_input_ids = input_ids[start_idx:end_idx]
                    mb_batch = {
                        "labels": labels[start_idx:end_idx],
                        "loss_mask": loss_mask[start_idx:end_idx],
                        "advantages": advantages[start_idx:end_idx],
                    }

                    # Forward pass
                    outputs = model(input_ids=mb_input_ids)
                    logits = outputs.logits  # [micro_batch, seq_len, vocab_size]

                    # Compute loss for this micro-batch (returns loss, metrics)
                    mb_loss, mb_metrics = grpo_loss(logits, mb_batch)

                    # Scale loss for gradient accumulation (Slime pattern)
                    scaled_loss = mb_loss / num_minibatches
                    scaled_loss.backward()

                    total_loss += mb_loss.item()

                    # Accumulate metrics
                    for k, v in mb_metrics.items():
                        accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v

                # Average loss and metrics for logging
                loss = total_loss / num_minibatches
                for k in accumulated_metrics:
                    accumulated_metrics[k] /= num_minibatches

                # 6. Optimizer step (after all micro-batches)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.max_grad_norm)
                optimizer.step()

                # 7. Log metrics
                reward_std = (
                    (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
                    if rewards
                    else 0.0
                )
                step_metrics = {
                    "step": step + 1,
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "num_samples": len(batch.tokens),
                    **accumulated_metrics,  # Include pg_loss, entropy, avg_logprob, etc.
                }
                metrics_history.append(step_metrics)

                if (step + 1) % config.log_every == 0:
                    # Log with more informative metrics
                    pg_loss = accumulated_metrics.get("pg_loss", loss)
                    entropy = accumulated_metrics.get("entropy", 0.0)
                    logger.info(
                        f"Step {step + 1}: reward={mean_reward:.3f} | "
                        f"pg_loss={pg_loss:.4f} | entropy={entropy:.2f}"
                    )

                # 8. Checkpoint + Weight Sync
                if (step + 1) % config.checkpoint_every == 0:
                    # Save checkpoint in HuggingFace format (SGLang needs this)
                    hf_ckpt_path = output_dir / f"hf_checkpoint_{step + 1}"
                    model.save_pretrained(hf_ckpt_path)
                    tokenizer.save_pretrained(hf_ckpt_path)
                    logger.info(f"Saved HF checkpoint: {hf_ckpt_path}")

                    # Also save optimizer state separately
                    torch.save(
                        {
                            "step": step + 1,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": step_metrics,
                        },
                        output_dir / f"optimizer_{step + 1}.pt",
                    )

                    # Sync weights to inference server
                    logger.info("Syncing weights to SGLang...")
                    try:
                        responses = await sync_weights_to_engines(
                            [inference_engine],
                            str(hf_ckpt_path),
                        )
                        if responses and responses[0].get("success"):
                            logger.info("Weight sync successful")
                        else:
                            logger.warning(f"Weight sync response: {responses}")
                    except Exception as e:
                        logger.warning(f"Weight sync failed: {e}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)
        if metrics_history:
            first_loss = metrics_history[0]["pg_loss"]
            last_loss = metrics_history[-1]["pg_loss"]
            first_reward = metrics_history[0]["mean_reward"]
            last_reward = metrics_history[-1]["mean_reward"]
            logger.info(f"First: pg_loss={first_loss:.4f}, mean_reward={first_reward:.3f}")
            logger.info(f"Last:  pg_loss={last_loss:.4f}, mean_reward={last_reward:.3f}")

        return metrics_history

    finally:
        # Shutdown SGLang server (kill tmux session)
        logger.info("Shutting down SGLang server...")
        subprocess.run(["tmux", "kill-session", "-t", tmux_session], capture_output=True)
        logger.info(f"SGLang logs saved to: {sglang_log}")


def train(config: RLConfig) -> list[dict[str, Any]]:
    """Run RL training with the given config."""
    import torch
    import trio

    if not torch.cuda.is_available():
        print("CUDA not available - skipping training")
        return []

    return trio.run(_train_async, config)


# ──────────────────────── Remote Execution ──────────────────────────────────


def run_remote(
    script_path: str,
    keep_alive: bool = False,
    gpu_id: str | None = None,
    use_tui: bool = False,
    tui_debug: bool = False,
) -> None:
    """Run training script on remote GPU via broker/bifrost.

    Same pattern as examples/sft/base_config.py.

    Args:
        script_path: Path to the training script
        keep_alive: Keep GPU after completion
        gpu_id: Reuse existing GPU instance
        use_tui: Show TUI monitor for logs (sparklines, multi-pane)
        tui_debug: Print raw JSONL instead of TUI (for debugging)
    """
    from dotenv import load_dotenv

    from bifrost.client import BifrostClient
    from broker.client import GPUClient

    load_dotenv()

    # Get credentials
    runpod_key = os.getenv("RUNPOD_API_KEY")
    ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")

    assert runpod_key, "RUNPOD_API_KEY required for remote execution"

    client_gpu = GPUClient(
        credentials={"runpod": runpod_key},
        ssh_key_path=ssh_key_path,
    )

    # Get or provision GPU
    if gpu_id:
        provider, instance_id = gpu_id.split(":", 1)
        instance = client_gpu.get_instance(instance_id, provider)
        assert instance, f"Instance not found: {gpu_id}"
    else:
        print("Provisioning 2x GPU...")
        instance = client_gpu.create(
            client_gpu.gpu_type.contains("A100") | client_gpu.gpu_type.contains("4090"),
            gpu_count=2,
            cloud_type="secure",
            container_disk_gb=100,
            sort=lambda x: x.price_per_hour,
        )
        print(f"Instance: {instance.provider}:{instance.id}")

    print("Waiting for SSH...")
    instance.wait_until_ssh_ready(timeout=600)

    key_path = client_gpu.get_ssh_key_path(instance.provider)
    bifrost = BifrostClient(instance.ssh_connection_string(), ssh_key_path=key_path)

    try:
        # Deploy code with bootstrap
        print("Deploying code...")
        bootstrap = [
            "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
            "uv pip install torch transformers datasets accelerate sglang[all] curl_cffi",
        ]
        workspace = bifrost.push("~/.bifrost/workspaces/rollouts-rl", bootstrap_cmd=bootstrap)
        print("Code deployed")

        # Run training (PYTHONUNBUFFERED=1 for real-time output)
        print("Running training...")
        script_name = Path(script_path).name
        env_vars = "PYTHONUNBUFFERED=1"
        if use_tui or tui_debug:
            env_vars += " ROLLOUTS_JSON_LOGS=true"
        cmd = f"cd {workspace}/rollouts && {env_vars} uv run python examples/rl/calculator/{script_name}"
        print(f"Running: {cmd}")

        if tui_debug:
            # Just print raw JSONL for debugging
            print("-" * 50)
            for line in bifrost.exec_stream(cmd):
                print(line, flush=True)
            print("-" * 50)
        elif use_tui:
            # Route output through TUI monitor
            from rollouts.tui.monitor import TrainingMonitor

            monitor = TrainingMonitor()

            def feed_monitor():
                """Feed lines to monitor in background."""
                import threading
                import time

                lines_queue = []
                done = threading.Event()

                def collect_lines():
                    try:
                        for line in bifrost.exec_stream(cmd):
                            lines_queue.append(line)
                    finally:
                        done.set()

                collector = threading.Thread(target=collect_lines, daemon=True)
                collector.start()

                # Run TUI with line feeding
                monitor._running = True
                monitor.terminal.start(on_input=lambda x: None, on_resize=monitor._on_resize)

                try:
                    while monitor._running and not done.is_set():
                        # Feed queued lines
                        while lines_queue:
                            raw_line = lines_queue.pop(0)
                            log_line = monitor.parse_jsonl_line(raw_line)
                            if log_line:
                                pane_name = monitor.route_log_line(log_line)
                                monitor.panes[pane_name].add_line(log_line)
                                monitor._needs_redraw = True

                        # Handle keyboard input
                        data = monitor.terminal.read_input()
                        if data:
                            monitor._handle_input(data)

                        # Render if needed
                        if monitor._needs_redraw:
                            monitor._render()
                            monitor._needs_redraw = False

                        time.sleep(0.05)
                finally:
                    monitor.terminal.stop()

            feed_monitor()
        else:
            print("-" * 50)
            for line in bifrost.exec_stream(cmd):
                print(line, flush=True)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Syncing logs before exit...")

    finally:
        # Always sync results/logs (even on ctrl+c)
        print("\nSyncing results...")
        local_results = Path("results/rl")
        local_results.mkdir(parents=True, exist_ok=True)

        # Sync only logs and config (NOT checkpoints/optimizer state - those are huge)
        remote_output_dir = "/tmp/rollouts_rl/calculator_grpo"
        files_to_sync = ["sglang_server.log", "config.json", "metrics.json", "training.log"]

        for filename in files_to_sync:
            try:
                result = bifrost.download_files(
                    remote_path=f"{remote_output_dir}/{filename}",
                    local_path=str(local_results / filename),
                    recursive=False,
                )
                if result and result.success:
                    print(f"  Synced: {filename}")
            except Exception:
                # File might not exist yet, that's ok
                pass

        if not keep_alive:
            print(f"\nTerminating instance {instance.provider}:{instance.id}...")
            instance.terminate()
        else:
            print(f"\nInstance kept alive: {instance.provider}:{instance.id}")
            print(f"Reuse with: --gpu-id {instance.provider}:{instance.id}")
