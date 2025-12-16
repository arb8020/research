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
import random
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

    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
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
    from rollouts.training.rl_losses import grpo_loss
    from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
    from rollouts.training.types import RolloutConfig
    from rollouts.training.weight_sync import SGLangEngine, sync_weights_to_engines

    # Setup logging
    setup_logging(level="INFO", use_color=True)
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
    # 1. Launch inference server (SGLang)
    # ─────────────────────────────────────────────────────────────────────────
    inference_gpu_ids = config.inference.gpu_ids
    inference_port = config.inference.port

    logger.info(f"Launching SGLang on GPUs {inference_gpu_ids}, port {inference_port}...")

    sglang_cmd = [
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
    ]

    # Multi-GPU: add tensor parallel size if more than 1 GPU
    if len(inference_gpu_ids) > 1:
        sglang_cmd.extend(["--tp-size", str(len(inference_gpu_ids))])

    sglang_env = os.environ.copy()
    sglang_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in inference_gpu_ids)

    sglang_process = subprocess.Popen(
        sglang_cmd,
        env=sglang_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    health_url = f"http://localhost:{inference_port}/health"
    server_ready = False
    for attempt in range(120):  # 2 minutes timeout
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
        sglang_process.terminate()
        raise RuntimeError("SGLang server failed to start within 2 minutes")

    logger.info(f"SGLang ready at http://localhost:{inference_port}")

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
            from rollouts.training.types import Sample as TrainingSample

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
                    logger.warning(f"Rollout failed for prompt '{prompt[:50]}...': {e}")

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

                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                logger.info(f"Mean reward: {mean_reward:.3f}")

                # 3. Compute advantages (GRPO: reward - mean)
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

                # 5. Forward pass + GRPO loss
                model.train()
                optimizer.zero_grad()

                outputs = model(input_ids=input_ids)
                logits = outputs.logits  # [batch, seq_len, vocab_size]

                loss = grpo_loss(logits, labels, loss_mask, advantages)

                # 6. Backward pass + optimizer step
                loss.backward()
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
                    "loss": loss.item(),
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "num_samples": len(batch.tokens),
                    "mean_advantage": advantages.mean().item(),
                }
                metrics_history.append(step_metrics)

                if (step + 1) % config.log_every == 0:
                    logger.info(
                        f"Step {step + 1}: loss={loss.item():.4f}, mean_reward={mean_reward:.3f}"
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
            first_loss = metrics_history[0]["loss"]
            last_loss = metrics_history[-1]["loss"]
            first_reward = metrics_history[0]["mean_reward"]
            last_reward = metrics_history[-1]["mean_reward"]
            logger.info(f"First: loss={first_loss:.4f}, mean_reward={first_reward:.3f}")
            logger.info(f"Last:  loss={last_loss:.4f}, mean_reward={last_reward:.3f}")

        return metrics_history

    finally:
        # Shutdown SGLang server
        logger.info("Shutting down SGLang server...")
        sglang_process.terminate()
        sglang_process.wait(timeout=10)


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
) -> None:
    """Run training script on remote GPU via broker/bifrost.

    Same pattern as examples/sft/base_config.py.
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
            "uv pip install torch 'transformers<4.52' datasets accelerate sglang[all] curl_cffi",
        ]
        workspace = bifrost.push("~/.bifrost/workspaces/rollouts-rl", bootstrap_cmd=bootstrap)
        print("Code deployed")

        # Run training
        print("Running training...")
        script_name = Path(script_path).name
        cmd = f"cd {workspace}/rollouts && uv run python examples/rl/calculator/{script_name}"
        print(f"Running: {cmd}")
        print("-" * 50)
        for line in bifrost.exec_stream(cmd):
            print(line, end="")
        print("-" * 50)

    finally:
        if not keep_alive:
            print(f"\nTerminating instance {instance.provider}:{instance.id}...")
            instance.terminate()
        else:
            print(f"\nInstance kept alive: {instance.provider}:{instance.id}")
            print(f"Reuse with: --gpu-id {instance.provider}:{instance.id}")
