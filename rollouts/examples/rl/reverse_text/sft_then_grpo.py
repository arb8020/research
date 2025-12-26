#!/usr/bin/env python3
"""SFT → GRPO Pipeline for Reverse Text.

This demonstrates the proper way to train RL on a new skill:
1. SFT warmup: Teach the model the basic skill (text reversal)
2. GRPO refinement: Improve quality via RL on shorter examples

Based on Prime-RL's reverse-text example which achieves:
- Base model: ~5% reward (can't reverse text)
- After SFT: ~50% reward (knows the skill)
- After RL: ~80% reward (refined and robust)

Usage:
    # Full pipeline (SFT + RL)
    python examples/rl/reverse_text/sft_then_grpo.py

    # Skip SFT, use existing checkpoint
    python examples/rl/reverse_text/sft_then_grpo.py --skip-sft --sft-checkpoint /path/to/sft

    # Use Prime's pre-trained SFT model
    python examples/rl/reverse_text/sft_then_grpo.py --use-prime-sft

    # Remote execution
    python examples/rl/reverse_text/sft_then_grpo.py --provision
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Configuration for SFT → RL pipeline."""

    # Model
    base_model: str = "Qwen/Qwen3-0.6B"

    # SFT settings (following Prime-RL)
    sft_dataset: str = "willcb/R1-reverse-wikipedia-paragraphs-v1-1000"
    sft_num_steps: int = 100
    sft_batch_size: int = 32
    sft_lr: float = 2e-5  # Higher LR for SFT
    sft_max_seq_len: int = 4096

    # RL settings (following Prime-RL)
    rl_dataset: str = "PrimeIntellect/Reverse-Text-RL"
    rl_num_steps: int = 100  # verifiers uses 100, prime-rl uses 20
    rl_batch_size: int = 8
    rl_n_samples_per_prompt: int = 16
    rl_lr: float = 3e-6  # Lower LR for RL
    rl_max_tokens: int = 128  # Prime uses 128
    rl_temperature: float = 1.0  # Prime uses default 1.0

    # Output
    output_dir: str = "/tmp/reverse_text_pipeline"


def run_sft(config: PipelineConfig) -> Path:
    """Run SFT warmup training.

    Returns:
        Path to HuggingFace-format checkpoint
    """
    import logging

    import torch
    import trio
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rollouts._logging import setup_logging
    from rollouts.training import PyTorchTrainingBackend, SFTTrainingConfig, load_sft_dataset

    setup_logging(level="INFO", use_color=True)
    logger = logging.getLogger(__name__)

    output_dir = Path(config.output_dir) / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Stage 1: SFT Warmup")
    logger.info("=" * 60)
    logger.info(f"Model: {config.base_model}")
    logger.info(f"Dataset: {config.sft_dataset}")
    logger.info(f"Steps: {config.sft_num_steps}")
    logger.info(f"LR: {config.sft_lr}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    # Load dataset
    samples = load_sft_dataset(
        config.sft_dataset,
        tokenizer=tokenizer,
        max_length=config.sft_max_seq_len,
    )
    logger.info(f"Loaded {len(samples)} samples")

    # Setup backend
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.sft_lr)

    def sft_loss_fn(logits: torch.Tensor, batch: dict) -> torch.Tensor:
        import torch.nn.functional as F

        labels = batch["labels"]
        loss_mask = batch["loss_mask"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = loss_mask[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        )
        masked_loss = per_token_loss * shift_mask.view(-1)
        return masked_loss.sum() / (shift_mask.sum() + 1e-8)

    backend = PyTorchTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=sft_loss_fn,
        checkpoint_dir=output_dir / "checkpoints",
        device=torch.device("cuda:0"),
    )

    # Training config
    training_config = SFTTrainingConfig(
        num_steps=config.sft_num_steps,
        batch_size=config.sft_batch_size,
        log_every=10,
        checkpoint_every=config.sft_num_steps,  # Save at end
    )

    # Run training
    async def train() -> Path:
        from rollouts.training.loops import run_sft_training

        await run_sft_training(backend, samples, training_config)
        # Save in HF format for GRPO
        hf_path = output_dir / "hf_model"
        await backend.save_hf_checkpoint(hf_path, tokenizer)
        return hf_path

    hf_checkpoint_path = trio.run(train)
    logger.info(f"SFT complete! Checkpoint: {hf_checkpoint_path}")
    return hf_checkpoint_path


def run_grpo(config: PipelineConfig, sft_checkpoint: Path | str) -> dict[str, Any]:
    """Run GRPO refinement training.

    Args:
        config: Pipeline configuration
        sft_checkpoint: Path to SFT checkpoint (HF format)

    Returns:
        Training metrics
    """
    import logging
    from difflib import SequenceMatcher

    from datasets import load_dataset

    from rollouts._logging import setup_logging
    from rollouts.dtypes import Metric, Score
    from rollouts.environments.no_tools import BasicEnvironment
    from rollouts.training.grpo import GRPOConfig, grpo_train

    setup_logging(level="INFO", use_color=True)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Stage 2: GRPO Refinement")
    logger.info("=" * 60)
    logger.info(f"Model: {sft_checkpoint}")
    logger.info(f"Dataset: {config.rl_dataset}")
    logger.info(f"Steps: {config.rl_num_steps}")
    logger.info(f"LR: {config.rl_lr}")

    # Load RL dataset
    dataset = load_dataset(config.rl_dataset, split="train")

    # Convert to prompts format
    system_prompt = (
        "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
    )
    prompts = []
    for row in dataset:
        text = row["prompt"]
        reversed_text = text[::-1]
        prompts.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "reversed": reversed_text,
        })

    logger.info(f"Loaded {len(prompts)} prompts")

    # Score function (LCS similarity)
    def score_fn(sample: Any) -> Score:
        expected = sample.metadata.get("reversed", "")
        response = sample.response if hasattr(sample, "response") else ""

        # Parse <reversed_text> tags if present
        if "<reversed_text>" in response:
            start = response.find("<reversed_text>") + len("<reversed_text>")
            end = response.find("</reversed_text>")
            if end > start:
                response = response[start:end]

        response = response.strip().strip("\"'")
        similarity = SequenceMatcher(None, response, expected).ratio()

        return Score(
            metrics=(
                Metric("similarity", similarity, weight=1.0),
                Metric("exact_match", 1.0 if response == expected else 0.0, weight=0.0),
            )
        )

    # GRPO config
    grpo_config = GRPOConfig(
        experiment_name="reverse_text_sft_grpo",
        model_name=str(sft_checkpoint),
        num_steps=config.rl_num_steps,
        batch_size=config.rl_batch_size,
        n_samples_per_prompt=config.rl_n_samples_per_prompt,
        lr=config.rl_lr,
        temperature=config.rl_temperature,
        max_tokens=config.rl_max_tokens,
        max_seq_len=512,
        output_dir=str(Path(config.output_dir) / "grpo"),
    )

    # Run GRPO
    results = grpo_train(
        config=grpo_config,
        prompts=prompts,
        score_fn=score_fn,
        environment_cls=BasicEnvironment,
    )

    logger.info("GRPO complete!")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT → GRPO Pipeline for Reverse Text")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT, use existing checkpoint")
    parser.add_argument("--sft-checkpoint", type=str, help="Path to existing SFT checkpoint")
    parser.add_argument(
        "--use-prime-sft",
        action="store_true",
        help="Use Prime's pre-trained SFT model (PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT)",
    )
    parser.add_argument("--provision", action="store_true", help="Provision new GPU instance")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--node-id", type=str, help="Reuse existing instance ID")
    args = parser.parse_args()

    # Remote execution
    if args.provision or args.node_id:
        from examples.rl.base_config import run_remote

        run_remote(
            __file__,
            keep_alive=args.keep_alive,
            node_id=args.node_id,
        )
        return

    # Local execution
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    config = PipelineConfig()

    # Determine SFT checkpoint
    if args.use_prime_sft:
        sft_checkpoint = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
        print(f"Using Prime's pre-trained SFT model: {sft_checkpoint}")
    elif args.skip_sft:
        if not args.sft_checkpoint:
            print("Error: --skip-sft requires --sft-checkpoint")
            sys.exit(1)
        sft_checkpoint = args.sft_checkpoint
        print(f"Using existing SFT checkpoint: {sft_checkpoint}")
    else:
        # Run SFT
        sft_checkpoint = run_sft(config)

    # Run GRPO
    results = run_grpo(config, sft_checkpoint)

    # Summary
    if results.get("metrics_history"):
        history = results["metrics_history"]
        first_reward = history[0].get("mean_reward", 0)
        last_reward = history[-1].get("mean_reward", 0)
        print()
        print("=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"First step reward: {first_reward:.3f}")
        print(f"Last step reward:  {last_reward:.3f}")
        print(f"Improvement: {(last_reward - first_reward) / max(first_reward, 0.01) * 100:.1f}%")


if __name__ == "__main__":
    main()
