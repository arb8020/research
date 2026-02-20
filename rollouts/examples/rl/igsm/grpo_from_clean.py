"""GRPO training on iGSM from a clean-pretrained model.

This is the key experiment: can RL learn error-correction behavior
that LoRA couldn't instill?

Hypothesis:
- Paper showed: pretrain_clean → LoRA_retry → FAILS
- Paper showed: pretrain_retry → WORKS
- Question: pretrain_clean → GRPO (outcome reward only) → ???

If GRPO can learn self-correction from outcome reward alone,
it would suggest RL can discover behaviors that supervised
finetuning (even full finetune, definitely LoRA) cannot.

Usage:
    # From pretrained checkpoint
    python grpo_from_clean.py --checkpoint output/igsm_clean/step_100000.pt

    # Quick test with base GPT2
    python grpo_from_clean.py --tiny

    # Remote GPU
    python grpo_from_clean.py --checkpoint ... --provision --provider runpod
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

# Support running as script
sys.path.insert(0, str(Path(__file__).parent))

from base_config import igsm_score_fn, load_igsm_prompts

from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.grpo import (
    GRPOConfig,
    GRPOOutputConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
    grpo_train,
)


def get_config(
    checkpoint_path: str | Path | None = None,
    tiny: bool = False,
    difficulty: Literal["easy", "med", "hard"] = "med",
) -> GRPOConfig:
    """Get GRPO config for iGSM.

    Args:
        checkpoint_path: Path to pretrained checkpoint
        tiny: Use tiny config for testing
        difficulty: Problem difficulty

    Returns:
        GRPOConfig
    """
    if tiny:
        return GRPOConfig(
            model=ModelConfig(
                name="gpt2",  # Base GPT2 for quick testing
            ),
            output=GRPOOutputConfig(
                experiment_name=f"igsm_{difficulty}_grpo_tiny",
                output_dir="output",
            ),
            trainer=TrainerConfig(
                lr=1e-5,
                loss_type="vanilla",
            ),
            rollout=RolloutConfig(
                n_samples_per_prompt=4,
                temperature=0.8,
                max_seq_len=512,
                batch_size=2,
            ),
        )
    else:
        # Full config for real experiments
        model_name = str(checkpoint_path) if checkpoint_path else "gpt2"
        return GRPOConfig(
            model=ModelConfig(
                name=model_name,
                dtype="bfloat16",
            ),
            output=GRPOOutputConfig(
                experiment_name=f"igsm_{difficulty}_grpo",
                output_dir="output",
            ),
            trainer=TrainerConfig(
                lr=1e-6,  # Low LR for RL on pretrained model
                loss_type="clipped",  # PPO-style clipping
                clip_ratio=0.2,
            ),
            rollout=RolloutConfig(
                n_samples_per_prompt=8,  # Paper used 8 for GSM8K-style
                temperature=0.8,
                max_seq_len=1024,  # Allow longer for reasoning
                batch_size=8,
            ),
        )


def train(
    checkpoint_path: str | Path | None = None,
    max_samples: int | None = None,
    difficulty: Literal["easy", "med", "hard"] = "med",
    tiny: bool = False,
) -> dict[str, Any]:
    """Run GRPO training on iGSM.

    Args:
        checkpoint_path: Path to pretrained checkpoint
        max_samples: Limit dataset size
        difficulty: Problem difficulty
        tiny: Use tiny config for testing

    Returns:
        Dict with metrics_history
    """
    config = get_config(
        checkpoint_path=checkpoint_path,
        tiny=tiny,
        difficulty=difficulty,
    )

    # Load prompts (different seed than pretrain for held-out problems)
    prompts = load_igsm_prompts(
        max_samples=max_samples or (50 if tiny else 1000),
        difficulty=difficulty,
        seed=12345,  # Different from pretrain seed
        mode="train",
    )

    print(f"Loaded {len(prompts)} iGSM prompts")
    print(f"Config: {config}")

    return grpo_train(
        config=config,
        prompts=prompts,
        score_fn=igsm_score_fn,
        environment_cls=BasicEnvironment,
    )


def main():
    parser = argparse.ArgumentParser(description="GRPO on iGSM from clean pretrain")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to pretrained checkpoint"
    )
    parser.add_argument("--tiny", action="store_true", help="Use tiny config for testing")
    parser.add_argument(
        "--difficulty", choices=["easy", "med", "hard"], default="med", help="Problem difficulty"
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size")
    # Remote execution args (handled by rollouts.run)
    parser.add_argument("--provision", action="store_true", help="Provision remote GPU")
    parser.add_argument("--provider", type=str, default="runpod", help="GPU provider")
    parser.add_argument("--keep-alive", action="store_true", help="Keep pod alive after completion")
    args = parser.parse_args()

    if args.provision:
        # Remote execution - delegate to rollouts.run
        print("Remote execution not yet wired up for this script")
        print("Use: python -m rollouts.run --config examples/rl/igsm/grpo_from_clean.py ...")
        return

    train(
        checkpoint_path=args.checkpoint,
        max_samples=args.max_samples,
        difficulty=args.difficulty,
        tiny=args.tiny,
    )


if __name__ == "__main__":
    main()
