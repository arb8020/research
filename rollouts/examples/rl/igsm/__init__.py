"""iGSM experiments for Physics of Language Models Part 2.2.

Testing whether RL can learn error-correction behavior where LoRA fails.

Experiment structure:
- pretrain_clean.py: Pretrain on clean iGSM data (baseline)
- pretrain_retry.py: Pretrain with retry data (paper's winning condition)
- grpo_from_clean.py: RL on clean-pretrained model (our hypothesis)
- eval_accuracy.py: Evaluate accuracy and retry usage

Key question:
  Paper showed LoRA can't learn retry behavior from clean pretrain.
  Can GRPO (outcome reward only) discover it?

Usage:
    # Quick test (tiny model, CPU)
    python examples/rl/igsm/pretrain_clean.py --tiny

    # Full pretrain (GPU)
    python examples/rl/igsm/pretrain_clean.py

    # RL from checkpoint
    python examples/rl/igsm/grpo_from_clean.py --checkpoint output/igsm_clean/step_100000.pt
"""

__all__ = [
    "load_igsm_prompts",
    "igsm_score_fn",
    "extract_answer",
    "normalize_answer",
    "RETRY_TOKEN",
]
