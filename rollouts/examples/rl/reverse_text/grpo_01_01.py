"""Reverse Text GRPO baseline experiment.

Run with:
    # Local (requires GPU + SGLang) - uses Prime's SFT model by default
    python examples/rl/reverse_text/grpo_01_01.py

    # Use base model (will likely fail - no SFT warmup!)
    python examples/rl/reverse_text/grpo_01_01.py --base-model

    # Remote (provisions GPU automatically)
    python examples/rl/reverse_text/grpo_01_01.py --provision

    # Reuse existing GPU
    python examples/rl/reverse_text/grpo_01_01.py --node-id runpod:abc123

Note:
    Using the base Qwen3-0.6B model without SFT warmup typically achieves
    only ~5% reward because the model doesn't know how to reverse text.

    Prime's SFT model (PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT) was
    trained on willcb/R1-reverse-wikipedia-paragraphs-v1-1000 and starts
    at ~50% reward, which RL can then improve to ~80%.

    For the full SFT → RL pipeline, see sft_then_grpo.py
"""

from examples.rl.reverse_text.base_config import train  # noqa: F401 (used by runner)
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)

# Default: Use Prime's pre-trained SFT model (recommended)
# This model already knows how to reverse text, so RL can refine it
DEFAULT_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

# Alternative: Base model (will struggle without SFT warmup)
BASE_MODEL = "Qwen/Qwen3-0.6B"

# Matches prime-rl nightly CI: examples/reverse_text/rl.toml
# - batch_size=128, rollouts_per_example=16, max_tokens=128
# - seq_len=2048 (A100), seq_len=512 (24GB GPUs like A5000)
# - max_steps=20, lr=3e-6
# - Tested nightly: reward must reach >= 0.65
config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_grpo_01"),
    model=ModelConfig(name=DEFAULT_MODEL),
    checkpoint=CheckpointConfig(
        num_steps=20,
        checkpoint_every=5,
        sync_weights_every=1,  # on-policy
    ),
    rollout=RolloutConfig(
        batch_size=8,  # prompts per step (× 16 rollouts = 128 total)
        n_samples_per_prompt=16,
        temperature=1.0,
        max_seq_len=512,  # Reduced from 2048 for 24GB GPUs (reverse_text needs <256)
        max_tokens=128,
        extra_params={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ),
    trainer=TrainerConfig(
        lr=3e-6,
        num_minibatches=32,  # 128 total / 32 = micro_batch_size=4 (fits 24GB GPU)
        loss_type="masked",  # Prime-RL importance ratio masking
    ),
    inference=InferenceConfig(
        mem_fraction=0.5,  # Reduced from 0.7 to leave more room for training on 24GB
    ),
)

# For base model variant, create a separate config file or use:
# python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py

if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
