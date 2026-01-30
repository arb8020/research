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
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)

# Default: Use Prime's pre-trained SFT model (recommended)
# This model already knows how to reverse text, so RL can refine it
DEFAULT_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

# Alternative: Base model (will struggle without SFT warmup)
BASE_MODEL = "Qwen/Qwen3-0.6B"

# Prime-RL uses:
#   - max_steps=20 (prime-rl) or 100 (verifiers)
#   - max_tokens=128
#   - batch_size=128 total (8 prompts × 16 rollouts)
#   - lr=3e-6
#   - seq_len=512-2048
#   - sync_weights_every=1 (on-policy, max_async_level=1)
config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_grpo_01"),
    model=ModelConfig(name=DEFAULT_MODEL),
    checkpoint=CheckpointConfig(
        num_steps=20,  # prime-rl uses 20
        checkpoint_every=5,
        sync_weights_every=1,  # on-policy
    ),
    rollout=RolloutConfig(
        batch_size=8,  # prompts per step (× 16 rollouts = 128 total)
        n_samples_per_prompt=16,
        temperature=1.0,
        max_seq_len=1024,  # 2048 OOMs on single GPU; 1024 fits with num_minibatches=8
        max_tokens=512,  # enough for think + answer; 128 truncated, 2048 OOMed
        extra_params={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ),
    trainer=TrainerConfig(
        lr=3e-6,
        loss_type="masked",  # Prime-RL importance ratio masking
    ),
)

# For base model variant, create a separate config file or use:
# python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py

if __name__ == "__main__":
    train(config)
