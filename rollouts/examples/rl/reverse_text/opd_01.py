"""Reverse Text On-Policy Distillation (OPD).

Demonstrates On-Policy Distillation where a student model learns to match
a teacher model's token-level predictions on student-generated rollouts.

OPD provides dense feedback (O(N) bits per episode) compared to sparse
RL rewards (O(1) bit per episode), resulting in 9-30x compute savings.

Architecture:
- Student: Qwen3-0.6B with LoRA (trained)
- Teacher: Qwen3-8B (frozen, provides dense signal)

Both run as SGLang servers on different ports. After student generates
a rollout, we query the teacher for log probs on those same tokens,
then use (teacher_logprob - student_logprob) as per-token advantages.

Run:
    # Local (requires 2 GPUs - one for student, one for teacher)
    python rollouts/run.py --config examples/rl/reverse_text/opd_01.py --local

    # Remote (RunPod 2xA100)
    python rollouts/run.py --config examples/rl/reverse_text/opd_01.py --provision --provider runpod

References:
    - https://thinkingmachines.ai/blog/on-policy-distillation/
    - /tmp/miles/examples/on_policy_distillation/
"""

from examples.rl.reverse_text.base_config import train  # noqa: F401
from rollouts.training.configs import HardwareConfig
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)

# =============================================================================
# Hardware Configuration
# =============================================================================

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=2,  # One for student, one for teacher
    provider="runpod",
)

# =============================================================================
# Training Configuration
# =============================================================================

# Student model (trained with LoRA)
STUDENT_MODEL = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"

# Teacher model (frozen, provides dense signal)
# Use a larger model that knows the task well
TEACHER_MODEL = "Qwen/Qwen3-8B"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_opd_01"),
    model=ModelConfig(
        name=STUDENT_MODEL,
        # LoRA for efficient training
        use_lora=True,
        lora_rank=16,
        lora_alpha=32,
    ),
    checkpoint=CheckpointConfig(
        num_steps=20,
        checkpoint_every=10,
        sync_weights_every=1,
    ),
    rollout=RolloutConfig(
        batch_size=8,
        # Fewer samples per prompt needed with dense signal
        n_samples_per_prompt=4,
        temperature=1.0,
        max_seq_len=2048,
        max_tokens=128,
    ),
    trainer=TrainerConfig(
        lr=1e-4,  # Higher LR for LoRA
        num_minibatches=16,
        # Use OPD advantage estimator instead of GRPO
        advantage_estimator="opd",
        teacher_model=TEACHER_MODEL,
        teacher_port=30100,  # Separate port for teacher server
    ),
    inference=InferenceConfig(
        port=30000,  # Student inference port
        cuda_device_ids=(0,),  # Student on GPU 0
        mem_fraction=0.5,
    ),
)


if __name__ == "__main__":
    import sys

    from rollouts.run import main

    sys.argv = [sys.argv[0], "--config", __file__] + sys.argv[1:]
    main()
