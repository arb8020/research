"""Training-architecture smoke test 2: NMOE backend + MoE model (10 steps).

Runs GRPO for a few steps on a tiny synthetic reverse-text prompt set.

Run (from `rollouts/`):
  `python -m rollouts.run --config examples/training_architecture/test2_nmoe_moe_glm_4_7_flash.py`
  `python -m rollouts.run --config examples/training_architecture/test2_nmoe_moe_glm_4_7_flash.py --provider modal`
"""

from examples.training_architecture.shared import train  # noqa: F401 (used by runner)
from rollouts.training.configs import DepsConfig, HardwareConfig
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=2,  # 1x inference + 1x trainer
    provider="runpod",
    deps=DepsConfig(
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
        pip_packages=(
            "torch>=2.4",
            "transformers>=5.0",
            "accelerate",
            "safetensors",
            "curl_cffi",
            "peft",
            "huggingface_hub>=1.4.0",
            "httpx",
            "trio",
            "requests",
            "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python",
        ),
    ),
)


config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="ta_test2_nmoe_moe_glm47flash"),
    model=ModelConfig(
        name="zai-org/GLM-4.7-Flash",
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="nmoe",
        cuda_device_ids=(1,),
        lr=1e-6,
        weight_decay=0.0,
        num_minibatches=4,
        max_grad_norm=1.0,
    ),
    inference=InferenceConfig(
        backend="sglang",
        cuda_device_ids=(0,),
        port=30000,
        mem_fraction=0.7,
        tensor_parallel_size=1,
    ),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=2,
        max_seq_len=256,
        max_tokens=128,
        temperature=0.8,
    ),
    checkpoint=CheckpointConfig(
        num_steps=10,
        log_every=1,
        checkpoint_every=10,
        # Keep weight sync off for this backend smoke test.
        # Weight sync is exercised in the test3_* configs.
        sync_weights_every=1000,
        weight_sync_mode="disk",
        pipeline_mode="sync",
    ),
)
