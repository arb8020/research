"""Dense SFT witness path using TorchTitan.

Run locally on a machine with access to the Llama 3.1 weights and a CUDA GPU.
This is the canonical dense supervised TorchTitan instantiation path.
"""

from examples.sft.base_config import SFTConfig
from examples.sft.base_config import train as _base_train
from rollouts.training.configs import CheckpointConfig, ModelConfig, OutputConfig, TrainerConfig

config = SFTConfig(
    model=ModelConfig(
        name="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="torchtitan",
        torchtitan_model="llama3",
        torchtitan_model_size="8B",
        lr=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        torchtitan_tp=1,
        torchtitan_cp=1,
        torchtitan_pp=1,
    ),
    checkpoint=CheckpointConfig(
        num_steps=100,
        log_every=10,
        checkpoint_every=50,
    ),
    output=OutputConfig(
        output_dir="/tmp/rollouts_sft_torchtitan",
        experiment_name="llama3_torchtitan_dense_sft",
    ),
    batch_size=4,
    device="cuda:0",
)


def train(config: SFTConfig | None = None) -> list[dict]:
    return _base_train(config or globals()["config"])
