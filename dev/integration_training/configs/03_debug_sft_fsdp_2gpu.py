"""Debug config for SFT with FSDP (2 GPUs).

Minimal FSDP test on just 2 GPUs to validate distributed training.
If this works, 4/8/16 GPUs should work the same way.

Expected runtime: ~15 minutes on 2 GPUs.

Tiger Style: ALL parameters explicit, no defaults.
"""

import sys
from pathlib import Path

# Add parent to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_config import (
    Config, DatasetSpec,
    TargetConfig, ModelConfig, DataConfig, SFTConfig, RLConfig, OutputConfig,
)

# Tiger Style: Construct config with ALL parameters explicit
config = Config(
    target=TargetConfig(
        # Training: 2 GPUs with FSDP (minimal test)
        gpu_ranks=[1, 2],
        device_type="cuda",

        # Backend: FSDP for multi-GPU training
        train_backend="fsdp",

        # Distributed training parameters
        master_addr="localhost",
        master_port=29500,
    ),
    model=ModelConfig(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="bfloat16",
        compile=False,
    ),
    data=DataConfig(
        # SFT data: Just SmolTalk for quick test
        sft_mixture=[
            DatasetSpec(
                name="HuggingFaceTB/smol-smoltalk",
                split="train",
                subset=None,
                max_samples=1000,  # 1K samples
                filepath=None,
                size=None,
                repeat=1,
            ),
        ],
        # RL data: Not used for SFT-only mode
        rl_mixture=[],
        max_length=512,
        shuffle_seed=42,
    ),
    sft=SFTConfig(
        # Training schedule
        num_epochs=1,
        num_iterations=100,  # Just 100 steps for debug
        batch_size=2,  # Per-GPU batch size
        target_examples_per_step=8,  # Total across 2 GPUs

        # Learning rates (nanochat defaults)
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        init_lr_frac=0.02,

        # Logging
        eval_every=25,
        eval_steps=10,
        checkpoint_every=50,
        log_every=10,
    ),
    rl=RLConfig(
        # Not used for SFT-only mode, but required by Config
        num_epochs=1,
        examples_per_step=8,
        num_samples=8,
        batch_size=4,
        max_new_tokens=128,
        temperature=1.0,
        top_k=50,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        init_lr_frac=0.05,
        eval_every=20,
        eval_examples=100,
        save_every=20,
        baseline=0.0,
    ),
    output=OutputConfig(
        save_dir=Path("./results"),
        log_level="INFO",
        experiment_name="03_debug_sft_fsdp_2gpu",
        use_wandb=False,
        wandb_project="integration_training",
        mode="sft",  # SFT only
        source_checkpoint=None,
    ),
)
