"""Perplexity smoke test - GPT-OSS-20B.

Quick perplexity computation to validate pipeline on large MoE model.
Runtime: ~5-10 minutes on 2xGPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: GPT-OSS-20B (21.5B total)
config.model.name = "openai/gpt-oss-20b"
config.model.device_map = "balanced"

# Dataset: Small number of sequences for smoke test
config.dataset.num_sequences = 10  # Just 10 sequences for quick test
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Use perplexity batch size
config.analysis.batch_size = 2  # Process 2 at a time

# Deployment: Multi-GPU
config.deployment.min_vram = 40  # Explicit: 2x40GB
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.0
config.deployment.safety_factor = 1.3
config.deployment.container_disk = 150
config.deployment.volume_disk = 0

# Output
config.output.experiment_name = "perplexity_smoke_gpt_oss_20b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
