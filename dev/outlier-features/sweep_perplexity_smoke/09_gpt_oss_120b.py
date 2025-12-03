"""Perplexity smoke test - GPT-OSS-120B.

Quick perplexity computation to validate pipeline on largest model.
Runtime: ~5-10 minutes on 2xGPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: GPT-OSS-120B (117B total, 5.1B active)
config.model.name = "openai/gpt-oss-120b"
config.model.device_map = "balanced"

# Dataset: Small number of sequences for smoke test
config.dataset.num_sequences = 10  # Just 10 sequences for quick test
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Use perplexity batch size
config.analysis.batch_size = 2  # Process 2 at a time

# Deployment: Multi-GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.5
config.deployment.safety_factor = 1.3
config.deployment.container_disk = 250  # Large models need more disk space
config.deployment.volume_disk = 0  # No volume to avoid mount issues

# Output
config.output.experiment_name = "perplexity_smoke_gpt_oss_120b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
