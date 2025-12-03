"""Perplexity computation - GPT-OSS-20B.

First systematic MoE model in original analysis.
Runtime: ~15-20 minutes on 2xA100.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: GPT-OSS-20B (21.5B total, ? active)
config.model.name = "openai/gpt-oss-20b"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Perplexity: Standard batch size
config.perplexity.batch_size = 1
config.perplexity.stride = None  # No overlapping windows

# Deployment: Multi-GPU
config.deployment.min_vram = 40  # Explicit: 2x40GB = 80GB total
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.0
config.deployment.safety_factor = 1.3
config.deployment.container_disk = 150  # Standard for 20B model
config.deployment.volume_disk = 0  # No volume to avoid mount issues

# Output
config.output.experiment_name = "perplexity_gpt_oss_20b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
