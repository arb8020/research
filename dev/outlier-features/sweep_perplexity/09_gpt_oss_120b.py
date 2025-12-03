"""Perplexity computation - GPT-OSS-120B.

Largest model - validates GPT-OSS systematicity at scale.
Runtime: ~35-50 minutes on 2xA100.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: GPT-OSS-120B (117B total, 5.1B active)
config.model.name = "openai/gpt-oss-120b"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Perplexity: Standard batch size
config.perplexity.batch_size = 1
config.perplexity.stride = None  # No overlapping windows

# Deployment: Multi-GPU with high VRAM
config.deployment.min_vram = 80  # Explicit: 2x80GB = 160GB total
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 5.0
config.deployment.safety_factor = 1.4
config.deployment.container_disk = 250  # Large model needs more disk
config.deployment.volume_disk = 0  # No volume to avoid mount issues

# Output
config.output.experiment_name = "perplexity_gpt_oss_120b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
