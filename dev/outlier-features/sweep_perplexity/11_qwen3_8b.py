"""Perplexity computation - Qwen3-8B.

DENSE model - above Dettmers' 6.7B phase transition threshold.
Runtime: ~15-20 minutes on 1xA100.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-8B (dense)
config.model.name = "Qwen/Qwen3-8B"
config.model.device_map = "auto"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Perplexity: Standard batch size
config.perplexity.batch_size = 1
config.perplexity.stride = None  # No overlapping windows

# Deployment: Single GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 1
config.deployment.gpu_filter = None  # Allow any GPU
config.deployment.min_cpu_ram = 48
config.deployment.max_price = 2.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "perplexity_qwen3_8b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
