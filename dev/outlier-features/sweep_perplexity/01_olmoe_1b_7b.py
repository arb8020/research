"""Perplexity computation - OLMoE-1B-7B.

Compute perplexity on FineWeb-Edu for smallest MoE model.
Runtime: ~10-15 minutes on 1xA100.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: OLMoE-1B-7B (7B total, 1.3B active)
config.model.name = "allenai/OLMoE-1B-7B-0125"
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
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 32
config.deployment.max_price = 2.0
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "perplexity_olmoe_1b_7b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
