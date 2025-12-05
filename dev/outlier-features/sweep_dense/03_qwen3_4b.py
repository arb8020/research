"""Qwen3-4B (dense baseline).

DENSE model for comparison with MoE architectures.
Expected: PROBABILISTIC outliers (below 6.7B threshold).
Runtime: ~20-25 minutes on 1xA100.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-4B (dense)
config.model.name = "Qwen/Qwen3-4B"
config.model.device_map = "auto"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: No chunking needed
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = None

# Deployment: Single GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 1
config.deployment.gpu_filter = None  # Allow any GPU
config.deployment.min_cpu_ram = 24
config.deployment.max_price = 2.0
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "qwen3_4b_dense"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
