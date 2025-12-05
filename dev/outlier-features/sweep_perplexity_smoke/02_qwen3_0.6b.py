"""Perplexity smoke test - Qwen3-0.6B.

Quick perplexity computation to validate pipeline on small dense model.
Runtime: ~5-10 minutes on 1xGPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-0.6B (dense)
config.model.name = "Qwen/Qwen3-0.6B"
config.model.device_map = "auto"

# Dataset: Small number of sequences for smoke test
config.dataset.num_sequences = 10  # Just 10 sequences for quick test
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Use perplexity batch size
config.analysis.batch_size = 4  # Smaller model, can process more at once

# Deployment: Single GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 1
config.deployment.gpu_filter = None  # Allow any GPU
config.deployment.min_cpu_ram = 16
config.deployment.max_price = 1.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "perplexity_smoke_qwen3_0.6b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
