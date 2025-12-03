"""Perplexity smoke test - OLMoE-1B-7B.

Quick perplexity computation to validate pipeline on MoE model.
Runtime: ~5-10 minutes on 1xGPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: OLMoE-1B-7B (7B total, 1.3B active, 64 experts)
config.model.name = "allenai/OLMoE-1B-7B-0125"
config.model.device_map = "auto"

# Dataset: Small number of sequences for smoke test
config.dataset.num_sequences = 10  # Just 10 sequences for quick test
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Use perplexity batch size
config.analysis.batch_size = 2  # Process 2 at a time

# Deployment: Single GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 1
config.deployment.gpu_filter = None  # Allow any GPU
config.deployment.min_cpu_ram = 32
config.deployment.max_price = 2.0
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "perplexity_smoke_olmoe_1b_7b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
