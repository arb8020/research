"""Qwen3-30B-A3B (30.5B total, 3.3B active, 60 experts).

Counterexample: Larger than GPT-20B but PROBABILISTIC.
Expected: PROBABILISTIC outliers (~40% layer agreement).
Runtime: ~30-35 minutes on 2xA100.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-30B-A3B
config.model.name = "Qwen/Qwen3-30B-A3B"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Chunked for memory
config.analysis.batch_size = 1
config.analysis.layers = None
config.analysis.chunk_layers = 8

# Deployment: Multi-GPU
config.deployment.min_vram = None
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "qwen3_30b_sweep"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
