"""Qwen3-Next-80B-A3B (80B total, 3B active, 512 experts!).

Tests Qwen architecture at larger scale with massive expert count.
Expected: PROBABILISTIC outliers (Qwen architecture pattern).
Runtime: ~40-50 minutes on 2xA100 with aggressive chunking.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-Next-80B
config.model.name = "Qwen/Qwen3-Next-80B-A3B-Instruct"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Aggressive chunking for large model
config.analysis.batch_size = 1
config.analysis.layers = None
config.analysis.chunk_layers = 6  # Smaller chunks for 80B model

# Deployment: Multi-GPU with higher VRAM
config.deployment.min_vram = None
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64  # Reduced from 96 to match availability
config.deployment.max_price = 5.0  # Increased for 2-GPU availability
config.deployment.safety_factor = 1.4
config.deployment.container_disk = 250  # Increased from 150GB - avoid disk space failure
config.deployment.volume_disk = 0  # No volume disk (avoid mount issues)
config.deployment.analysis_timeout = 5400  # 90 minutes (previous run timed out at 45 min)

# Output
config.output.experiment_name = "qwen_next_80b_sweep"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
