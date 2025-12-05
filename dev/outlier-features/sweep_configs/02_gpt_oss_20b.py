"""GPT-OSS-20B (21.5B total, ? active).

First systematic MoE model in original analysis.
Expected: SYSTEMATIC outliers (100% layer agreement, dimension 773).
Runtime: ~25-30 minutes on 2xA100.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: GPT-OSS-20B
config.model.name = "openai/gpt-oss-20b"
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
config.deployment.min_vram = 40  # Explicit: 2x40GB = 80GB total (estimator fails with 404)
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.0
config.deployment.safety_factor = 1.3
config.deployment.container_disk = 150  # Standard for 20B model
config.deployment.volume_disk = 0  # No volume disk (avoid mount issues)

# Output
config.output.experiment_name = "gpt_oss_20b_sweep"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
