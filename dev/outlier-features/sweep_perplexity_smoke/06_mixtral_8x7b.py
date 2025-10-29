"""Perplexity smoke test - Mixtral-8x7B.

Quick perplexity computation to validate pipeline.
Runtime: ~5-10 minutes on 2xGPU.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Mixtral-8x7B (46.7B total, 12.9B active)
config.model.name = "mistralai/Mixtral-8x7B-v0.1"
config.model.device_map = "balanced"

# Dataset: Small number of sequences for smoke test
config.dataset.num_sequences = 10  # Just 10 sequences for quick test
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Use perplexity batch size
config.analysis.batch_size = 2  # Process 2 at a time

# Deployment: Multi-GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "perplexity_smoke_mixtral_8x7b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
