"""Perplexity smoke test - Qwen3-14B.

Quick perplexity computation to validate pipeline on largest dense model.
Runtime: ~5-10 minutes on 1xGPU.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-14B (dense)
config.model.name = "Qwen/Qwen3-14B"
config.model.device_map = "auto"

# Dataset: Small number of sequences for smoke test
config.dataset.num_sequences = 10  # Just 10 sequences for quick test
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Use perplexity batch size
config.analysis.batch_size = 2  # Process 2 at a time

# Deployment: Single GPU (large VRAM needed)
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 1
config.deployment.gpu_filter = "A100"  # Need larger GPU
config.deployment.min_cpu_ram = 32
config.deployment.max_price = 2.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "perplexity_smoke_qwen3_14b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
