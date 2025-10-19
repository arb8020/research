"""Qwen3-Next-80B tiny test config (quick validation).

Minimal config to test that Qwen3-Next loads and runs.
Expected runtime: 4-6 minutes on 1xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-Next-80B
config.model.name = "Qwen/Qwen3-Next-80B-A3B-Instruct"
config.model.device_map = "auto"

# Dataset: Tiny test (2 sequences, short length)
config.dataset.num_sequences = 2
config.dataset.sequence_length = 512  # Shorter for quick test
config.dataset.shuffle = False

# Analysis: Chunked + limited layers for test
config.analysis.batch_size = 1
config.analysis.layers = [0, 1, 2, 3]  # Just first 4 layers
config.analysis.chunk_layers = 2  # Small chunks for large model

# Deployment: Single GPU (may be tight, but worth testing)
config.deployment.min_vram = None
config.deployment.gpu_count = 1
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.0
config.deployment.safety_factor = 1.4

# Output
config.output.experiment_name = "qwen_next_tiny_test"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
