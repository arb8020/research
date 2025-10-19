"""Mixtral-8x7B tiny test config (quick validation).

Minimal config to test that Mixtral runs without errors.
Expected runtime: 3-5 minutes on 1xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Mixtral-8x7B
config.model.name = "mistralai/Mixtral-8x7B-v0.1"
config.model.device_map = "auto"

# Dataset: Tiny test (2 sequences, short length)
config.dataset.num_sequences = 2
config.dataset.sequence_length = 512  # Shorter for quick test
config.dataset.shuffle = False

# Analysis: No chunking for test
config.analysis.batch_size = 1
config.analysis.layers = [0, 1, 2, 3]  # Just first 4 layers
config.analysis.chunk_layers = None

# Deployment: Single GPU
config.deployment.min_vram = None
config.deployment.gpu_count = 1
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 32
config.deployment.max_price = 2.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "mixtral_tiny_test"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
