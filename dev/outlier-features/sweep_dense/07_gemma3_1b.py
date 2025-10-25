"""Gemma-3-1B-IT (dense baseline).

DENSE model for comparison with MoE architectures.
Expected: PROBABILISTIC outliers (below 6.7B threshold).
Runtime: ~12-18 minutes on 1xGPU.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Gemma-3-1B-IT (dense)
config.model.name = "google/gemma-3-1b-it"
config.model.device_map = "auto"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: No chunking needed for small model
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = None

# Deployment: Single GPU, allow cheaper GPUs
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 1
config.deployment.gpu_filter = None  # Allow any GPU
config.deployment.min_cpu_ram = 16
config.deployment.max_price = 1.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "gemma3_1b_dense"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
