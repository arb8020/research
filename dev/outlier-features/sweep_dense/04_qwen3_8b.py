"""Qwen3-8B (dense baseline).

DENSE model for comparison with MoE architectures.
CRITICAL: Just above Dettmers' 6.7B phase transition threshold.
Expected: SYSTEMATIC outliers (if Dettmers' findings hold for Qwen).
Runtime: ~25-30 minutes on 1xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-8B (dense)
config.model.name = "Qwen/Qwen3-8B"
config.model.device_map = "auto"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: May need chunking for memory
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = 8  # Chunk to manage memory

# Deployment: Single GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 1
config.deployment.gpu_filter = None  # Allow any GPU
config.deployment.min_cpu_ram = 32
config.deployment.max_price = 2.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "qwen3_8b_dense"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
