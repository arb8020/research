"""Baseline config for OLMoE-1B model.

Standard analysis - 4 sequences, all layers, full context.
Expected runtime: 10-15 minutes on A100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Small 1B MoE model
config.model.name = "allenai/OLMoE-1B-7B-0125-Instruct"

# Dataset: Standard
config.dataset.num_sequences = 4
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: All layers
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = None  # No chunking for 1B model

# Deployment: Single GPU
config.deployment.min_vram = 24  # Should fit on RTX 4090 or A100
config.deployment.gpu_count = 1
config.deployment.gpu_filter = None  # Any GPU with sufficient VRAM

# Output
config.output.experiment_name = "olmoe_baseline"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
