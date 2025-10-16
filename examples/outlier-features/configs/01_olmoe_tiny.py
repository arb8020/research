"""Tiny test config for local development.

Fast iteration - just 2 sequences, 4 layers, short context.
Expected runtime: 2-3 minutes on local GPU.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Small 1B MoE model
config.model.name = "allenai/OLMoE-1B-7B-0125-Instruct"

# Dataset: Minimal for speed
config.dataset.num_sequences = 2
config.dataset.sequence_length = 512  # Short context
config.dataset.shuffle = False

# Analysis: Just first 4 layers for speed
config.analysis.batch_size = 1
config.analysis.layers = [0, 1, 2, 3]  # Only 4 layers
config.analysis.chunk_layers = None  # No chunking needed for 4 layers

# Output
config.output.experiment_name = "olmoe_tiny"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
