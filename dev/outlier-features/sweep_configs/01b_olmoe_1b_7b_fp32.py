"""OLMoE-1B-7B with native float32 precision (re-run).

Re-analysis using native float32 instead of downcast bfloat16.
Original model config specifies float32 as native precision.
Expected: PROBABILISTIC outliers (but potentially different magnitudes).
Runtime: ~15-20 minutes on 1xA100.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: OLMoE-1B-7B with NATIVE float32
config.model.name = "allenai/OLMoE-1B-7B-0125"
config.model.device_map = "auto"
config.model.torch_dtype = "float32"  # CHANGED: Use native precision instead of bfloat16

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16  # Up from 4 - for cross-batch validation
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: No chunking needed for small model
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = None  # Small enough to load all layers

# Deployment: Single GPU (float32 requires ~2x memory vs bfloat16)
config.deployment.min_vram = None  # Auto-estimate (will be higher than bfloat16)
config.deployment.gpu_count = 1
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 32
config.deployment.max_price = 2.0
config.deployment.safety_factor = 1.5  # Increased safety factor for fp32

# Output
config.output.experiment_name = "olmoe_1b_7b_fp32_sweep"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
