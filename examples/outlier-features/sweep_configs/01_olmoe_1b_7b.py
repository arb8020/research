"""OLMoE-1B-7B (7B total, 1.3B active, 64 experts).

Smallest MoE model - baseline for size comparison.
Expected: PROBABILISTIC outliers.
Runtime: ~15-20 minutes on 1xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: OLMoE-1B-7B
config.model.name = "allenai/OLMoE-1B-7B-0125"
config.model.device_map = "auto"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16  # Up from 4 - for cross-batch validation
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: No chunking needed for small model
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = None  # Small enough to load all layers

# Deployment: Single GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 1
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 32
config.deployment.max_price = 2.0
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "olmoe_1b_7b_sweep"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
