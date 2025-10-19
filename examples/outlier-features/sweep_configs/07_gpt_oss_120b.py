"""GPT-OSS-120B (117B total, ? active).

Largest model - validates GPT-OSS systematicity at scale.
Expected: SYSTEMATIC outliers (100% layer agreement, dimension 773).
Runtime: ~50-70 minutes on 2-3xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: GPT-OSS-120B
config.model.name = "GAIR/GPT-OSS-120B"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Aggressive chunking for very large model
config.analysis.batch_size = 1
config.analysis.layers = None
config.analysis.chunk_layers = 6

# Deployment: Multi-GPU with high VRAM
config.deployment.min_vram = None
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 96
config.deployment.max_price = 5.0
config.deployment.safety_factor = 1.4

# Output
config.output.experiment_name = "gpt_oss_120b_sweep"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
