"""Perplexity computation - Mixtral-8x7B.

Well-known MoE model for comparison.
Runtime: ~20-25 minutes on 2xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Mixtral-8x7B (47B total, 13B active)
config.model.name = "mistralai/Mixtral-8x7B-v0.1"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Perplexity: Standard batch size
config.perplexity.batch_size = 1
config.perplexity.stride = None  # No overlapping windows

# Deployment: Multi-GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.0
config.deployment.safety_factor = 1.3
config.deployment.container_disk = 150  # Standard for 47B model
config.deployment.volume_disk = 0  # No volume to avoid mount issues

# Output
config.output.experiment_name = "perplexity_mixtral_8x7b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
