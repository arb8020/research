"""Perplexity computation - Qwen3-Next-80B.

Tests Qwen architecture at larger scale with massive expert count.
Runtime: ~25-35 minutes on 2xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-Next-80B (80B total, 3B active, 512 experts!)
config.model.name = "Qwen/Qwen3-Next-80B-A3B-Instruct"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Perplexity: Standard batch size
config.perplexity.batch_size = 1
config.perplexity.stride = None  # No overlapping windows

# Deployment: Multi-GPU with higher disk space
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 5.0
config.deployment.safety_factor = 1.4
config.deployment.container_disk = 250  # Large model needs more disk
config.deployment.volume_disk = 0  # No volume to avoid mount issues

# Output
config.output.experiment_name = "perplexity_qwen_next_80b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
