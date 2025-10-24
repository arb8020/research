"""Qwen3-Next-80B test config with 2 GPUs (optimized for memory).

Tests if 2xA100 (80GB each = 160GB total) is sufficient for 80B model.
Previous attempts:
- 09: 1xA100 ran out of disk space (150GB insufficient)
- 07: 3xA100 worked but may be over-provisioned

Key optimizations for 2-GPU deployment:
- Increased container disk to 250GB (model weights ~160GB)
- Aggressive layer chunking (chunk_layers=4)
- balanced device_map for multi-GPU memory distribution
- No volume disk (avoids mount errors)

Expected runtime: 5-8 minutes on 2xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-Next-80B (80B total, 3B active, 512 experts)
config.model.name = "Qwen/Qwen3-Next-80B-A3B-Instruct"
config.model.device_map = "balanced"  # Critical for 2-GPU distribution

# Dataset: Small test first (upgrade to 16 if successful)
config.dataset.num_sequences = 4
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Aggressive chunking for 2-GPU constraint
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = 4  # Smaller chunks than 3-GPU config (was 6)

# Deployment: 2xA100 80GB (160GB total VRAM)
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64  # Reduced to match availability (was 96)
config.deployment.max_price = 5.0  # Increased for 2-GPU availability
config.deployment.safety_factor = 1.4
config.deployment.container_disk = 250  # CRITICAL: Increased from 150GB
config.deployment.volume_disk = 0  # No volume disk (avoid mount issues)

# Output
config.output.experiment_name = "qwen_next_2gpu_test"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
