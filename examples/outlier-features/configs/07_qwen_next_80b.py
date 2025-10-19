"""Config for Qwen3-Next-80B-A3B (80B total, 3B active, 512 experts).

Massive MoE model with 512 experts - tests Qwen architecture at larger scale.
Expected runtime: 30-45 minutes on 2-3xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Qwen3-Next-80B (80B total, 3B active, 512 experts!)
config.model.name = "Qwen/Qwen3-Next-80B-A3B-Instruct"
config.model.device_map = "balanced"  # Multi-GPU balancing

# Dataset: Standard
config.dataset.num_sequences = 4
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Chunked for memory efficiency
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = 6  # Smaller chunks for larger model

# Deployment: Multi-GPU (may need 3 GPUs)
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 3
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 96
config.deployment.max_price = 5.0
config.deployment.safety_factor = 1.4

# Output
config.output.experiment_name = "qwen_next_80b"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
