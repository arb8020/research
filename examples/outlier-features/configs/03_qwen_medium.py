"""Medium-scale config for Qwen3-30B model.

Large MoE model - requires multi-GPU and layer chunking.
Expected runtime: 20-30 minutes on 2xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Large 30B MoE model
config.model.name = "Qwen/Qwen3-30B-A3B"
config.model.device_map = "balanced"  # Multi-GPU balancing

# Dataset: Standard
config.dataset.num_sequences = 4
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Chunked for memory efficiency
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = 8  # Process 8 layers at a time

# Deployment: Multi-GPU
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.50
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "qwen_medium"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
