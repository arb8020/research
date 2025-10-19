"""GLM-4.5-Air config - rerun with sufficient disk space.

Previous run failed with "out of disk space" error.
GLM-106B requires ~200GB for model weights + cache.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: GLM-106B (12B active per token)
config.model.name = "zai-org/GLM-4.5-Air"

# Dataset: Standard analysis
config.dataset.num_sequences = 4
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Chunk layers to manage memory
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = 8  # Process 8 layers at a time

# Deployment: Increased disk space for large model
config.deployment.min_vram = 80  # 2×A100 80GB or similar
config.deployment.min_cpu_ram = 96  # Relaxed from 128GB
config.deployment.max_price = 5.00  # Increased to $5/hr for availability
config.deployment.container_disk = 200  # Increased from 150GB
config.deployment.volume_disk = 200  # Reduced from 300GB for availability
config.deployment.gpu_count = 2
config.deployment.gpu_filter = None  # Removed A100 filter - accept any Nvidia 80GB+
config.deployment.keep_running = True  # Keep for inspection

# Output
config.output.experiment_name = "glm_rerun_fixed_disk"
config.output.log_level = "INFO"
config.output.save_dir = Path("./remote_results")
