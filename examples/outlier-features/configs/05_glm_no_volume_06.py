"""GLM-4.5-Air config - fix mount error by removing volume disk.

Previous run (04) failed with container mount error:
  "invalid mount config for type bind field target must not be empty"

Root cause: volumeInGb=200 without volumeMountPath parameter.
Fix: Remove volume_disk entirely, use only container_disk.
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

# Deployment: Fixed mount error
config.deployment.min_vram = 80  # 2×A100 80GB or similar
config.deployment.min_cpu_ram = 96
config.deployment.max_price = 5.00
config.deployment.container_disk = 250  # Increased - all storage in container disk
config.deployment.volume_disk = 0  # REMOVED - was causing mount error
config.deployment.gpu_count = 2
config.deployment.gpu_filter = None
config.deployment.keep_running = True

# Output
config.output.experiment_name = "glm_no_volume_fix"
config.output.log_level = "INFO"
config.output.save_dir = Path("./remote_results")
