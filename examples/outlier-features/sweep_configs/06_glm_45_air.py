"""GLM-4.5-Air (106B total, 12B active, 128 experts).

Re-run with 16 sequences for proper cross-batch validation.
Original run only had 2 valid batches (data quality issue).
Expected: PROBABILISTIC outliers (~49% layer agreement).
Runtime: ~45-60 minutes on 2xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: GLM-4.5-Air
config.model.name = "THUDM/glm-4-air-106b"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences (up from original 4)
config.dataset.num_sequences = 16  # CRITICAL: Need more batches for validation
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Aggressive chunking for very large model
config.analysis.batch_size = 1
config.analysis.layers = None
config.analysis.chunk_layers = 6  # Small chunks for 106B model

# Deployment: Multi-GPU with high VRAM
config.deployment.min_vram = None
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 96
config.deployment.max_price = 4.5
config.deployment.safety_factor = 1.4
config.deployment.container_disk = 150
config.deployment.volume_disk = 0  # No volume to avoid mount issues

# Output
config.output.experiment_name = "glm_45_air_sweep"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
