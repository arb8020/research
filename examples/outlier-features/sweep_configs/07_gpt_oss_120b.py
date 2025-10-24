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
config.model.name = "openai/gpt-oss-120b"
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
config.deployment.min_vram = 80  # Explicit: 2x80GB = 160GB total (estimator fails with 404)
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64  # Reduced from 96 to match availability
config.deployment.max_price = 5.0  # Already at $5/hr
config.deployment.safety_factor = 1.4
config.deployment.container_disk = 250  # Increased from 150GB - avoid disk space failure
config.deployment.volume_disk = 0  # No volume disk (avoid mount issues)

# Output
config.output.experiment_name = "gpt_oss_120b_sweep"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
