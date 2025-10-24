"""Mixtral-8x7B (46.7B total, 12.9B active, 8 experts).

Fills gap between Qwen3-30B and GLM-106B. Canonical MoE model.
Tests if active params (12.9B, similar to GLM's 12B) matter.
Expected: PROBABILISTIC outliers.
Runtime: ~30-40 minutes on 2xA100.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Mixtral-8x7B
config.model.name = "mistralai/Mixtral-8x7B-v0.1"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Chunked for memory
config.analysis.batch_size = 1
config.analysis.layers = None
config.analysis.chunk_layers = 8

# Deployment: Multi-GPU
config.deployment.min_vram = None
config.deployment.gpu_count = 2
config.deployment.gpu_filter = "A100"
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "mixtral_8x7b_sweep"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
