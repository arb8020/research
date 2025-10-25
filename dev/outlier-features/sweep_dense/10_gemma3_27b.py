"""Gemma-3-27B-IT (dense baseline).

DENSE model for comparison with MoE architectures.
Well above Dettmers' 6.7B phase transition threshold.
Comparable to Qwen3-30B MoE (30.5B total params).
Expected: SYSTEMATIC outliers (if Dettmers' findings hold for Gemma).
Runtime: ~40-50 minutes on 2xGPU.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

config = Config()

# Model: Gemma-3-27B-IT (dense)
config.model.name = "google/gemma-3-27b-it"
config.model.device_map = "balanced"

# Dataset: Standard 16 sequences for bootstrap CIs
config.dataset.num_sequences = 16
config.dataset.sequence_length = 2048
config.dataset.shuffle = False

# Analysis: Chunked for memory
config.analysis.batch_size = 1
config.analysis.layers = None  # All layers
config.analysis.chunk_layers = 8

# Deployment: Multi-GPU, allow cheaper GPUs
config.deployment.min_vram = None  # Auto-estimate
config.deployment.gpu_count = 2
config.deployment.gpu_filter = None  # Allow any GPU
config.deployment.min_cpu_ram = 64
config.deployment.max_price = 3.5
config.deployment.safety_factor = 1.3

# Output
config.output.experiment_name = "gemma3_27b_dense"
config.output.log_level = "INFO"
config.output.save_dir = Path("./results")
