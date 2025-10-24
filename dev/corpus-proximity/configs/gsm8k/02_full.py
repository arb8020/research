"""Full config for GPU run of GSM8K similarity measurement."""

from config import Config

config = Config()

# Use defaults (already set in SimilarityConfig)
# Just override if needed
config.similarity.num_gsm8k_samples = 10
config.embedding.batch_size = 128  # Larger batch for GPU
