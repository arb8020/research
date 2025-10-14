"""Small test config - 1 shard, bigger batch size for GPU."""

from config import Config

config = Config()

# Small dataset for quick testing
config.data.num_shards = 1

# Bigger batch size for GPU
config.embedding.batch_size = 128
