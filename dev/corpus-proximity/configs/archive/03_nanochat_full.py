"""Full nanochat corpus config - 240 shards from fineweb-edu-100b-shuffle."""

from config import Config

config = Config()

# Full nanochat pretrain corpus
config.data.num_shards = 240

# GPU batch size (adjust based on your GPU memory)
config.embedding.batch_size = 128
