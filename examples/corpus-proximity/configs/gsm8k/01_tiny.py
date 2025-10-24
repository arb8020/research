"""Tiny config for local testing of GSM8K similarity measurement."""

from config import Config

config = Config()

# Override for tiny test
config.similarity.num_gsm8k_samples = 3
config.similarity.corpus_sizes = {
    "pretrain": 100,
    "midtrain": 90,
    "sft": 100,
}
config.similarity.output_file = "gsm8k_similarity_tiny.csv"
config.embedding.batch_size = 32  # Smaller batch for CPU
