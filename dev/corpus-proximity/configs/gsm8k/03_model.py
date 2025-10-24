"""Config for testing model-generated answers (requires OpenAI API key)."""

from config import Config

config = Config()

# Very small test - only 2 samples to minimize API costs
config.similarity.num_gsm8k_samples = 2
config.similarity.corpus_sizes = {
    "pretrain": 50,
    "midtrain": 45,
    "sft": 50,
}

# Enable model answer generation
config.similarity.include_model_answers = True
config.similarity.model_name = "gpt-4o-mini"  # Cheapest model
config.similarity.model_temperature = 0.7

# Output
config.similarity.output_file = "gsm8k_similarity_model.csv"

# Smaller batch sizes
config.embedding.batch_size = 32
