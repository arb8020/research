#!/usr/bin/env python3
"""Compute validation perplexity for language models on FineWeb-Edu.

Lightweight script without nnsight dependency - just loads model and computes
perplexity on dataset sequences.

Adapted for Dettmers Figure 3b replication.
"""

import importlib.util
import json
import logging
import sys
from datetime import datetime
from typing import Any, cast

import torch

# Import local modules
from config import Config
from dataset_utils import get_text_sequences
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

# Import shared logging setup
from shared.logging_config import setup_logging
from shared.retry import retry

logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> Config:
    """Load config from Python file.

    Args:
        config_path: Path to config .py file

    Returns:
        Config object
    """
    assert config_path.endswith(".py"), f"Config must be .py file, got {config_path}"

    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    assert spec is not None, f"Failed to load spec from {config_path}"
    assert spec.loader is not None, f"Spec loader is None for {config_path}"

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "config"), "Config file must define 'config' variable"
    config: Config = module.config
    assert isinstance(config, Config), f"Expected Config object, got {type(config)}"

    return config


@retry(max_attempts=3, delay=30, backoff=2, exceptions=(OSError,))
def load_tokenizer_with_retry(model_name: str) -> PreTrainedTokenizerBase:
    """Load tokenizer with retry on HuggingFace rate limits.

    External boundary: Network I/O to HuggingFace API.
    Retries on 429 rate limit errors with exponential backoff: 30s, 60s, 120s.
    Total max wait time: ~3.5 minutes.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Loaded tokenizer

    Raises:
        OSError: If loading fails after all retry attempts
    """
    assert model_name, "model_name must not be empty"

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer is not None, "Tokenizer loading returned None"
    return tokenizer


@retry(max_attempts=3, delay=30, backoff=2, exceptions=(OSError,))
def load_model_with_retry(
    model_name: str, device_map: str, torch_dtype, max_memory: dict | None = None
) -> AutoModelForCausalLM:
    """Load model with retry on HuggingFace rate limits.

    External boundary: Network I/O to HuggingFace API.
    Retries on 429 rate limit errors with exponential backoff: 30s, 60s, 120s.

    Args:
        model_name: HuggingFace model identifier
        device_map: Device mapping strategy ("auto" or "balanced")
        torch_dtype: PyTorch dtype for model weights
        max_memory: Optional memory limits per GPU

    Returns:
        Loaded model in eval mode

    Raises:
        OSError: If loading fails after all retry attempts
    """
    assert model_name, "model_name must not be empty"
    assert device_map in ["auto", "balanced"], f"Invalid device_map: {device_map}"
    assert torch_dtype is not None, "torch_dtype must not be None"

    if max_memory is None:
        model = cast(
            AutoModelForCausalLM,
            AutoModelForCausalLM.from_pretrained(
                model_name, device_map=device_map, torch_dtype=torch_dtype
            ),
        )
    else:
        model = cast(
            AutoModelForCausalLM,
            AutoModelForCausalLM.from_pretrained(
                model_name, device_map=device_map, torch_dtype=torch_dtype, max_memory=max_memory
            ),
        )

    assert model is not None, "Model loading returned None"
    # Cast to Any to call .eval() (exists at runtime)
    cast(Any, model).eval()
    return model


def load_model_and_tokenizer(
    config: Config,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load model and tokenizer without nnsight wrapper.

    Args:
        config: Configuration object

    Returns:
        Tuple of (model, tokenizer)
    """
    assert config is not None, "Config must not be None"
    assert config.model.name, "Config must specify model name"

    logger.info("=" * 80)
    logger.info("LOADING MODEL AND TOKENIZER")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model.name}")

    # Auto-detect available GPUs
    gpu_count = torch.cuda.device_count()
    logger.info(f"Detected {gpu_count} GPU(s)")

    # Load tokenizer with retry
    logger.info("Loading tokenizer (will retry on rate limits)...")
    tokenizer = load_tokenizer_with_retry(config.model.name)
    logger.info(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Load model with retry
    logger.info("Loading model (will retry on rate limits)...")
    torch_dtype = getattr(torch, config.model.torch_dtype)
    assert torch_dtype is not None, f"Invalid torch_dtype: {config.model.torch_dtype}"

    if gpu_count == 1:
        logger.info("Using single-GPU configuration with device_map='auto'")
        model = load_model_with_retry(config.model.name, device_map="auto", torch_dtype=torch_dtype)
    else:
        logger.info("Using multi-GPU balanced configuration...")
        max_memory = {i: "76GiB" for i in range(gpu_count)}
        model = load_model_with_retry(
            config.model.name, device_map="balanced", torch_dtype=torch_dtype, max_memory=max_memory
        )

    assert model is not None, "Model loading failed"
    logger.info("✓ Model loaded successfully")
    logger.info("=" * 80 + "\n")

    return model, tokenizer


def compute_perplexity_single_text(
    model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerBase, text: str, sequence_length: int
) -> tuple[float, int]:
    """Compute loss and token count for a single text.

    Args:
        model: Loaded language model
        tokenizer: Tokenizer for the model
        text: Input text
        sequence_length: Target sequence length in tokens

    Returns:
        Tuple of (loss_value, num_tokens)
    """
    assert model is not None, "Model must not be None"
    assert tokenizer is not None, "Tokenizer must not be None"
    assert text, "Text must not be empty"
    assert sequence_length > 0, f"sequence_length must be positive, got {sequence_length}"

    # Tokenize
    # Cast tokenizer to Any to call it (tokenizers are callable at runtime)
    tokenizer_any = cast(Any, tokenizer)
    inputs = tokenizer_any(
        text, return_tensors="pt", truncation=True, max_length=sequence_length, padding=False
    )

    # Move to same device as model
    # Cast model to Any to access .device attribute (exists at runtime)
    model_any = cast(Any, model)
    input_ids = inputs.input_ids.to(model_any.device)
    num_tokens = input_ids.shape[1]

    # Skip if sequence is too short
    if num_tokens < 2:
        return 0.0, 0

    # Compute loss (standard causal LM: predict next token)
    # Cast model to Any to call it (models are callable at runtime)
    outputs = model_any(input_ids, labels=input_ids)
    loss = outputs.loss
    assert loss is not None, "Model did not return loss"

    return loss.item(), num_tokens


def compute_perplexity_on_batch(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    sequence_length: int,
) -> dict:
    """Compute perplexity on a batch of text sequences.

    Args:
        model: Loaded language model
        tokenizer: Tokenizer for the model
        texts: List of text sequences
        sequence_length: Target sequence length in tokens

    Returns:
        Dict with perplexity metrics
    """
    assert model is not None, "Model must not be None"
    assert tokenizer is not None, "Tokenizer must not be None"
    assert texts, "Texts list must not be empty"
    assert len(texts) > 0, f"Must have at least one text, got {len(texts)}"
    assert sequence_length > 0, f"sequence_length must be positive, got {sequence_length}"

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity", leave=False):
            loss_value, num_tokens = compute_perplexity_single_text(
                model, tokenizer, text, sequence_length
            )
            total_loss += loss_value * num_tokens
            total_tokens += num_tokens

    assert total_tokens > 0, "No tokens processed"

    # Compute average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_sequences": len(texts),
    }


def process_batches(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    text_sequences: list[str],
    config: Config,
) -> list[dict]:
    """Process sequences in batches and compute perplexity.

    Args:
        model: Loaded language model
        tokenizer: Tokenizer for the model
        text_sequences: All text sequences to process
        config: Configuration object

    Returns:
        List of batch results
    """
    assert model is not None, "Model must not be None"
    assert tokenizer is not None, "Tokenizer must not be None"
    assert text_sequences, "text_sequences must not be empty"
    assert config is not None, "Config must not be None"

    batch_size = config.analysis.batch_size
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"

    all_results = []
    num_batches = (len(text_sequences) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(text_sequences), batch_size):
        batch_texts = text_sequences[batch_idx : batch_idx + batch_size]
        logger.info(f"Processing batch {batch_idx // batch_size + 1}/{num_batches}")

        batch_result = compute_perplexity_on_batch(
            model, tokenizer, batch_texts, config.dataset.sequence_length
        )
        all_results.append(batch_result)

        logger.info(f"  Batch perplexity: {batch_result['perplexity']:.4f}")

        # Clear cache
        torch.cuda.empty_cache()

    assert len(all_results) > 0, "No batches processed"
    return all_results


def aggregate_batch_results(batch_results: list[dict]) -> dict:
    """Aggregate results across all batches.

    Args:
        batch_results: List of batch result dicts

    Returns:
        Dict with final aggregated metrics
    """
    assert batch_results, "batch_results must not be empty"
    assert len(batch_results) > 0, "Must have at least one batch result"

    total_loss = sum(r["avg_loss"] * r["total_tokens"] for r in batch_results)
    total_tokens = sum(r["total_tokens"] for r in batch_results)
    assert total_tokens > 0, "No tokens in batch results"

    avg_loss = total_loss / total_tokens
    final_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "perplexity": final_perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_sequences": sum(r["num_sequences"] for r in batch_results),
    }


def save_results(results: dict, config: Config, batch_results: list[dict]):
    """Save results to disk.

    Args:
        results: Final aggregated results
        config: Configuration object
        batch_results: List of batch results
    """
    assert results is not None, "Results must not be None"
    assert config is not None, "Config must not be None"
    assert batch_results is not None, "batch_results must not be None"

    config.output.save_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        **results,
        "model": config.model.name,
        "dataset": config.dataset.name,
        "sequence_length": config.dataset.sequence_length,
        "timestamp": datetime.now().isoformat(),
        "batch_results": batch_results,
    }

    results_file = config.output.save_dir / "perplexity_results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"📁 Results saved: {results_file}")

    # Save config used
    config.save(config.output.save_dir / "config.json")


def run_perplexity_pipeline(config: Config) -> int:
    """Run the perplexity computation pipeline.

    Args:
        config: Configuration object

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    assert config is not None, "Config must not be None"

    logger.info("=" * 80)
    logger.info("PERPLEXITY COMPUTATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(
        f"Sequences: {config.dataset.num_sequences} x {config.dataset.sequence_length} tokens"
    )
    if config.output.experiment_name:
        logger.info(f"Experiment: {config.output.experiment_name}")
    logger.info("=" * 80 + "\n")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Load dataset sequences
    logger.info("=" * 80)
    logger.info("LOADING DATASET")
    logger.info("=" * 80)
    logger.info("Loading dataset sequences...")
    text_sequences = get_text_sequences(
        dataset_name=config.dataset.name,
        num_sequences=config.dataset.num_sequences,
        sequence_length=config.dataset.sequence_length,
        tokenizer=tokenizer,
        split=config.dataset.split,
        skip_sequences=config.dataset.skip_sequences,
        shuffle=config.dataset.shuffle,
        seed=config.dataset.seed,
        buffer_size=config.dataset.shuffle_buffer,
    )
    logger.info(f"✓ Loaded {len(text_sequences)} sequences")
    logger.info("=" * 80 + "\n")

    # Compute perplexity
    logger.info("=" * 80)
    logger.info("COMPUTING PERPLEXITY")
    logger.info("=" * 80)

    batch_results = process_batches(model, tokenizer, text_sequences, config)
    final_results = aggregate_batch_results(batch_results)

    logger.info("=" * 80)
    logger.info(f"✓ Final perplexity: {final_results['perplexity']:.4f}")
    logger.info(f"  Total tokens: {final_results['total_tokens']:,}")
    logger.info(f"  Total sequences: {final_results['num_sequences']}")
    logger.info("=" * 80 + "\n")

    # Save results
    save_results(final_results, config, batch_results)

    logger.info("🎉 PERPLEXITY COMPUTATION COMPLETE")
    return 0


def main():
    """Main entry point with error handling boundary."""
    try:
        # Load config
        if len(sys.argv) > 1 and sys.argv[1].endswith(".py"):
            config = load_config_from_file(sys.argv[1])
        else:
            config = Config()  # Use defaults

        # Setup logging
        setup_logging(
            level=config.output.log_level,
            logger_levels={"httpx": "WARNING", "urllib3": "WARNING", "transformers": "WARNING"},
        )

        # Run pipeline
        return run_perplexity_pipeline(config)

    except Exception as e:
        logger.error(f"✗ Perplexity computation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
