#!/usr/bin/env python3
"""Push-button script to extract activations and analyze for outlier features.

Combines extraction and analysis into a single workflow with config-based execution.
Adapted from llm-workbench/examples/outlier_features_moe/run_full_analysis.py

Split into functions <70 lines following Tiger Style.
"""

import sys
import json
import shutil
import logging
import importlib.util
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer

# Import shared logging setup
from shared.logging_config import setup_logging

# Import local modules
from config import Config
from extract_activations import extract_activations_optimized
from analyze_activations import analyze_run_for_outliers
from dataset_utils import get_text_sequences

logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> Config:
    """Load config from Python file.

    Args:
        config_path: Path to config .py file

    Returns:
        Config object

    Raises:
        ImportError: If config file cannot be loaded
        AttributeError: If config file doesn't have 'config' variable
    """
    assert config_path.endswith('.py'), f"Config must be .py file, got {config_path}"

    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    assert spec is not None, f"Failed to load spec from {config_path}"
    assert spec.loader is not None, f"Spec loader is None for {config_path}"

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'config'), f"Config file must define 'config' variable"
    config = module.config
    assert isinstance(config, Config), f"Expected Config object, got {type(config)}"

    return config


def load_dataset_sequences(config: Config) -> tuple[AutoTokenizer, list[str]]:
    """Load tokenizer and dataset sequences.

    Args:
        config: Configuration object

    Returns:
        Tuple of (tokenizer, text_sequences)

    Raises:
        RuntimeError: If tokenizer or dataset loading fails
    """
    logger.info("="*80)
    logger.info("LOADING TOKENIZER AND DATASET")
    logger.info("="*80)

    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model.name)
        logger.info(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")

        # Load dataset sequences using tokenizer
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
        logger.info("="*80 + "\n")

        return tokenizer, text_sequences

    except Exception as e:
        logger.error(f"✗ Dataset/tokenizer loading failed: {e}")
        raise RuntimeError(f"Failed to load dataset: {e}")


def load_model_optimized(config: Config):
    """Load model with memory-optimized settings.

    Args:
        config: Configuration object

    Returns:
        Loaded LanguageModel instance
    """
    import torch
    from nnsight import LanguageModel

    logger.info("="*80)
    logger.info("LOADING MODEL (MEMORY OPTIMIZED)")
    logger.info("="*80)
    logger.info(f"Model: {config.model.name}")

    # Auto-detect available GPUs
    gpu_count = torch.cuda.device_count()
    logger.info(f"Detected {gpu_count} GPU(s)")

    # Configure device mapping
    if gpu_count == 1:
        logger.info("Using single-GPU configuration with device_map='auto'")
        llm = LanguageModel(
            config.model.name,
            device_map="auto",
            torch_dtype=getattr(torch, config.model.torch_dtype)
        )
    else:
        logger.info(f"Using multi-GPU balanced configuration with 76GB per GPU limit...")
        max_memory = {i: "76GiB" for i in range(gpu_count)}
        llm = LanguageModel(
            config.model.name,
            device_map="balanced",
            max_memory=max_memory,
            torch_dtype=getattr(torch, config.model.torch_dtype)
        )

    # Disable KV cache to save memory
    llm.model.config.use_cache = config.model.use_cache

    logger.info("✓ Model loaded successfully")
    logger.info("="*80 + "\n")

    return llm


def cleanup_huggingface_cache():
    """Clean up HuggingFace cache to free disk space."""
    import os

    logger.info("🧹 Cleaning up HuggingFace cache to free disk space...")

    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(cache_dir)
            for filename in filenames
        ) / (1024**3)  # GB
        logger.info(f"   Cache directory size: {cache_size:.2f} GB")
        shutil.rmtree(cache_dir)
        logger.info(f"✓ Deleted HuggingFace cache ({cache_size:.2f} GB freed)")
    else:
        logger.info("   No HuggingFace cache found")


def process_single_batch(
    llm,
    batch_texts: list[str],
    batch_idx: int,
    config: Config,
    save_dir: Path
) -> dict:
    """Extract and analyze activations for a single batch.

    Args:
        llm: Loaded LanguageModel
        batch_texts: Texts for this batch
        batch_idx: Batch index (0-based)
        config: Configuration object
        save_dir: Base save directory

    Returns:
        Dict with batch results
    """
    import torch

    logger.info(f"\n{'='*20} BATCH {batch_idx + 1} {'='*20}")
    logger.info(f"Processing {len(batch_texts)} sequences")

    # Step 1: Extract activations
    run_dir, metadata = extract_activations_optimized(
        llm=llm,
        texts=batch_texts,
        layers=config.analysis.layers,
        save_dir=str(save_dir),
        chunk_size=config.analysis.chunk_layers
    )
    logger.info(f"✓ Activation extraction completed: {run_dir}")

    # Step 2: Analyze for outliers
    logger.info(f"🔍 Analyzing batch {batch_idx + 1} for outliers...")
    systematic_outliers, outlier_info = analyze_run_for_outliers(
        run_dir=run_dir,
        magnitude_threshold=config.analysis.magnitude_threshold,
        min_layer_percentage=config.analysis.min_layer_percentage,
        min_seq_percentage=config.analysis.min_seq_percentage
    )

    # Step 3: Create batch result summary
    batch_result = {
        "batch_id": batch_idx + 1,
        "run_dir": str(run_dir),
        "sequences_processed": len(batch_texts),
        "systematic_outliers": systematic_outliers,
        "outlier_info": outlier_info,
        "timestamp": datetime.now().isoformat()
    }

    logger.info(f"✓ Batch {batch_idx + 1}: Found {len(systematic_outliers)} systematic outlier features")

    # Step 4: Cleanup activation files to save disk space
    run_dir_path = Path(run_dir)
    if run_dir_path.exists():
        disk_freed_mb = sum(f.stat().st_size for f in run_dir_path.rglob('*.pt')) / (1024*1024)
        shutil.rmtree(run_dir_path)
        logger.info(f"🗑️  Cleaned up activation files: {run_dir_path.name} ({disk_freed_mb:.1f}MB freed)")

    # Step 5: Clear GPU cache
    torch.cuda.empty_cache()

    return batch_result


def aggregate_batch_results(batch_results: list[dict], config: Config) -> dict:
    """Aggregate results across all batches.

    Args:
        batch_results: List of batch result dicts
        config: Configuration object

    Returns:
        Dict with aggregated results
    """
    logger.info("="*80)
    logger.info("AGGREGATING RESULTS")
    logger.info("="*80)
    logger.info(f"Aggregating results from {len(batch_results)} completed batches...")

    all_systematic_outliers = []
    for batch_result in batch_results:
        all_systematic_outliers.extend(batch_result['systematic_outliers'])
        logger.info(f"Batch {batch_result['batch_id']}: "
                   f"{len(batch_result['systematic_outliers'])} systematic outlier features")

    logger.info(f"Total systematic outlier features found: {len(all_systematic_outliers)}")

    # Aggregate features by dimension
    feature_aggregates = {}
    if all_systematic_outliers:
        for feature in all_systematic_outliers:
            dim = feature['feature_dim']
            if dim not in feature_aggregates:
                feature_aggregates[dim] = {
                    'feature_dim': dim,
                    'max_magnitude': feature['max_magnitude'],
                    'occurrences': 1
                }
            else:
                feature_aggregates[dim]['max_magnitude'] = max(
                    feature_aggregates[dim]['max_magnitude'],
                    feature['max_magnitude']
                )
                feature_aggregates[dim]['occurrences'] += 1

        # Sort by max magnitude
        top_features = sorted(
            feature_aggregates.values(),
            key=lambda x: x['max_magnitude'],
            reverse=True
        )

        logger.info(f"\nTop outlier features across all batches:")
        for i, feature in enumerate(top_features[:5], 1):
            logger.info(
                f"  {i}. Feature {feature['feature_dim']}: "
                f"max_mag={feature['max_magnitude']:.2f}, "
                f"appeared in {feature['occurrences']}/{len(batch_results)} batches"
            )
    else:
        top_features = []
        logger.info("\nNo systematic outlier features found across any batch.")

    logger.info("="*80 + "\n")

    return {
        'all_systematic_outliers': all_systematic_outliers,
        'top_features': top_features
    }


def save_final_results(
    batch_results: list[dict],
    aggregated: dict,
    config: Config,
    output_dir: Path
):
    """Save final aggregated results to disk.

    Args:
        batch_results: List of batch result dicts
        aggregated: Aggregated results dict
        config: Configuration object
        output_dir: Output directory
    """
    # Compute shard range
    shard_start = config.dataset.skip_sequences
    shard_end = shard_start + config.dataset.num_sequences - 1

    final_results = {
        "analysis_summary": {
            "total_batches": len(batch_results),
            "total_sequences": sum(br['sequences_processed'] for br in batch_results),
            "total_systematic_outliers": len(aggregated['all_systematic_outliers']),
            "completion_time": datetime.now().isoformat()
        },
        "run_config": {
            "model": config.model.name,
            "dataset": config.dataset.name,
            "num_sequences": config.dataset.num_sequences,
            "sequence_length": config.dataset.sequence_length,
            "batch_size": config.analysis.batch_size,
            "threshold": config.analysis.magnitude_threshold,
            "layers": config.analysis.layers,
            "shard_range_inclusive": [shard_start, shard_end],
            "experiment_name": config.output.experiment_name,
        },
        "top_features": aggregated['top_features'],
        "all_systematic_outliers": aggregated['all_systematic_outliers'],
        "batch_results": batch_results
    }

    final_results_file = output_dir / "final_analysis_results.json"
    with open(final_results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"📁 Final results saved: {final_results_file}")


def main():
    """Main orchestrator - config loading and batch coordination only."""
    # Load config
    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        config = load_config_from_file(sys.argv[1])
    else:
        config = Config()  # Use defaults

    # Setup logging
    setup_logging(level=config.output.log_level)

    logger.info("="*80)
    logger.info("OUTLIER FEATURES ANALYSIS PIPELINE")
    logger.info("="*80)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Sequences: {config.dataset.num_sequences} x {config.dataset.sequence_length} tokens")
    logger.info(f"Batch size: {config.analysis.batch_size}")
    logger.info(f"Threshold: {config.analysis.magnitude_threshold}")
    if config.output.experiment_name:
        logger.info(f"Experiment: {config.output.experiment_name}")
    logger.info("="*80 + "\n")

    try:
        # Validate config
        assert config.dataset.num_sequences > 0, "num_sequences must be positive"
        assert config.dataset.num_sequences % config.analysis.batch_size == 0, \
            f"num_sequences ({config.dataset.num_sequences}) must be divisible by batch_size ({config.analysis.batch_size})"

        # Load dataset
        tokenizer, text_sequences = load_dataset_sequences(config)

        # Load model
        llm = load_model_optimized(config)

        # Cleanup cache
        cleanup_huggingface_cache()

        # Process batches
        logger.info("="*80)
        logger.info("EXTRACTING AND ANALYZING ACTIVATIONS")
        logger.info("="*80)

        num_batches = config.dataset.num_sequences // config.analysis.batch_size
        batch_results = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.analysis.batch_size
            end_idx = start_idx + config.analysis.batch_size
            batch_texts = text_sequences[start_idx:end_idx]

            batch_result = process_single_batch(
                llm, batch_texts, batch_idx, config, config.output.save_dir
            )
            batch_results.append(batch_result)

            # Save intermediate batch result
            batch_file = config.output.save_dir / f"batch_{batch_idx + 1:03d}_results.json"
            batch_file.parent.mkdir(parents=True, exist_ok=True)
            with open(batch_file, 'w') as f:
                json.dump(batch_result, f, indent=2)

        logger.info(f"\n✓ All {num_batches} batches completed!")

        # Aggregate results
        aggregated = aggregate_batch_results(batch_results, config)

        # Save final results
        config.output.save_dir.mkdir(parents=True, exist_ok=True)
        save_final_results(batch_results, aggregated, config, config.output.save_dir)

        # Save config used
        config.save(config.output.save_dir / "config.json")

        logger.info("🎉 ANALYSIS COMPLETE")
        return 0

    except Exception as e:
        logger.error(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
