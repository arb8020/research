#!/usr/bin/env python3
"""Corpus definitions and streaming access without local downloads."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from datasets import IterableDataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class CorpusConfig:
    """Configuration for a training corpus.

    Following Casey Muratori: Data is transparent, not opaque.
    User controls what corpus to use, we just provide access.
    """

    name: str
    dataset_name: str  # HuggingFace dataset name
    split: str = "train"
    config_name: str | None = None  # HuggingFace dataset config (e.g., "all", "main")
    num_shards: int | None = None  # None = all shards
    streaming: bool = True  # Stream instead of download

    def __post_init__(self):
        assert self.name, "corpus name cannot be empty"
        assert self.dataset_name, "dataset_name cannot be empty"


# Predefined corpus configs
NANOCHAT_PRETRAIN = CorpusConfig(
    name="nanochat_pretrain",
    dataset_name="karpathy/fineweb-edu-100b-shuffle",
    split="train",
    num_shards=240,  # ~24GB, ~11.2B tokens
    streaming=True,
)

FINEWEB_EDU_FULL = CorpusConfig(
    name="fineweb_edu_full",
    dataset_name="HuggingFaceFW/fineweb-edu",
    split="train",
    num_shards=None,  # All 1.3T tokens
    streaming=True,
)

FINEWEB_EDU_SAMPLE = CorpusConfig(
    name="fineweb_edu_sample",
    dataset_name="HuggingFaceFW/fineweb-edu-sample-10BT",
    split="train",
    num_shards=None,
    streaming=True,
)

# Nanochat midtrain corpora
SMOLTALK = CorpusConfig(
    name="smoltalk",
    dataset_name="HuggingFaceTB/smoltalk",
    config_name="all",
    split="train",
    num_shards=None,  # ~460K examples
    streaming=True,
)

MMLU_AUX_TRAIN = CorpusConfig(
    name="mmlu_aux_train",
    dataset_name="cais/mmlu",
    config_name="all",  # Use 'all' config, then select auxiliary_train split
    split="auxiliary_train",
    num_shards=None,  # ~100K examples
    streaming=True,
)

GSM8K_TRAIN = CorpusConfig(
    name="gsm8k_train",
    dataset_name="openai/gsm8k",
    config_name="main",
    split="train",
    num_shards=None,  # ~7.4K examples (train split, NOT test)
    streaming=True,
)

# Nanochat SFT corpora (subset of midtrain + eval datasets)
ARC_EASY_TRAIN = CorpusConfig(
    name="arc_easy_train",
    dataset_name="allenai/ai2_arc",
    config_name="ARC-Easy",
    split="train",
    num_shards=None,  # ~2.3K examples
    streaming=True,
)

ARC_CHALLENGE_TRAIN = CorpusConfig(
    name="arc_challenge_train",
    dataset_name="allenai/ai2_arc",
    config_name="ARC-Challenge",
    split="train",
    num_shards=None,  # ~1.1K examples
    streaming=True,
)


def stream_corpus(config: CorpusConfig) -> Iterator[str]:
    """Stream text from a corpus without downloading.

    Following Casey Muratori: Granularity - let user decide how to chunk.
    This yields raw text, user can chunk however they want.

    Args:
        config: Corpus configuration

    Yields:
        Raw text strings from the corpus

    Example:
        for i, text in enumerate(stream_corpus(NANOCHAT_PRETRAIN)):
            if i >= 1000:
                break
            process_text(text)
    """
    logger.info(f"Streaming corpus: {config.name}")

    dataset = load_dataset(
        config.dataset_name, name=config.config_name, split=config.split, streaming=config.streaming
    )

    # Cast to IterableDataset for type safety
    dataset = cast(IterableDataset, dataset)

    # Limit to num_shards if specified
    if config.num_shards is not None:
        dataset = dataset.take(config.num_shards)

    for item in dataset:
        # Most text datasets have a 'text' field
        if isinstance(item, dict) and "text" in item:
            yield item["text"]
        elif isinstance(item, dict):
            # Fallback: try to find any text-like field
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 0:
                    yield value
                    break


def sample_corpus(config: CorpusConfig, n: int = 10) -> list[str]:
    """Get n sample texts from corpus (for dry-run testing).

    Args:
        config: Corpus configuration
        n: Number of samples to return

    Returns:
        List of text samples

    Example:
        samples = sample_corpus(NANOCHAT_PRETRAIN, n=5)
        for i, text in enumerate(samples):
            print(f"Sample {i}: {text[:100]}...")
    """
    samples = []
    for i, text in enumerate(stream_corpus(config)):
        if i >= n:
            break
        samples.append(text)

    logger.info(f"Sampled {len(samples)} texts from {config.name}")
    return samples


def verify_corpus_access(config: CorpusConfig) -> bool:
    """Dry-run test: verify we can access the corpus.

    Args:
        config: Corpus configuration

    Returns:
        True if corpus is accessible, False otherwise
    """
    try:
        logger.info(f"Verifying access to corpus: {config.name}")
        samples = sample_corpus(config, n=3)

        if len(samples) == 0:
            logger.error(f"Corpus {config.name} returned no samples")
            return False

        logger.info(f"✅ Successfully accessed {config.name}")
        logger.info(f"   First sample: {samples[0][:200]}...")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to access {config.name}: {e}")
        return False


def main():
    """Dry-run test all corpus configs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    corpora = [
        NANOCHAT_PRETRAIN,
        SMOLTALK,
        MMLU_AUX_TRAIN,
        GSM8K_TRAIN,
        ARC_EASY_TRAIN,
        ARC_CHALLENGE_TRAIN,
    ]

    logger.info("=" * 80)
    logger.info("Testing corpus access (dry-run)")
    logger.info("=" * 80)

    results = {}
    for corpus in corpora:
        logger.info(f"\nTesting: {corpus.name}")
        results[corpus.name] = verify_corpus_access(corpus)

    logger.info("\n" + "=" * 80)
    logger.info("Results:")
    logger.info("=" * 80)
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
