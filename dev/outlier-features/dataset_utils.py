"""Pure functions for loading and processing datasets for outlier analysis.

Adapted from dataset_utils.py
"""

import logging
from collections.abc import Iterator
from typing import cast

from datasets import IterableDataset, load_dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_TOKENIZER_MAX_LEN_SENTINEL = 1_000_000_000  # treat extremely large values as "unbounded"


def _effective_sequence_length(
    requested_length: int, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None
) -> int:
    """Return the sequence length we should actually use for this tokenizer."""

    if tokenizer is None:
        return requested_length

    max_length = getattr(tokenizer, "model_max_length", None)

    if isinstance(max_length, int) and 0 < max_length < _TOKENIZER_MAX_LEN_SENTINEL:
        if requested_length > max_length:
            logger.warning(
                "Requested sequence length %s exceeds tokenizer limit %s; clipping to %s",
                requested_length,
                max_length,
                max_length,
            )
            return max_length

    return requested_length


def load_streaming_dataset(
    dataset_name: str,
    split: str = "train",
    *,
    shuffle: bool = False,
    seed: int | None = None,
    buffer_size: int = 10_000,
) -> Iterator[str]:
    """Load streaming dataset and yield text content.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "HuggingFaceFW/fineweb-edu")
        split: Dataset split to use
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling (deterministic if provided)
        buffer_size: Buffer size for streaming shuffle

    Yields:
        str: Text content from dataset

    Raises:
        RuntimeError: If dataset fails to load
    """
    assert isinstance(dataset_name, str), f"Expected str, got {type(dataset_name)}"
    assert isinstance(split, str), f"Expected str, got {type(split)}"
    assert buffer_size > 0, f"buffer_size must be positive, got {buffer_size}"

    try:
        dataset = cast(IterableDataset, load_dataset(dataset_name, split=split, streaming=True))
        if shuffle:
            # Streaming shuffle with buffer; deterministic with seed
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        for item in dataset:
            assert isinstance(item, dict), f"Expected dict, got {type(item)}"
            yield item["text"]
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")


def chunk_text_by_chars(text: str, chunk_size: int) -> list[str]:
    """Split text into character-based chunks.

    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters

    Returns:
        List of text chunks
    """
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"
    assert isinstance(text, str), f"Expected str, got {type(text)}"

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk)

    return chunks


def chunk_text_by_tokens(
    text: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, chunk_size: int
) -> list[str]:
    """Split text into token-based chunks.

    Args:
        text: Input text to chunk
        tokenizer: HuggingFace tokenizer to use for tokenization
        chunk_size: Target size of each chunk in tokens

    Returns:
        List of text chunks, each approximately chunk_size tokens
    """
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"
    assert isinstance(text, str), f"Expected str, got {type(text)}"

    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():  # Skip empty chunks
            chunks.append(chunk_text)

    return chunks


def get_text_sequences(
    dataset_name: str,
    num_sequences: int,
    sequence_length: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
    split: str = "train",
    *,
    skip_sequences: int = 0,
    shuffle: bool = False,
    seed: int | None = None,
    buffer_size: int = 10_000,
) -> list[str]:
    """Get N sequences of specified length from streaming dataset.

    Args:
        dataset_name: HuggingFace dataset identifier
        num_sequences: Number of sequences to extract
        sequence_length: Target length of each sequence (in tokens if tokenizer provided, chars otherwise)
        tokenizer: Optional tokenizer for token-based chunking. If None, uses character-based chunking
        split: Dataset split to use
        skip_sequences: Skip this many sequences before collecting (for sharding)
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        buffer_size: Buffer size for streaming shuffle

    Returns:
        List of text sequences, each approximately sequence_length tokens/characters

    Raises:
        RuntimeError: If dataset fails to load or not enough sequences available
    """
    assert num_sequences > 0, f"num_sequences must be positive, got {num_sequences}"
    assert sequence_length > 0, f"sequence_length must be positive, got {sequence_length}"
    assert skip_sequences >= 0, f"skip_sequences must be non-negative, got {skip_sequences}"

    sequences = []
    current_text = ""
    produced = 0

    dataset_stream = load_streaming_dataset(
        dataset_name,
        split,
        shuffle=shuffle,
        seed=seed,
        buffer_size=buffer_size,
    )

    sequence_length = _effective_sequence_length(sequence_length, tokenizer)

    try:
        for text in dataset_stream:
            current_text += " " + text  # Add space between documents

            # Determine if we have enough content for sequence extraction
            if tokenizer is not None:
                # Token-based: check if we have enough tokens
                current_tokens = tokenizer.encode(current_text, add_special_tokens=False)
                content_length = len(current_tokens)
                min_needed = sequence_length
            else:
                # Character-based: check if we have enough characters
                content_length = len(current_text)
                min_needed = sequence_length

            # Extract sequences while we have enough content
            while content_length >= min_needed and (produced < skip_sequences + num_sequences):
                if tokenizer is not None:
                    # Token-based extraction
                    tokens = tokenizer.encode(current_text, add_special_tokens=False)
                    sequence_tokens = tokens[:sequence_length]
                    sequence = tokenizer.decode(sequence_tokens, skip_special_tokens=True)

                    # Remove extracted tokens plus some overlap for fresh content
                    overlap_size = sequence_length // 2
                    remaining_tokens = tokens[overlap_size:]
                    current_text = tokenizer.decode(remaining_tokens, skip_special_tokens=True)
                    current_tokens = remaining_tokens
                    content_length = len(current_tokens)
                else:
                    # Character-based extraction (original logic)
                    sequence = current_text[:sequence_length]
                    current_text = current_text[sequence_length // 2 :]  # 50% overlap
                    content_length = len(current_text)

                # Decide whether to keep or skip this sequence
                if produced >= skip_sequences and len(sequences) < num_sequences:
                    sequences.append(sequence)

                produced += 1

            if len(sequences) >= num_sequences:
                break

    except Exception as e:
        raise RuntimeError(f"Failed to extract sequences from {dataset_name}: {e}")

    assert len(sequences) == num_sequences, (
        f"Could only extract {len(sequences)} sequences, needed {num_sequences}"
    )

    return sequences
