#!/usr/bin/env python3
"""Chunking strategies for splitting text into smaller pieces."""

import re
from typing import List, Optional

# Optional imports - will use if available, fallback to regex if not
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load model lazily (on first use)
    _spacy_nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    _spacy_nlp = None

try:
    import nltk
    NLTK_AVAILABLE = True
    # Download punkt tokenizer if not already available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False


def chunk_text(text: str, strategy: str, chunk_size: int = 512) -> List[str]:
    """Split text into chunks based on strategy.

    Following Casey Muratori: Redundancy for convenience.
    Single entry point that dispatches to specific strategies.

    Args:
        text: Text to chunk
        strategy: "paragraph" | "fixed_chars" | "sentence_spacy" | "sentence_nltk"
        chunk_size: For fixed_chars strategy, target chunk size

    Returns:
        List of text chunks

    Example:
        chunks = chunk_text(text, "paragraph")
        chunks = chunk_text(text, "fixed_chars", chunk_size=512)
        chunks = chunk_text(text, "sentence_spacy")  # Use spaCy
        chunks = chunk_text(text, "sentence_nltk")  # Use NLTK
    """
    if strategy == "paragraph":
        return chunk_paragraph(text)
    elif strategy == "fixed_chars":
        return chunk_fixed_chars(text, chunk_size)
    elif strategy == "sentence_spacy":
        return chunk_sentence_spacy(text)
    elif strategy == "sentence_nltk":
        return chunk_sentence_nltk(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def chunk_paragraph(text: str) -> List[str]:
    """Split by double newline (paragraphs).

    Following Casey Muratori: Simplicity - just split and filter.

    Args:
        text: Text to chunk

    Returns:
        List of paragraph chunks (empty paragraphs filtered out)

    Example:
        text = "Para 1.\\n\\nPara 2.\\n\\nPara 3."
        chunks = chunk_paragraph(text)
        # ["Para 1.", "Para 2.", "Para 3."]
    """
    # Split on double newline
    paragraphs = text.split('\n\n')

    # Filter out empty/whitespace-only paragraphs
    chunks = [p.strip() for p in paragraphs if p.strip()]

    return chunks


def chunk_fixed_chars(text: str, size: int) -> List[str]:
    """Split into fixed character length chunks with overlap.

    Following Casey Muratori: Explicit about overlap strategy.
    Uses 20% overlap to avoid cutting important context.

    Args:
        text: Text to chunk
        size: Target chunk size in characters

    Returns:
        List of fixed-size chunks with overlap

    Example:
        text = "a" * 1000
        chunks = chunk_fixed_chars(text, size=512)
        # First chunk: chars 0-512
        # Second chunk: chars 410-922 (20% overlap = 102 chars)
        # etc.
    """
    assert size > 0, f"chunk size must be positive, got {size}"

    if len(text) <= size:
        return [text] if text.strip() else []

    chunks = []
    overlap = int(size * 0.2)  # 20% overlap
    stride = size - overlap

    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()

        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

        # Move forward by stride
        start += stride

        # Stop if we've reached the end
        if end == len(text):
            break

    return chunks


def chunk_sentence_spacy(text: str) -> List[str]:
    """Split by sentence boundaries using spaCy.

    Most accurate method - uses trained models and linguistic analysis.
    Handles complex cases like abbreviations, decimals, etc.

    Args:
        text: Text to chunk

    Returns:
        List of sentence chunks

    Raises:
        ImportError: If spaCy is not installed or model is not available

    Example:
        text = "Dr. Smith went home. He bought 3.14 kg of apples."
        chunks = chunk_sentence_spacy(text)
        # ["Dr. Smith went home.", "He bought 3.14 kg of apples."]
    """
    if not SPACY_AVAILABLE:
        raise ImportError("spaCy is not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")

    # Lazy load spaCy model
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            # Model not available - provide clear error message
            raise ImportError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Install with: python -m spacy download en_core_web_sm"
            ) from e

    doc = _spacy_nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Fallback: if no sentences found, return whole text
    if not sentences and text.strip():
        return [text.strip()]

    return sentences


def chunk_sentence_nltk(text: str) -> List[str]:
    """Split by sentence boundaries using NLTK.

    Good accuracy - uses trained Punkt tokenizer.
    Handles common abbreviations and sentence boundaries.

    Args:
        text: Text to chunk

    Returns:
        List of sentence chunks

    Example:
        text = "Dr. Smith went home. Mrs. Jones stayed."
        chunks = chunk_sentence_nltk(text)
        # ["Dr. Smith went home.", "Mrs. Jones stayed."]
    """
    if not NLTK_AVAILABLE:
        raise ImportError("NLTK is not installed. Install with: pip install nltk")

    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Fallback: if no sentences found, return whole text
    if not sentences and text.strip():
        return [text.strip()]

    return sentences




def main():
    """Test all chunking strategies."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    test_text = """This is the first paragraph. It has multiple sentences.

Some have punctuation! Others have questions? And some are just normal.

This is the second paragraph. Dr. Smith went to the store. Mrs. Jones did too.

Third paragraph here. Mr. Johnson calculated that 3.14 is pi."""

    logging.info("="*80)
    logging.info("Testing chunking strategies")
    logging.info("="*80)
    logging.info(f"\nAvailable: spaCy={SPACY_AVAILABLE}, NLTK={NLTK_AVAILABLE}")

    logging.info("\n1. Paragraph chunking:")
    para_chunks = chunk_paragraph(test_text)
    for i, chunk in enumerate(para_chunks):
        logging.info(f"  Chunk {i}: {chunk[:60]}...")

    logging.info("\n2. Fixed chars chunking (size=100):")
    fixed_chunks = chunk_fixed_chars(test_text, size=100)
    for i, chunk in enumerate(fixed_chunks):
        logging.info(f"  Chunk {i} (len={len(chunk)}): {chunk[:60]}...")

    # Test spaCy if available
    if SPACY_AVAILABLE:
        logging.info("\n3. Sentence chunking (spaCy):")
        try:
            spacy_chunks = chunk_sentence_spacy(test_text)
            for i, chunk in enumerate(spacy_chunks):
                logging.info(f"  Chunk {i}: {chunk}")
        except Exception as e:
            logging.error(f"  spaCy error: {e}")

    # Test NLTK if available
    if NLTK_AVAILABLE:
        logging.info("\n4. Sentence chunking (NLTK):")
        try:
            nltk_chunks = chunk_sentence_nltk(test_text)
            for i, chunk in enumerate(nltk_chunks):
                logging.info(f"  Chunk {i}: {chunk}")
        except Exception as e:
            logging.error(f"  NLTK error: {e}")

    logging.info("\n" + "="*80)
    logging.info("Summary:")
    logging.info(f"  Paragraphs: {len(para_chunks)} chunks")
    logging.info(f"  Fixed chars: {len(fixed_chunks)} chunks")
    if SPACY_AVAILABLE:
        try:
            logging.info(f"  Sentences (spaCy): {len(chunk_sentence_spacy(test_text))} chunks")
        except:
            logging.info(f"  Sentences (spaCy): Not available (model not installed)")
    if NLTK_AVAILABLE:
        try:
            logging.info(f"  Sentences (NLTK): {len(chunk_sentence_nltk(test_text))} chunks")
        except:
            logging.info(f"  Sentences (NLTK): Error")
    logging.info("="*80)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
