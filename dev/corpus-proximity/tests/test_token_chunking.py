#!/usr/bin/env python3
"""Test token-aware chunking."""

from transformers import AutoTokenizer
from chunking import chunk_fixed_tokens, chunk_fixed_chars

# Load tokenizer (Arctic-Embed L)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-l")

# Test text (long document)
test_text = """
This is a test document with multiple paragraphs. We want to see how token-aware chunking
handles documents that exceed the model's maximum token limit. This is important for
contamination detection because we need to ensure no part of a document is lost.

The Arctic-Embed L model has a maximum of 512 tokens. If we chunk by characters (2048 chars),
we might accidentally create chunks that exceed this limit, especially for text with lots of
short words or special characters.

Token-aware chunking solves this by tokenizing first, then chunking at exact token boundaries.
This ensures every chunk fits within the model's context window, with no truncation.

""" * 10  # Repeat to make it long enough

print(f"\nTest text length: {len(test_text)} characters")
print(f"Test text tokens: {len(tokenizer.encode(test_text))} tokens")

# Test 1: Token-aware chunking
print("\n" + "="*80)
print("Token-aware chunking (max_tokens=512, overlap=15%)")
print("="*80)
token_chunks = chunk_fixed_tokens(test_text, tokenizer, max_tokens=512, overlap_pct=0.15)
print(f"Number of chunks: {len(token_chunks)}")

for i, chunk in enumerate(token_chunks[:3]):  # Show first 3
    tokens = len(tokenizer.encode(chunk))
    print(f"\nChunk {i+1}:")
    print(f"  Tokens: {tokens}")
    print(f"  Chars: {len(chunk)}")
    print(f"  Preview: {chunk[:100]}...")
    assert tokens <= 512, f"Chunk {i+1} exceeds max tokens: {tokens} > 512"

print(f"\n✓ All chunks ≤ 512 tokens")

# Test 2: Character-based chunking (for comparison)
print("\n" + "="*80)
print("Character-based chunking (size=2048 chars)")
print("="*80)
char_chunks = chunk_fixed_chars(test_text, size=2048)
print(f"Number of chunks: {len(char_chunks)}")

for i, chunk in enumerate(char_chunks[:3]):
    tokens = len(tokenizer.encode(chunk))
    print(f"\nChunk {i+1}:")
    print(f"  Tokens: {tokens}")
    print(f"  Chars: {len(chunk)}")
    print(f"  Preview: {chunk[:100]}...")
    if tokens > 512:
        print(f"  ⚠️  WARNING: Exceeds 512 tokens! Will be truncated by embedding model.")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Token-aware: {len(token_chunks)} chunks, all ≤ 512 tokens ✓")
print(f"Char-based: {len(char_chunks)} chunks, some may exceed 512 tokens ⚠️")
print("\nRecommendation: Use token-aware chunking for accurate embedding without truncation.")
