#!/usr/bin/env python3
"""Test SFT sample preparation.

Smoke tests for tokenization and loss masking without requiring Megatron.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.sft import (
    compute_loss_mask,
    tokenize_conversation,
    prepare_sft_sample,
    example_sft_rollout_fn,
)


# Mock tokenizer for testing (avoids HuggingFace dependency in test)
class MockTokenizer:
    """Minimal tokenizer mock for testing."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        """Simple template: <role>: content"""
        parts = []
        for msg in messages:
            parts.append(f"<{msg['role']}>: {msg['content']}")
        return " ".join(parts)

    def encode(
        self, text: str, max_length: int = 2048, truncation: bool = True
    ) -> list[int]:
        """Encode as character codes (simple mock)."""
        tokens = [ord(c) for c in text]
        if truncation:
            tokens = tokens[:max_length]
        return tokens


def test_compute_loss_mask():
    """Test loss mask computation."""
    print("Testing compute_loss_mask...")

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    user_spans = [(0, 3), (6, 8)]

    mask = compute_loss_mask(tokens, user_spans)

    expected = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert mask == expected, f"Expected {expected}, got {mask}"

    print(f"✓ Loss mask: {mask}")
    print(f"✓ User spans masked correctly")


def test_tokenize_conversation():
    """Test conversation tokenization."""
    print("\nTesting tokenize_conversation...")

    tokenizer = MockTokenizer()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "How are you?"},
    ]

    tokens, user_spans = tokenize_conversation(messages, tokenizer)

    print(f"✓ Tokens: {len(tokens)} tokens")
    print(f"✓ User spans: {user_spans}")

    # Verify we got some tokens
    assert len(tokens) > 0
    # Verify we tracked user messages
    assert len(user_spans) == 2


def test_prepare_sft_sample_simple():
    """Test simple prompt/response."""
    print("\nTesting prepare_sft_sample (simple)...")

    tokenizer = MockTokenizer()
    sample = prepare_sft_sample(
        prompt="What is 2+2?",
        response="4",
        tokenizer=tokenizer,
    )

    assert sample.prompt == "What is 2+2?"
    assert sample.response == "4"
    assert len(sample.tokens) > 0
    assert len(sample.loss_mask) == len(sample.tokens)

    print(f"✓ Prompt: {sample.prompt}")
    print(f"✓ Response: {sample.response}")
    print(f"✓ Tokens: {len(sample.tokens)}")
    print(f"✓ Loss mask: {len(sample.loss_mask)} values")


def test_prepare_sft_sample_multiturn():
    """Test multi-turn conversation."""
    print("\nTesting prepare_sft_sample (multi-turn)...")

    tokenizer = MockTokenizer()
    sample = prepare_sft_sample(
        prompt=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ],
        response="I'm doing well!",
        tokenizer=tokenizer,
    )

    assert isinstance(sample.prompt, list)
    assert sample.response == "I'm doing well!"
    assert len(sample.tokens) > 0
    assert len(sample.loss_mask) == len(sample.tokens)

    # Verify some tokens are masked (user messages)
    assert 0.0 in sample.loss_mask
    # Verify some tokens are unmasked (assistant messages)
    assert 1.0 in sample.loss_mask

    print(f"✓ Multi-turn prompt with {len(sample.prompt)} messages")
    print(f"✓ Response: {sample.response}")
    print(f"✓ Tokens: {len(sample.tokens)}")
    print(f"✓ Masked tokens: {sample.loss_mask.count(0.0)}")
    print(f"✓ Unmasked tokens: {sample.loss_mask.count(1.0)}")


def test_example_rollout_fn():
    """Test example SFT rollout function."""
    print("\nTesting example_sft_rollout_fn...")

    tokenizer = MockTokenizer()

    # Mock dataset
    dataset = [
        {"id": "q1", "prompt": "What is 2+2?", "response": "4"},
        {"id": "q2", "prompt": "What is 3+3?", "response": "6"},
    ]

    # Prompts with IDs
    prompts = [{"id": "q1"}, {"id": "q2"}]

    samples = example_sft_rollout_fn(
        prompts=prompts,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    assert len(samples) == 2
    assert samples[0].response == "4"
    assert samples[1].response == "6"

    print(f"✓ Generated {len(samples)} samples")
    print(f"✓ Sample 1 response: {samples[0].response}")
    print(f"✓ Sample 2 response: {samples[1].response}")


if __name__ == "__main__":
    print("=" * 60)
    print("SFT Sample Preparation Tests")
    print("=" * 60)

    test_compute_loss_mask()
    test_tokenize_conversation()
    test_prepare_sft_sample_simple()
    test_prepare_sft_sample_multiturn()
    test_example_rollout_fn()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
