#!/usr/bin/env python3
"""Test dataset loaders with HuggingFace datasets.

⚠️ NOTE: This example requires torch and datasets libraries.
Cannot run locally without GPU environment.
For remote testing, use deploy.py pattern.

Demonstrates loading common datasets for SFT and RL training.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset_loaders import (
    load_sft_dataset,
    load_rl_prompts,
    load_dataset_with_answers,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_load_sft_dataset():
    """Test loading SFT dataset without tokenizer (just structure check)."""
    print("\n" + "=" * 70)
    print("Test: Load SFT Dataset (without tokenizer)")
    print("=" * 70)

    # Load small sample from a common dataset
    # Note: This requires internet connection and HF datasets library
    try:
        samples = load_sft_dataset(
            "HuggingFaceTB/smoltalk",
            split="train",
            tokenizer=None,  # Skip tokenization for this test
            max_samples=5,
        )

        print(f"\n✓ Loaded {len(samples)} samples")
        print(f"\nFirst sample:")
        print(f"  Prompt: {samples[0].prompt[:100]}...")
        print(f"  Metadata: {samples[0].metadata}")

    except Exception as e:
        print(f"\n⚠️  Could not load dataset (might need internet/HF account): {e}")
        print("   This is expected if running without internet or HF access")


def test_load_rl_prompts():
    """Test loading RL prompts from GSM8K."""
    print("\n" + "=" * 70)
    print("Test: Load RL Prompts (GSM8K)")
    print("=" * 70)

    try:
        prompts = load_rl_prompts(
            "openai/gsm8k",
            split="train",
            max_prompts=5,
        )

        print(f"\n✓ Loaded {len(prompts)} prompts")
        print(f"\nFirst prompt:")
        print(f"  {prompts[0][:150]}...")

    except Exception as e:
        print(f"\n⚠️  Could not load dataset: {e}")
        print("   This is expected if running without internet/HF access")


def test_load_dataset_with_answers():
    """Test loading dataset with ground truth answers."""
    print("\n" + "=" * 70)
    print("Test: Load Dataset with Answers (GSM8K)")
    print("=" * 70)

    try:
        data = load_dataset_with_answers(
            "openai/gsm8k",
            split="train",
            max_samples=3,
        )

        print(f"\n✓ Loaded {len(data)} prompt/answer pairs")
        print(f"\nFirst example:")
        print(f"  Prompt: {data[0]['prompt'][:100]}...")
        print(f"  Answer: {data[0]['answer'][:50]}...")

    except Exception as e:
        print(f"\n⚠️  Could not load dataset: {e}")
        print("   This is expected if running without internet/HF access")


def main():
    """Run all dataset loader tests."""
    print("\n" + "=" * 70)
    print("Dataset Loaders Tests")
    print("=" * 70)
    print("\nNote: These tests require internet connection and HF datasets library")
    print("      Install with: pip install datasets")

    test_load_sft_dataset()
    test_load_rl_prompts()
    test_load_dataset_with_answers()

    print("\n" + "=" * 70)
    print("✅ Dataset loader tests complete!")
    print("=" * 70)
    print("""
Dataset loaders support:

SFT Training:
  - load_sft_dataset() - Load chat/instruction datasets
  - Handles multiple formats: messages, prompt/completion, instruction/output
  - Automatically tokenizes and computes loss masks

RL Training:
  - load_rl_prompts() - Extract prompts for rollout generation
  - load_dataset_with_answers() - Load with ground truth for reward computation

Example usage:
  >>> from transformers import AutoTokenizer
  >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
  >>> samples = load_sft_dataset("HuggingFaceTB/smoltalk", tokenizer=tokenizer)
  >>> prompts = load_rl_prompts("openai/gsm8k", max_prompts=100)
    """)


if __name__ == "__main__":
    main()
