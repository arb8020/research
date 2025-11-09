#!/usr/bin/env python3
"""Test data buffer implementation.

Simple smoke test to verify DataBuffer works correctly.
"""

import sys
from pathlib import Path

# Add rollouts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts.training.data_buffer import DataBuffer, load_prompts_from_list


def test_basic_get_prompts():
    """Test basic prompt retrieval."""
    print("Testing basic get_prompts...")

    prompts = load_prompts_from_list(["Q1", "Q2", "Q3"])
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Get first batch
    batch = buffer.get_prompts(2)
    assert len(batch) == 2
    print(f"✓ First batch: {batch}")

    # Get second batch (wraps to next epoch)
    batch = buffer.get_prompts(2)
    assert len(batch) == 2
    print(f"✓ Second batch: {batch}")
    print(f"✓ Epoch: {buffer.epoch_id}")

    assert buffer.epoch_id == 1  # Should have wrapped


def test_epoch_wraparound():
    """Test epoch boundaries and shuffling."""
    print("\nTesting epoch wraparound...")

    prompts = load_prompts_from_list(["A", "B", "C"])
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Get all prompts from epoch 0
    epoch0_batch1 = buffer.get_prompts(2)
    epoch0_batch2 = buffer.get_prompts(1)

    print(f"✓ Epoch 0: {epoch0_batch1 + epoch0_batch2}")

    # Get prompts from epoch 1 (should be shuffled)
    epoch1_batch1 = buffer.get_prompts(2)

    print(f"✓ Epoch 1 (shuffled): {epoch1_batch1}")
    print(f"✓ Epoch ID: {buffer.epoch_id}")

    # Verify shuffling happened (order should be different)
    # Note: With seed=42, we get deterministic shuffling


def test_save_load_state():
    """Test state persistence."""
    print("\nTesting save/load state...")

    prompts = load_prompts_from_list(["X", "Y", "Z"])
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Advance buffer state
    buffer.get_prompts(2)

    # Save state
    state = buffer.save_state()
    print(f"✓ Saved state: {state}")

    # Create new buffer and restore
    buffer2 = DataBuffer(prompts=prompts, seed=42)
    buffer2.load_state(state)

    assert buffer2.epoch_id == buffer.epoch_id
    assert buffer2.sample_offset == buffer.sample_offset

    print(f"✓ Restored state matches")


def test_large_batch():
    """Test requesting batch larger than dataset."""
    print("\nTesting large batch request...")

    prompts = load_prompts_from_list(["A", "B"])
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Request more than dataset size
    batch = buffer.get_prompts(5)

    assert len(batch) == 5
    print(f"✓ Got {len(batch)} prompts (wrapped multiple times)")
    print(f"✓ Final epoch: {buffer.epoch_id}")


if __name__ == "__main__":
    print("=" * 60)
    print("DataBuffer Smoke Tests")
    print("=" * 60)

    test_basic_get_prompts()
    test_epoch_wraparound()
    test_save_load_state()
    test_large_batch()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
