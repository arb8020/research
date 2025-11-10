#!/usr/bin/env python3
"""Test rollout generation with pure functions.

Smoke tests for the full data pipeline: DataBuffer → rollout function → RolloutBatch.

Migrated from test_rollout_manager.py to use pure function generator pattern.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_buffer import DataBuffer, load_prompts_from_list
from training.types import Sample, RolloutConfig
from training.rollout_generation import (
    generate_rollout_batches,
    convert_to_batch,
    apply_sample_transforms,
)


# Mock rollout function for testing
def mock_sft_rollout(prompts, **kwargs):
    """Simple rollout that converts prompts to samples."""
    samples = []
    for i, prompt in enumerate(prompts):
        sample = Sample(
            prompt=prompt,
            response=f"Response {i}",
            tokens=[1, 2, 3, i],
            loss_mask=[0.0, 1.0, 1.0, 1.0],
            reward=0.0,
        )
        samples.append(sample)
    return samples


def test_convert_to_batch():
    """Test convert_to_batch helper."""
    print("Testing convert_to_batch...")

    samples = [
        Sample(
            prompt="Q1",
            response="A1",
            tokens=[1, 2, 3],
            loss_mask=[0.0, 1.0, 1.0],
        ),
        Sample(
            prompt="Q2",
            response="A2",
            tokens=[4, 5, 6],
            loss_mask=[0.0, 1.0, 1.0],
        ),
    ]

    batch = convert_to_batch(samples, epoch_id=0, step_id=42)

    assert len(batch.tokens) == 2
    assert batch.tokens[0] == [1, 2, 3]
    assert batch.tokens[1] == [4, 5, 6]
    assert batch.metadata["epoch_id"] == 0
    assert batch.metadata["step_id"] == 42
    assert batch.metadata["batch_size"] == 2

    print(f"✓ Converted {len(samples)} samples to batch")
    print(f"✓ Metadata: {batch.metadata}")


def test_rollout_generation_basic():
    """Test basic rollout batch generation."""
    print("\nTesting rollout batch generation basic iteration...")

    # Setup
    prompts = load_prompts_from_list(["Q1", "Q2", "Q3", "Q4"])
    buffer = DataBuffer(prompts=prompts, seed=42)
    config = RolloutConfig(
        batch_size=2,
        generate_fn=mock_sft_rollout,
    )

    batches = generate_rollout_batches(buffer, config)

    # Get first batch
    batch1 = next(batches)
    assert len(batch1.tokens) == 2
    assert len(batch1.metadata["prompts"]) == 2
    print(f"✓ Batch 1: {batch1.metadata['prompts']}")

    # Get second batch (should wrap to next epoch)
    batch2 = next(batches)
    assert len(batch2.tokens) == 2
    print(f"✓ Batch 2: {batch2.metadata['prompts']}")

    # Check epoch wrapped
    assert buffer.epoch_id == 1
    print(f"✓ Epoch wrapped: {buffer.epoch_id}")


def test_rollout_generation_with_kwargs():
    """Test passing kwargs to rollout function."""
    print("\nTesting rollout generation with kwargs...")

    def rollout_with_kwargs(prompts, tokenizer_name=None, **kwargs):
        """Rollout that uses kwargs."""
        assert tokenizer_name == "test-tokenizer"
        return mock_sft_rollout(prompts)

    prompts = load_prompts_from_list(["Q1", "Q2"])
    buffer = DataBuffer(prompts=prompts, seed=42)
    config = RolloutConfig(
        batch_size=2,
        generate_fn=rollout_with_kwargs,
    )

    batches = generate_rollout_batches(
        buffer,
        config,
        tokenizer_name="test-tokenizer",  # Passed to rollout function
    )

    batch = next(batches)
    assert len(batch.tokens) == 2
    print(f"✓ Rollout function received kwargs correctly")


def test_rollout_generation_with_filter():
    """Test filter function."""
    print("\nTesting rollout generation with filter...")

    def filter_fn(samples):
        """Filter out samples with short responses."""
        return [s for s in samples if len(s.response) > 10]

    prompts = load_prompts_from_list(["Q1", "Q2", "Q3", "Q4"])
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Rollout that generates variable-length responses
    def variable_rollout(prompts, **kwargs):
        samples = []
        for i, prompt in enumerate(prompts):
            response = "Short" if i % 2 == 0 else "Long response here"
            sample = Sample(
                prompt=prompt,
                response=response,
                tokens=[1, 2, 3],
                loss_mask=[0.0, 1.0, 1.0],
            )
            samples.append(sample)
        return samples

    config = RolloutConfig(
        batch_size=4,
        generate_fn=variable_rollout,
        filter_fn=filter_fn,
    )

    batches = generate_rollout_batches(buffer, config)
    batch = next(batches)

    # Should only have long responses
    assert all(len(r) > 10 for r in batch.metadata["responses"])
    print(f"✓ Filtered to {len(batch.metadata['responses'])} samples")


def test_rollout_generation_iterator_protocol():
    """Test using generator in for loop."""
    print("\nTesting rollout generation iterator protocol...")

    prompts = load_prompts_from_list(["Q1", "Q2", "Q3"])
    buffer = DataBuffer(prompts=prompts, seed=42)
    config = RolloutConfig(
        batch_size=1,
        generate_fn=mock_sft_rollout,
    )

    batches = generate_rollout_batches(buffer, config)

    # Iterate a few times
    batch_list = []
    for i, batch in enumerate(batches):
        batch_list.append(batch)
        if i >= 4:  # Get 5 batches
            break

    assert len(batch_list) == 5
    print(f"✓ Iterated {len(batch_list)} batches")
    print(f"✓ Epochs completed: {buffer.epoch_id}")


def test_metadata_aggregation():
    """Test metadata from samples is aggregated."""
    print("\nTesting metadata aggregation...")

    def rollout_with_metadata(prompts, **kwargs):
        samples = []
        for i, prompt in enumerate(prompts):
            sample = Sample(
                prompt=prompt,
                response=f"R{i}",
                tokens=[1, 2, 3],
                loss_mask=[0.0, 1.0, 1.0],
                metadata={"score": i * 10, "source": f"dataset_{i}"},
            )
            samples.append(sample)
        return samples

    prompts = load_prompts_from_list(["Q1", "Q2"])
    buffer = DataBuffer(prompts=prompts, seed=42)
    config = RolloutConfig(
        batch_size=2,
        generate_fn=rollout_with_metadata,
    )

    batches = generate_rollout_batches(buffer, config)
    batch = next(batches)

    assert "sample_score" in batch.metadata
    assert batch.metadata["sample_score"] == [0, 10]
    assert "sample_source" in batch.metadata
    print(f"✓ Metadata aggregated: {list(batch.metadata.keys())}")


def test_apply_sample_transforms():
    """Test apply_sample_transforms helper function."""
    print("\nTesting apply_sample_transforms...")

    def filter_fn(samples):
        return [s for s in samples if s.reward > 0.5]

    samples = [
        Sample(prompt="Q1", response="A1", tokens=[1], loss_mask=[1.0], reward=0.3),
        Sample(prompt="Q2", response="A2", tokens=[2], loss_mask=[1.0], reward=0.8),
        Sample(prompt="Q3", response="A3", tokens=[3], loss_mask=[1.0], reward=0.9),
    ]

    config = RolloutConfig(
        batch_size=2,
        generate_fn=lambda x: x,
        filter_fn=filter_fn,
    )

    filtered = apply_sample_transforms(samples, config)

    assert len(filtered) == 2
    assert all(s.reward > 0.5 for s in filtered)
    print(f"✓ Filtered {len(samples)} samples to {len(filtered)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Rollout Generation Tests (Pure Function Pattern)")
    print("=" * 60)

    test_convert_to_batch()
    test_rollout_generation_basic()
    test_rollout_generation_with_kwargs()
    test_rollout_generation_with_filter()
    test_rollout_generation_iterator_protocol()
    test_metadata_aggregation()
    test_apply_sample_transforms()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
