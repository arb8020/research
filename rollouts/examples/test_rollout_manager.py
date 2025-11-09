#!/usr/bin/env python3
"""Test RolloutManager orchestration.

Smoke tests for the full data pipeline: DataBuffer → rollout function → RolloutBatch.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_buffer import DataBuffer, load_prompts_from_list
from training.types import Sample, RolloutConfig
from training.rollout_manager import RolloutManager, convert_to_batch


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


def test_rollout_manager_basic():
    """Test basic RolloutManager iteration."""
    print("\nTesting RolloutManager basic iteration...")

    # Setup
    prompts = load_prompts_from_list(["Q1", "Q2", "Q3", "Q4"])
    buffer = DataBuffer(prompts=prompts, seed=42)
    config = RolloutConfig(
        batch_size=2,
        generate_fn=mock_sft_rollout,
    )

    manager = RolloutManager(buffer, config)

    # Get first batch
    batch1 = next(manager)
    assert len(batch1.tokens) == 2
    assert len(batch1.metadata["prompts"]) == 2
    print(f"✓ Batch 1: {batch1.metadata['prompts']}")

    # Get second batch (should wrap to next epoch)
    batch2 = next(manager)
    assert len(batch2.tokens) == 2
    print(f"✓ Batch 2: {batch2.metadata['prompts']}")

    # Check epoch wrapped
    assert buffer.epoch_id == 1
    print(f"✓ Epoch wrapped: {buffer.epoch_id}")


def test_rollout_manager_with_kwargs():
    """Test passing kwargs to rollout function."""
    print("\nTesting RolloutManager with kwargs...")

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

    manager = RolloutManager(
        buffer,
        config,
        tokenizer_name="test-tokenizer",  # Passed to rollout function
    )

    batch = next(manager)
    assert len(batch.tokens) == 2
    print(f"✓ Rollout function received kwargs correctly")


def test_rollout_manager_with_filter():
    """Test filter function."""
    print("\nTesting RolloutManager with filter...")

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

    manager = RolloutManager(buffer, config)
    batch = next(manager)

    # Should only have long responses
    assert all(len(r) > 10 for r in batch.metadata["responses"])
    print(f"✓ Filtered to {len(batch.metadata['responses'])} samples")


def test_rollout_manager_state_dict():
    """Test state save/load."""
    print("\nTesting RolloutManager state dict...")

    prompts = load_prompts_from_list(["Q1", "Q2", "Q3"])
    buffer = DataBuffer(prompts=prompts, seed=42)
    config = RolloutConfig(
        batch_size=2,
        generate_fn=mock_sft_rollout,
    )

    manager1 = RolloutManager(buffer, config)

    # Advance state
    next(manager1)
    next(manager1)

    # Save state
    state = manager1.state_dict()
    print(f"✓ Saved state: step={state['step_count']}")

    # Create new manager and restore
    buffer2 = DataBuffer(prompts=prompts, seed=42)
    manager2 = RolloutManager(buffer2, config)
    manager2.load_state_dict(state)

    assert manager2._step_count == manager1._step_count
    assert manager2.data_buffer.epoch_id == manager1.data_buffer.epoch_id
    print(f"✓ Restored state matches")


def test_rollout_manager_iterator_protocol():
    """Test using manager in for loop."""
    print("\nTesting RolloutManager iterator protocol...")

    prompts = load_prompts_from_list(["Q1", "Q2", "Q3"])
    buffer = DataBuffer(prompts=prompts, seed=42)
    config = RolloutConfig(
        batch_size=1,
        generate_fn=mock_sft_rollout,
    )

    manager = RolloutManager(buffer, config)

    # Iterate a few times
    batches = []
    for i, batch in enumerate(manager):
        batches.append(batch)
        if i >= 4:  # Get 5 batches
            break

    assert len(batches) == 5
    print(f"✓ Iterated {len(batches)} batches")
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

    manager = RolloutManager(buffer, config)
    batch = next(manager)

    assert "sample_score" in batch.metadata
    assert batch.metadata["sample_score"] == [0, 10]
    assert "sample_source" in batch.metadata
    print(f"✓ Metadata aggregated: {list(batch.metadata.keys())}")


if __name__ == "__main__":
    print("=" * 60)
    print("RolloutManager Tests")
    print("=" * 60)

    test_convert_to_batch()
    test_rollout_manager_basic()
    test_rollout_manager_with_kwargs()
    test_rollout_manager_with_filter()
    test_rollout_manager_state_dict()
    test_rollout_manager_iterator_protocol()
    test_metadata_aggregation()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
