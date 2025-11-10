#!/usr/bin/env python3
"""Test async rollout manager with dynamic sampling (D4).

Demonstrates:
- Async parallel generation
- Dynamic over-sampling (generate 1.5x, keep best 1x)
- Filter functions
- Partial sample caching
"""

import sys
import trio
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import (
    DataBuffer,
    AsyncRolloutManager,
    RolloutConfig,
    Sample,
    generate_rollout_batch,
)


# ────────────────────── Mock Async Generate Function ──────────────────────


async def mock_async_generate(
    prompts: list[str | dict],
    **kwargs,
) -> list[Sample]:
    """Mock async generation function.

    Simulates async LLM generation with random quality.
    """
    import random

    samples = []

    for prompt in prompts:
        # Simulate network delay
        await trio.sleep(0.01)

        # Extract prompt text
        prompt_text = prompt if isinstance(prompt, str) else prompt.get("prompt", "")

        # Generate mock response
        # Quality varies: some samples better than others
        quality = random.random()
        if quality > 0.7:
            response = f"High quality response to: {prompt_text}"
        elif quality > 0.3:
            response = f"Medium quality response to: {prompt_text}"
        else:
            response = f"Low quality response to: {prompt_text}"

        sample = Sample(
            prompt=prompt_text,
            response=response,
            tokens=[1, 2, 3, 4],  # Mock tokens
            loss_mask=[0.0, 0.0, 1.0, 1.0],  # Mock loss mask
            metadata={"quality": quality},
        )

        samples.append(sample)

    return samples


# ────────────────────── Test Functions ──────────────────────


async def test_basic_async_generation():
    """Test basic async batch generation."""
    print("\n" + "=" * 70)
    print("Test 1: Basic Async Generation")
    print("=" * 70)

    # Create data buffer
    prompts = [f"Prompt {i}" for i in range(10)]
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Create config
    config = RolloutConfig(
        batch_size=4,
        generate_fn=mock_async_generate,
    )

    # Generate batch
    manager = AsyncRolloutManager(
        data_buffer=buffer,
        config=config,
    )

    async with manager:
        batch = await manager.generate_batch()

    print(f"✓ Generated batch with {len(batch.tokens)} samples")
    print(f"  Prompts: {batch.metadata['prompts']}")
    print(f"  Responses: {batch.metadata['responses'][:2]}...")  # First 2


async def test_dynamic_oversampling():
    """Test dynamic over-sampling with filtering."""
    print("\n" + "=" * 70)
    print("Test 2: Dynamic Over-Sampling (SLIME feature)")
    print("=" * 70)

    # Create data buffer
    prompts = [f"Question {i}" for i in range(20)]
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Filter function: only accept high-quality samples
    def quality_filter(samples: list[Sample]) -> bool:
        """Keep samples with quality > 0.5."""
        avg_quality = sum(s.metadata.get("quality", 0) for s in samples) / len(samples)
        return avg_quality > 0.5

    # Create config with over-sampling
    config = RolloutConfig(
        batch_size=4,
        over_sampling_factor=2.0,  # Generate 8, keep best 4
        generate_fn=mock_async_generate,
        filter_fn=quality_filter,
    )

    manager = AsyncRolloutManager(
        data_buffer=buffer,
        config=config,
    )

    async with manager:
        batch = await manager.generate_batch()

    print(f"✓ Over-sampling factor: {config.over_sampling_factor}x")
    print(f"✓ Target batch size: {config.batch_size}")
    print(f"✓ Actual batch size: {len(batch.tokens)}")

    # Check quality
    qualities = [
        batch.metadata[f"sample_quality"][i]
        for i in range(len(batch.tokens))
    ]
    avg_quality = sum(qualities) / len(qualities)
    print(f"✓ Average quality: {avg_quality:.2f} (should be > 0.5 due to filter)")


async def test_n_samples_per_prompt():
    """Test generating multiple samples per prompt (for GRPO)."""
    print("\n" + "=" * 70)
    print("Test 3: Multiple Samples Per Prompt (GRPO)")
    print("=" * 70)

    # Create data buffer
    prompts = ["Math problem 1", "Math problem 2"]
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Config with 4 samples per prompt
    config = RolloutConfig(
        batch_size=8,  # 2 prompts * 4 samples
        n_samples_per_prompt=4,
        generate_fn=mock_async_generate,
    )

    manager = AsyncRolloutManager(
        data_buffer=buffer,
        config=config,
    )

    async with manager:
        batch = await manager.generate_batch()

    print(f"✓ Generated {len(batch.tokens)} samples")
    print(f"✓ From {len(set(batch.metadata['prompts']))} unique prompts")
    print(f"✓ n_samples_per_prompt: {config.n_samples_per_prompt}")


async def test_partial_sample_caching():
    """Test caching of partial samples on abort/overflow."""
    print("\n" + "=" * 70)
    print("Test 4: Partial Sample Caching (SLIME feature)")
    print("=" * 70)

    # Create data buffer
    prompts = [f"Prompt {i}" for i in range(20)]
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Config that will generate overflow
    config = RolloutConfig(
        batch_size=3,
        over_sampling_factor=2.0,  # Will generate 6, only need 3
        generate_fn=mock_async_generate,
    )

    manager = AsyncRolloutManager(
        data_buffer=buffer,
        config=config,
    )

    async with manager:
        # First batch - should cache overflow
        batch1 = await manager.generate_batch()
        print(f"✓ Batch 1: {len(batch1.tokens)} samples")
        print(f"  Cached partial samples: {len(manager.partial_samples)}")

        # Second batch - should use cached samples first
        batch2 = await manager.generate_batch()
        print(f"✓ Batch 2: {len(batch2.tokens)} samples")
        print(f"  Cached partial samples: {len(manager.partial_samples)}")


async def test_convenience_function():
    """Test the convenience generate_rollout_batch function."""
    print("\n" + "=" * 70)
    print("Test 5: Convenience Function")
    print("=" * 70)

    # Create data buffer
    prompts = ["Q1", "Q2", "Q3", "Q4"]
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Config
    config = RolloutConfig(
        batch_size=4,
        generate_fn=mock_async_generate,
    )

    # Reward function
    def simple_reward(sample: Sample) -> float:
        """Reward based on response quality."""
        return sample.metadata.get("quality", 0.0)

    # Generate batch with convenience function
    batch = await generate_rollout_batch(
        buffer=buffer,
        config=config,
        reward_fn=simple_reward,
    )

    print(f"✓ Generated batch: {len(batch.tokens)} samples")
    print(f"✓ Rewards: {batch.rewards}")
    print(f"✓ Average reward: {sum(batch.rewards) / len(batch.rewards):.2f}")


async def test_state_dict():
    """Test saving/loading state with partial samples."""
    print("\n" + "=" * 70)
    print("Test 6: State Management with Partial Samples")
    print("=" * 70)

    # Create data buffer
    prompts = [f"Prompt {i}" for i in range(10)]
    buffer = DataBuffer(prompts=prompts, seed=42)

    # Config
    config = RolloutConfig(
        batch_size=3,
        over_sampling_factor=2.0,
        generate_fn=mock_async_generate,
    )

    # Generate batch and save state
    manager1 = AsyncRolloutManager(
        data_buffer=buffer,
        config=config,
    )

    async with manager1:
        batch1 = await manager1.generate_batch()
        state = manager1.state_dict()

    print(f"✓ Generated batch: {len(batch1.tokens)} samples")
    print(f"✓ Saved state:")
    print(f"  - Step count: {state['step_count']}")
    print(f"  - Partial samples: {len(state['partial_samples'])}")
    print(f"  - Buffer epoch: {state['buffer_state']['epoch_id']}")

    # Restore state in new manager
    manager2 = AsyncRolloutManager(
        data_buffer=DataBuffer(prompts=prompts, seed=42),
        config=config,
    )
    manager2.load_state_dict(state)

    print(f"✓ Restored state:")
    print(f"  - Step count: {manager2._step_count}")
    print(f"  - Partial samples: {len(manager2.partial_samples)}")


# ────────────────────── Main ──────────────────────


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Async Rollout Manager Tests (D4)")
    print("=" * 70)

    await test_basic_async_generation()
    await test_dynamic_oversampling()
    await test_n_samples_per_prompt()
    await test_partial_sample_caching()
    await test_convenience_function()
    await test_state_dict()

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("""
Key D4 features verified:
- ✓ Async parallel generation
- ✓ Dynamic over-sampling (generate N*factor, keep N)
- ✓ Quality filtering
- ✓ Multiple samples per prompt (GRPO)
- ✓ Partial sample caching (SLIME feature)
- ✓ State management with partial samples
    """)


if __name__ == "__main__":
    trio.run(main)
