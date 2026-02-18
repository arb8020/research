"""Tests for DeterministicLoader.

Key properties to verify:
1. Determinism - same seed = same data
2. Multi-shard - reads across shard boundaries correctly
3. SWRR mixing - correct weight distribution
4. State resume - exact position recovery
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rollouts.pretrain.dataloader import (
    Cursor,
    DeterministicLoader,
    ShardedDataset,
    build_loader,
)


@pytest.fixture
def temp_shards():
    """Create temporary .npy shards for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)

        # Create two shards with sequential tokens
        shard1 = np.arange(0, 1000, dtype=np.int64)
        shard2 = np.arange(1000, 2000, dtype=np.int64)

        np.save(path / "shard1.npy", shard1)
        np.save(path / "shard2.npy", shard2)

        yield path, [str(path / "shard1.npy"), str(path / "shard2.npy")]


class TestShardedDataset:
    """Tests for multi-shard reading."""

    def test_reads_across_shard_boundary(self, temp_shards):
        """Should seamlessly read across shard boundaries."""
        _, paths = temp_shards
        ds = ShardedDataset(paths)

        # Read window that crosses from shard1 (0-999) to shard2 (1000-1999)
        cursor = Cursor(file_idx=0, pos_in_file=990)
        tokens, new_cursor = ds.next_window(cursor, length=20)

        # Should contain 990-999 from shard1 and 1000-1009 from shard2
        expected = np.arange(990, 1010, dtype=np.int64)
        assert np.array_equal(tokens, expected), f"Got {tokens}, expected {expected}"

    def test_cursor_advances_correctly(self, temp_shards):
        """Cursor should track position across reads."""
        _, paths = temp_shards
        ds = ShardedDataset(paths)

        cursor = Cursor()
        tokens1, cursor = ds.next_window(cursor, length=100)
        tokens2, cursor = ds.next_window(cursor, length=100)

        # Second batch should start where first ended
        assert tokens1[-1] == 99
        assert tokens2[0] == 100

    def test_wraps_at_end(self, temp_shards):
        """Should wrap around when reaching end of all shards."""
        _, paths = temp_shards
        ds = ShardedDataset(paths)

        # Start near end of shard2
        cursor = Cursor(file_idx=1, pos_in_file=990)
        tokens, new_cursor = ds.next_window(cursor, length=20)

        # Should wrap to beginning
        assert new_cursor.wrap_count == 1
        assert new_cursor.file_idx == 0


class TestDeterministicLoader:
    """Tests for the full data loader."""

    def test_deterministic_output(self, temp_shards):
        """Same config should produce identical batches."""
        path, _ = temp_shards

        loader1 = build_loader(path, seq_len=32, batch_size=2, device="cpu")
        loader2 = build_loader(path, seq_len=32, batch_size=2, device="cpu")

        batch1a, _ = loader1.next()
        batch1b, _ = loader1.next()
        batch2a, _ = loader2.next()
        batch2b, _ = loader2.next()

        assert torch.equal(batch1a, batch2a), "First batches differ"
        assert torch.equal(batch1b, batch2b), "Second batches differ"

    def test_state_dict_resume(self, temp_shards):
        """Should resume exactly from saved state."""
        path, _ = temp_shards

        loader1 = build_loader(path, seq_len=32, batch_size=2, device="cpu")

        # Advance a few steps
        for _ in range(5):
            loader1.next()

        # Save state
        state = loader1.state_dict()

        # Get next batch
        expected, _ = loader1.next()

        # New loader, restore state
        loader2 = build_loader(path, seq_len=32, batch_size=2, device="cpu")
        loader2.load_state_dict(state)

        # Should get same batch
        actual, _ = loader2.next()
        assert torch.equal(actual, expected), "Resume produced different batch"


class TestSWRRMixing:
    """Tests for Smooth Weighted Round-Robin mixing."""

    def test_weights_respected(self, temp_shards):
        """Source weights should control mixing ratio."""
        path, paths = temp_shards

        # Create loader with 80/20 mix (need two separate sources)
        sources = [
            {"id": "source_a", "paths": [paths[0]], "weight": 0.8},
            {"id": "source_b", "paths": [paths[1]], "weight": 0.2},
        ]

        loader = build_loader(sources, seq_len=32, batch_size=1, device="cpu")

        # Track which source each batch comes from
        # (source_a has tokens 0-999, source_b has 1000-1999)
        from_a = 0
        from_b = 0
        total = 100

        for _ in range(total):
            batch, _ = loader.next()
            first_token = batch[0, 0].item()
            if first_token < 1000:
                from_a += 1
            else:
                from_b += 1

        ratio_a = from_a / total
        # Should be approximately 80% from source_a
        assert 0.7 < ratio_a < 0.9, f"Expected ~80% from source_a, got {ratio_a:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
