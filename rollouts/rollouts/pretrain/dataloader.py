"""Deterministic dataloader for pretraining with multi-shard and multi-source support.

Based on nmoe's Global Stream, Local Slice pattern with SWRR mixing.

TODO: BOS-aligned packing (nanochat style)
  - Current: contiguous streaming (sequences can start mid-document)
  - Better: pack documents so every sequence starts with BOS token
  - Requires: document boundary index (.idx files with EOS positions)
  - Benefit: ~35% less token waste from cropping, cleaner document boundaries
  - See nanochat/dataloader.py for best-fit packing implementation

Features:
- Multi-shard: Read from multiple .npy/.bin files as one stream
- Multi-source: Mix multiple datasets with weighted sampling (SWRR)
- Cursor state: Exact resume via state_dict/load_state_dict
- Distributed: Each rank gets its slice of the global sequence stream

Usage:
    # Single source (simple)
    loader = build_loader(
        sources=[{"id": "fineweb", "paths": ["shard1.npy", "shard2.npy"], "weight": 1.0}],
        seq_len=512,
        batch_size=4,
        rank=0,
        world_size=1,
    )

    for step in range(1000):
        input_ids, labels = loader.next()
        # train...

    # Save state for resume
    state = loader.state_dict()

    # Multi-source mixing
    loader = build_loader(
        sources=[
            {"id": "fineweb", "paths": ["fineweb/*.npy"], "weight": 0.8},
            {"id": "code", "paths": ["code/*.npy"], "weight": 0.2},
        ],
        ...
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Cursor:
    """Position in a multi-shard dataset.

    Tracks (file_idx, pos_in_file, wrap_count) for exact resume.
    """

    file_idx: int = 0
    pos_in_file: int = 0
    wrap_count: int = 0


class ShardedDataset:
    """Memory-mapped multi-shard dataset.

    Concatenates multiple .npy or .bin files as one token stream.
    Uses numpy memmap for efficiency - only loads pages on access.
    """

    def __init__(self, shard_paths: list[str | Path]) -> None:
        if not shard_paths:
            raise ValueError("No shard paths provided")

        self.paths = [str(p) for p in shard_paths]
        self.arrs: list[np.ndarray] = []
        self.lens: list[int] = []

        for p in self.paths:
            if p.endswith(".npy"):
                arr = np.load(p, mmap_mode="r")
            elif p.endswith(".bin"):
                # modded-nanogpt format: 256 int32 header + uint16 tokens
                header = np.fromfile(p, dtype=np.int32, count=256)
                assert header[0] == 20240520, f"Invalid magic: {header[0]}"
                arr = np.memmap(p, dtype=np.uint16, mode="r", offset=256 * 4)
            else:
                # Default: assume raw uint32 tokens
                arr = np.memmap(p, dtype=np.uint32, mode="r")

            self.arrs.append(arr)
            self.lens.append(len(arr))

    def total_tokens(self) -> int:
        return sum(self.lens)

    def next_window(self, cursor: Cursor, length: int) -> tuple[np.ndarray, Cursor]:
        """Read a contiguous window of tokens, crossing shard boundaries if needed.

        Args:
            cursor: Current position
            length: Number of tokens to read

        Returns:
            (tokens, new_cursor) where tokens is [length] int64 array
        """
        remaining = length
        parts: list[np.ndarray] = []
        fidx = cursor.file_idx
        pos = cursor.pos_in_file
        wrap = cursor.wrap_count

        while remaining > 0:
            if fidx >= len(self.arrs):
                # Wrap around to start
                fidx = 0
                pos = 0
                wrap += 1

            arr = self.arrs[fidx]
            n = min(remaining, max(0, self.lens[fidx] - pos))

            if n > 0:
                parts.append(arr[pos : pos + n])
                pos += n
                remaining -= n
            else:
                # Move to next file
                fidx += 1
                pos = 0

        out = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return out.astype(np.int64), Cursor(fidx, pos, wrap)

    def advance(self, cursor: Cursor, length: int) -> Cursor:
        """Advance cursor by length tokens without reading data."""
        remaining = length
        fidx = cursor.file_idx
        pos = cursor.pos_in_file
        wrap = cursor.wrap_count

        while remaining > 0:
            if fidx >= len(self.arrs):
                fidx = 0
                pos = 0
                wrap += 1

            take = min(remaining, max(0, self.lens[fidx] - pos))
            if take > 0:
                pos += take
                remaining -= take
            else:
                fidx += 1
                pos = 0

        return Cursor(fidx, pos, wrap)


@dataclass
class SourceState:
    """State for a single data source."""

    dataset: ShardedDataset
    cursor: Cursor = field(default_factory=Cursor)
    emitted_sequences: int = 0


@dataclass
class SourceConfig:
    """Configuration for a data source."""

    id: str
    paths: list[str]
    weight: float = 1.0


class DeterministicLoader:
    """Deterministic data loader with multi-source SWRR mixing.

    Global Stream, Local Slice pattern:
    - All ranks see the same global sequence order
    - Each rank takes its slice based on rank/world_size
    - SWRR (Smooth Weighted Round-Robin) for deterministic source mixing
    """

    def __init__(
        self,
        sources: list[SourceConfig],
        seq_len: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # Initialize sources
        self.sources = sources
        self._source_states: dict[str, SourceState] = {}
        for src in sources:
            ds = ShardedDataset(src.paths)
            self._source_states[src.id] = SourceState(dataset=ds)

        # SWRR accumulators (one per source)
        self._acc = [0.0 for _ in sources]
        self._w_sum = sum(src.weight for src in sources) or 1.0

        # Global sequence counter
        self._global_seq_idx = 0

        # Compute per-step slicing
        self._seqs_per_step = batch_size
        q, r = divmod(self._seqs_per_step, world_size)
        self._mine = q + (1 if rank < r else 0)
        self._start_off = rank * q + min(rank, r)

    def _swrr_next_source_idx(self) -> int:
        """Select next source using Smooth Weighted Round-Robin."""
        # Add weights
        for i, src in enumerate(self.sources):
            self._acc[i] += src.weight

        # Select argmax
        k = int(np.argmax(self._acc))

        # Subtract total
        self._acc[k] -= self._w_sum

        return k

    def _emit_from_source(self, src_id: str) -> torch.Tensor:
        """Emit one sequence from the given source."""
        ss = self._source_states[src_id]
        arr, new_cursor = ss.dataset.next_window(ss.cursor, self.seq_len + 1)
        ss.cursor = new_cursor
        ss.emitted_sequences += 1
        self._global_seq_idx += 1
        return torch.from_numpy(arr)

    def _advance_source(self, src_id: str) -> None:
        """Advance source cursor without emitting."""
        ss = self._source_states[src_id]
        ss.cursor = ss.dataset.advance(ss.cursor, self.seq_len + 1)
        ss.emitted_sequences += 1
        self._global_seq_idx += 1

    @torch.no_grad()
    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of (input_ids, labels).

        Returns:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] (shifted by 1)
        """
        seqs: list[torch.Tensor] = []

        for i in range(self._seqs_per_step):
            # Select source via SWRR
            src_idx = self._swrr_next_source_idx()
            src_id = self.sources[src_idx].id

            # Check if this sequence belongs to our rank
            if self._start_off <= i < self._start_off + self._mine:
                t = self._emit_from_source(src_id)
                seqs.append(t)
            else:
                self._advance_source(src_id)

        if not seqs:
            raise StopIteration("No sequences for this rank")

        batch = torch.stack(seqs).to(self.device, non_blocking=True)
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict[str, Any]:
        """Get loader state for checkpointing."""
        return {
            "version": 1,
            "global_seq_idx": self._global_seq_idx,
            "accumulators": list(self._acc),
            "sources": {
                src_id: {
                    "cursor": {
                        "file_idx": ss.cursor.file_idx,
                        "pos_in_file": ss.cursor.pos_in_file,
                        "wrap_count": ss.cursor.wrap_count,
                    },
                    "emitted_sequences": ss.emitted_sequences,
                }
                for src_id, ss in self._source_states.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore loader state from checkpoint."""
        self._global_seq_idx = state.get("global_seq_idx", 0)
        self._acc = [float(v) for v in state.get("accumulators", self._acc)]

        sources_state = state.get("sources", {})
        for src_id, ss in self._source_states.items():
            if src_id in sources_state:
                src_state = sources_state[src_id]
                cursor_data = src_state.get("cursor", {})
                ss.cursor = Cursor(
                    file_idx=cursor_data.get("file_idx", 0),
                    pos_in_file=cursor_data.get("pos_in_file", 0),
                    wrap_count=cursor_data.get("wrap_count", 0),
                )
                ss.emitted_sequences = src_state.get("emitted_sequences", 0)


def build_loader(
    sources: list[dict[str, Any]] | str | Path,
    seq_len: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    device: str | torch.device = "cuda",
) -> DeterministicLoader:
    """Build a deterministic data loader.

    Args:
        sources: Either:
            - List of source configs: [{"id": "name", "paths": [...], "weight": 1.0}, ...]
            - Path to directory of .npy/.bin shards (creates single source)
        seq_len: Sequence length
        batch_size: Batch size
        rank: Current process rank
        world_size: Total number of processes
        device: Device for output tensors

    Returns:
        DeterministicLoader instance
    """
    # Handle path shorthand
    if isinstance(sources, (str, Path)):
        data_path = Path(sources)
        if data_path.is_dir():
            # Glob for shards
            npy_files = sorted(glob(str(data_path / "**/*.npy"), recursive=True))
            bin_files = sorted(glob(str(data_path / "**/*.bin"), recursive=True))
            paths = npy_files + bin_files
            if not paths:
                raise ValueError(f"No .npy or .bin files found in {data_path}")
        else:
            paths = [str(data_path)]

        sources = [{"id": "data", "paths": paths, "weight": 1.0}]

    # Parse source configs
    source_configs: list[SourceConfig] = []
    for src in sources:
        # Expand globs in paths
        expanded_paths: list[str] = []
        for p in src["paths"]:
            if "*" in p:
                expanded_paths.extend(sorted(glob(p, recursive=True)))
            else:
                expanded_paths.append(p)

        if not expanded_paths:
            raise ValueError(f"No files found for source '{src['id']}': {src['paths']}")

        source_configs.append(
            SourceConfig(
                id=src["id"],
                paths=expanded_paths,
                weight=src.get("weight", 1.0),
            )
        )

    return DeterministicLoader(
        sources=source_configs,
        seq_len=seq_len,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        device=device,
    )
