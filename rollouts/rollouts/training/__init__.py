"""Training system for rollouts

Public API:
- Types: Sample, TrainFuture, WeightVersion
- Sample prep: trajectory_to_sample, prepare_sft_batch
- Protocol: TrainingBackend
"""

from rollouts.training.types import Sample, TrainFuture, WeightVersion
from rollouts.training.sample_prep import (
    trajectory_to_sample,
    compute_loss_mask,
    prepare_sft_batch,
)
from rollouts.training.backends.protocol import TrainingBackend

__all__ = [
    # Types
    "Sample",
    "TrainFuture",
    "WeightVersion",
    # Sample prep
    "trajectory_to_sample",
    "compute_loss_mask",
    "prepare_sft_batch",
    # Protocol
    "TrainingBackend",
]
