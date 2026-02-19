"""Test Welford's algorithm correctness.

This is non-obvious math - streaming mean must match batch mean
despite floating-point accumulation across many updates.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from examples.reap.metrics import OnlineStatsTracker


@given(n_updates=st.integers(1, 100))
@settings(max_examples=30)
def test_online_mean_matches_batch(n_updates: int) -> None:
    """Online mean should match batch mean for any sequence of updates.

    Welford's algorithm with Kahan summation should be numerically stable.
    This property can't be asserted in code - it's statistical.
    """
    dim = 10
    tracker = OnlineStatsTracker((dim,), torch.device("cpu"))

    all_values = []
    for _ in range(n_updates):
        values = torch.randn(dim)
        tracker.update(values)
        all_values.append(values)

    batch_mean = torch.stack(all_values).mean(dim=0)
    assert torch.allclose(tracker.get_mean(), batch_mean, atol=1e-5)
