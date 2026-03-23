from __future__ import annotations

from types import SimpleNamespace

import torch

from rollouts.training.backends.megatron_backend import (
    MegatronTrainingBackend,
    _split_megatron_batch,
)


class _FakeMegatronScheduler:
    def __init__(self) -> None:
        self.increments: list[int] = []

    def step(self, *, increment: int) -> None:
        self.increments.append(increment)

    def get_lr(self, _param_group: dict[str, object]) -> float:
        return 1.25e-5


class _FakeChainedMegatronOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 3.5e-5}]
        self.step_calls = 0
        self.clip_grad_norm_calls = 0

    def clip_grad_norm(self, _clip_grad: float) -> float:
        self.clip_grad_norm_calls += 1
        raise AssertionError("adapter must not call clip_grad_norm() directly on ChainedOptimizer")

    def step(self) -> tuple[bool, float, None]:
        self.step_calls += 1
        return True, 7.5, None


def test_megatron_optim_step_uses_megatron_step_contract_for_chained_optimizer() -> None:
    backend = object.__new__(MegatronTrainingBackend)
    backend.optimizer = _FakeChainedMegatronOptimizer()
    backend.opt_param_scheduler = _FakeMegatronScheduler()
    backend.config = SimpleNamespace(global_batch_size=8)
    backend._step = 0

    metrics = backend.optim_step()._result

    assert backend.optimizer.step_calls == 1
    assert backend.optimizer.clip_grad_norm_calls == 0
    assert backend.opt_param_scheduler.increments == [8]
    assert metrics == {
        "step": 1,
        "lr": 1.25e-5,
        "grad_norm": 7.5,
        "update_successful": 1.0,
    }


def test_split_megatron_batch_slices_example_axis() -> None:
    batch = {
        "input_ids": torch.arange(12).reshape(6, 2),
        "labels": torch.arange(12).reshape(6, 2),
        "advantages": torch.arange(6),
        "group_ids": torch.arange(6),
    }

    chunks = _split_megatron_batch(batch, micro_batch_size=2)

    assert [chunk["input_ids"].shape[0] for chunk in chunks] == [2, 2, 2]
    assert torch.equal(chunks[0]["advantages"], torch.tensor([0, 1]))
    assert torch.equal(chunks[2]["group_ids"], torch.tensor([4, 5]))
