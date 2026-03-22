from __future__ import annotations

from types import SimpleNamespace

from rollouts.training.backends.megatron_backend import MegatronTrainingBackend


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
