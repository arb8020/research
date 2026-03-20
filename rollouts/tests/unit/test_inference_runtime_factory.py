from pathlib import Path

import pytest

from rollouts.training.configs import (
    CheckpointConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
)
from rollouts.training.inference_runtime_factory import create_inference_backend_runtime


def test_qed_vllm_realization_selects_patched_vllm_entrypoint(tmp_path: Path) -> None:
    runtime = create_inference_backend_runtime(
        model=ModelConfig(name="Qwen/Qwen3-0.6B", dtype="bfloat16"),
        inference=InferenceConfig(
            backend="vllm",
            realization="qed-vllm",
            cuda_device_ids=(0,),
        ),
        rollout=RolloutConfig(),
        checkpoint=CheckpointConfig(
            inference_sync_realization="vllm_custom_nccl_broadcast",
        ),
        output_dir=tmp_path,
    )

    engine = runtime.engines[0]
    assert runtime.realization.name == "qed-vllm"
    assert runtime.sync_realization is not None
    assert runtime.sync_realization.name == "vllm_custom_nccl_broadcast"
    assert engine.name == "qed-vllm"
    assert "python -m rollouts.inference.realizations.qed_vllm" in engine.build_launch_cmd()


def test_slime_sglang_realization_selects_local_launcher(tmp_path: Path) -> None:
    runtime = create_inference_backend_runtime(
        model=ModelConfig(name="zai-org/GLM-4.5-Air", dtype="bfloat16"),
        inference=InferenceConfig(
            backend="sglang",
            realization="slime-sglang",
            cuda_device_ids=(0,),
        ),
        rollout=RolloutConfig(),
        checkpoint=CheckpointConfig(),
        output_dir=tmp_path,
    )

    engine = runtime.engines[0]
    assert runtime.realization.name == "slime-sglang"
    assert runtime.sync_realization is None
    assert engine.name == "slime-sglang"
    assert "python -m rollouts.inference.realizations.slime_sglang" in engine.build_launch_cmd()


def test_inference_realization_backend_mismatch_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires backend"):
        create_inference_backend_runtime(
            model=ModelConfig(name="Qwen/Qwen3-0.6B", dtype="bfloat16"),
            inference=InferenceConfig(
                backend="vllm",
                realization="slime-sglang",
                cuda_device_ids=(0,),
            ),
            rollout=RolloutConfig(),
            checkpoint=CheckpointConfig(),
            output_dir=tmp_path,
        )
