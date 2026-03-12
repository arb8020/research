from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from examples.rl.kernelbench.dataset import (
    load_kernelbench_v3_dataset,
    load_kernelbench_v3_prompts,
)
from examples.rl.kernelbench.subsets import (
    ELLIOT_V3_PROBLEM_NAMES,
    load_kernelbench_v3_elliot_prompts,
    load_kernelbench_v3_smoke_prompts,
)
from rollouts.config_status import ConfigConfidence


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_kernelbench_v3_dataset_reads_local_checkout(tmp_path: Path) -> None:
    repo_root = tmp_path / "KernelBench-v3"
    level1 = repo_root / "problems" / "level1"
    level1.mkdir(parents=True)
    (level1 / "23_Softmax.py").write_text("class Model: pass\n")

    problems = load_kernelbench_v3_dataset(root_path=repo_root, levels=[1])

    assert len(problems) == 1
    assert problems[0]["level"] == 1
    assert problems[0]["problem_id"] == 23
    assert problems[0]["name"] == "23_Softmax"
    assert problems[0]["problem_name"] == "Softmax"
    assert problems[0]["dataset"] == "kernelbench_v3"


def test_load_kernelbench_v3_prompts_preserves_problem_name(tmp_path: Path) -> None:
    repo_root = tmp_path / "KernelBench-v3"
    level1 = repo_root / "problems" / "level1"
    level1.mkdir(parents=True)
    (level1 / "23_Softmax.py").write_text(
        "class Model: pass\n\ndef get_inputs():\n    return []\n\ndef get_init_inputs():\n    return []\n"
    )

    prompts = load_kernelbench_v3_prompts(root_path=repo_root, levels=[1], backend="cuda")

    assert len(prompts) == 1
    assert prompts[0]["problem_name"] == "Softmax"
    assert prompts[0]["metadata"]["dataset"] == "kernelbench_v3"


def test_load_kernelbench_v3_elliot_prompts_selects_curated_names(tmp_path: Path) -> None:
    repo_root = tmp_path / "KernelBench-v3"
    level1 = repo_root / "problems" / "level1"
    level2 = repo_root / "problems" / "level2"
    level3 = repo_root / "problems" / "level3"
    level4 = repo_root / "problems" / "level4"
    level1.mkdir(parents=True)
    level2.mkdir(parents=True)
    level3.mkdir(parents=True)
    level4.mkdir(parents=True)

    for rel_path in (
        level1 / "23_Softmax.py",
        level4 / "1_DeepSeek_MLA.py",
        level4 / "8_KimiDeltaAttention.py",
        level1 / "19_ReLU.py",
    ):
        rel_path.write_text(
            "class Model: pass\n\ndef get_inputs():\n    return []\n\ndef get_init_inputs():\n    return []\n"
        )

    prompts = load_kernelbench_v3_elliot_prompts(root_path=repo_root, backend="cuda")
    problem_names = {prompt["problem_name"] for prompt in prompts}

    assert problem_names == {"Softmax", "DeepSeek_MLA", "KimiDeltaAttention"}


def test_load_kernelbench_v3_smoke_prompts_selects_softmax_only(tmp_path: Path) -> None:
    repo_root = tmp_path / "KernelBench-v3"
    level1 = repo_root / "problems" / "level1"
    level1.mkdir(parents=True)

    for name in ("23_Softmax.py", "19_ReLU.py"):
        (level1 / name).write_text(
            "class Model: pass\n\ndef get_inputs():\n    return []\n\ndef get_init_inputs():\n    return []\n"
        )

    prompts = load_kernelbench_v3_smoke_prompts(root_path=repo_root, backend="cuda")

    assert len(prompts) == 1
    assert prompts[0]["problem_name"] == "Softmax"


def test_trusted_kernelbench_v3_config_imports_with_explicit_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "KernelBench-v3"
    problems_root = repo_root / "problems"
    for level in ("level1", "level2", "level3", "level4"):
        (problems_root / level).mkdir(parents=True)

    for name in ELLIOT_V3_PROBLEM_NAMES:
        if name in {"VisionAttention", "MinGPTCausalAttention", "MiniGPTBlock"}:
            level = "level3"
        elif name in {
            "DeepSeek_MLA",
            "DeepSeek_MoE",
            "GroupedQueryAttention",
            "FP8_Matmul",
            "MoE_GatedGEMM",
            "INT4_Quantized_GEMM",
            "GatedDeltaNet",
            "KimiDeltaAttention",
        }:
            level = "level4"
        elif any(
            token in name
            for token in (
                "Conv2d",
                "Conv3d",
                "Matmul_",
            )
        ) and name not in {
            "Matmul_with_irregular_shapes_",
            "Matrix_vector_multiplication_",
            "Square_matrix_multiplication_",
            "Standard_matrix_multiplication_",
            "Batched_matrix_multiplication",
            "Tall_skinny_matrix_multiplication_",
        }:
            level = "level2"
        else:
            level = "level1"

        (problems_root / level / f"{name}.py").write_text(
            "class Model: pass\n\ndef get_inputs():\n    return []\n\ndef get_init_inputs():\n    return []\n"
        )

    monkeypatch.setenv("KERNELBENCH_V3_ROOT", str(repo_root))
    module = _load_module(
        Path("rollouts/configs/trusted/kernelbench/eval_elliot_v3_cuda.py").resolve()
    )

    assert len(module.tasks) == 41
    assert module.output.experiment_name == "kernelbench_v3_elliot_cuda"
    assert module.config_status.confidence == ConfigConfidence.DRAFT


def test_trusted_kernelbench_v3_smoke_config_imports_with_explicit_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "KernelBench-v3"
    level1 = repo_root / "problems" / "level1"
    level1.mkdir(parents=True)
    (level1 / "23_Softmax.py").write_text(
        "class Model: pass\n\ndef get_inputs():\n    return []\n\ndef get_init_inputs():\n    return []\n"
    )

    monkeypatch.setenv("KERNELBENCH_V3_ROOT", str(repo_root))
    module = _load_module(
        Path("rollouts/configs/trusted/kernelbench/eval_v3_smoke_cuda.py").resolve()
    )

    assert len(module.tasks) == 1
    assert module.tasks[0]["problem_name"] == "Softmax"
    assert module.output.experiment_name == "kernelbench_v3_smoke_cuda"
    assert module.config_status.confidence == ConfigConfidence.DRAFT


def test_trusted_kernelbench_v3_single_problem_config_imports_with_explicit_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "KernelBench-v3"
    level1 = repo_root / "problems" / "level1"
    level1.mkdir(parents=True)
    for name in ("1_Square_matrix_multiplication_.py", "23_Softmax.py"):
        (level1 / name).write_text(
            "class Model: pass\n\ndef get_inputs():\n    return []\n\ndef get_init_inputs():\n    return []\n"
        )

    monkeypatch.setenv("KERNELBENCH_V3_ROOT", str(repo_root))
    module = _load_module(
        Path("rollouts/configs/trusted/kernelbench/eval_v3_single_square_opus46.py").resolve()
    )

    assert len(module.tasks) == 1
    assert module.tasks[0]["problem_name"] == "Square_matrix_multiplication_"
    assert module.endpoint.model == "claude-opus-4-6"
    assert module.run.max_turns == 10
