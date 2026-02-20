# Unified RL Runner Design

## Goal

A single entry point for RL training that takes a config file specifying:
- Hardware requirements (GPU type, count, provider)
- Training algorithm and params
- Input weights
- Task definition (dataset, scoring, environment)

```bash
python -m rollouts.run_rl --config configs/kernelbench_grpo.py
```

## Current State

The existing `run.py` is close but:
1. Hardware is CLI args, not config (`--gpu-type`, `--modal`, `--provider`)
2. Task-specific logic lives in `base_config.py` files, not the config itself
3. No explicit weights input (always starts from HF model in ModelConfig)

## Proposed Config Structure

```python
# configs/kernelbench_grpo.py
from rollouts.training.configs import (
    GRPOConfig,
    HardwareConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
    CheckpointConfig,
    InferenceConfig,
)
from rollouts.training.tasks import kernelbench  # Task registry

# Hardware: what to provision
hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=1,
    provider="runpod",  # or "modal", "lambdalabs", "vast"
)

# Training config (existing GRPOConfig structure, unchanged)
training = GRPOConfig(
    model=ModelConfig(
        name="Qwen/Qwen2.5-Coder-7B-Instruct",
        # Or load from checkpoint:
        # checkpoint_path="s3://bucket/checkpoints/sft_run_123/final"
    ),
    trainer=TrainerConfig(lr=1e-6, num_minibatches=4),
    rollout=RolloutConfig(batch_size=4, n_samples_per_prompt=4, temperature=0.7),
    inference=InferenceConfig(mem_fraction=0.5),
    checkpoint=CheckpointConfig(num_steps=100, checkpoint_every=20),
)

# Task: what to train on
task = kernelbench.TaskConfig(
    levels=[1, 2],
    max_samples=100,
    backend="CUDA",
)
```

## New Components

### 1. HardwareConfig (new dataclass)

```python
@dataclass(frozen=True)
class HardwareConfig:
    """Hardware provisioning requirements."""
    gpu_type: str = "A100"
    gpu_count: int = 1
    provider: Literal["modal", "runpod", "lambdalabs", "vast", "local"] = "runpod"

    # Auto-derived from gpu_type for known GPUs
    gpu_memory_gb: int | None = None
    compute_capability: str | None = None
```

### 2. Task Registry

Each task (kernelbench, reverse_text, calculator, etc.) registers:
- `TaskConfig`: dataclass with task-specific params
- `load_prompts(config) -> list[dict]`
- `score_fn: Callable[[Sample], Score]`
- `environment_cls: type[Environment]`

```python
# rollouts/training/tasks/kernelbench.py
from dataclasses import dataclass
from rollouts.training.task_registry import register_task

@dataclass(frozen=True)
class TaskConfig:
    levels: list[int] = field(default_factory=lambda: [1])
    max_samples: int = 100
    backend: str = "CUDA"

def load_prompts(config: TaskConfig) -> list[dict]:
    return load_kernelbench_prompts(
        levels=config.levels,
        max_samples=config.max_samples,
        backend=config.backend,
    )

register_task(
    name="kernelbench",
    task_config_cls=TaskConfig,
    load_prompts=load_prompts,
    score_fn=kernelbench_score_fn,
    environment_cls=BasicEnvironment,
)
```

### 3. Unified Runner

```python
# rollouts/run_rl.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # CLI overrides (optional, override config values)
    parser.add_argument("--gpu-type", help="Override hardware.gpu_type")
    parser.add_argument("--provider", help="Override hardware.provider")
    parser.add_argument("--local", action="store_true", help="Run locally")
    args = parser.parse_args()

    module = load_config_module(args.config)

    hardware = getattr(module, "hardware", HardwareConfig())
    training = module.training  # GRPOConfig (required)
    task = module.task  # TaskConfig (required)

    # Apply CLI overrides
    if args.gpu_type:
        hardware = replace(hardware, gpu_type=args.gpu_type)
    if args.provider:
        hardware = replace(hardware, provider=args.provider)
    if args.local:
        hardware = replace(hardware, provider="local")

    # Dispatch to appropriate runner
    if hardware.provider == "local":
        run_local(training, task)
    elif hardware.provider == "modal":
        run_modal(training, task, hardware)
    else:
        run_ssh(training, task, hardware)  # runpod, lambdalabs, vast
```

## Migration Path

1. Add `HardwareConfig` to `rollouts/training/configs.py`
2. Create task registry in `rollouts/training/tasks/`
3. Move task logic from `examples/rl/*/base_config.py` to registry
4. Create `run_rl.py` that wraps existing `run.py` logic
5. Keep old entry points working (backwards compat)

## Example Configs

### Minimal (uses defaults)
```python
from rollouts.training.tasks import kernelbench

hardware = HardwareConfig(gpu_type="A100", provider="modal")
training = GRPOConfig()  # All defaults
task = kernelbench.TaskConfig()
```

### Full control
```python
hardware = HardwareConfig(
    gpu_type="H100",
    gpu_count=2,
    provider="runpod",
)

training = GRPOConfig(
    model=ModelConfig(
        name="Qwen/Qwen2.5-Coder-32B-Instruct",
        checkpoint_path="s3://weights/sft-run-123/final",
    ),
    trainer=TrainerConfig(
        lr=5e-7,
        cuda_device_ids=(1,),  # Training on GPU 1
    ),
    inference=InferenceConfig(
        cuda_device_ids=(0,),  # Inference on GPU 0
        mem_fraction=0.9,
    ),
    # ...
)

task = kernelbench.TaskConfig(
    levels=[1, 2, 3],
    max_samples=500,
    backend="HIP",
)
```

## Open Questions

1. **Checkpoint storage between stages**: S3? GCS? HF Hub?
2. **Remote scoring**: KernelBench needs GPU for scoring. Same node or separate?
3. **Resume support**: How to resume from checkpoint mid-training?
