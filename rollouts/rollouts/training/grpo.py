"""GRPO Training Loop.

Shared training infrastructure for GRPO (Group Relative Policy Optimization).
Each task provides prompts, score_fn, and environment_cls - this module handles
the rest: SGLang server, training backend, rollout generation, gradient updates.

Usage:
    from ..training.grpo import GRPOConfig, grpo_train

    config = GRPOConfig(model_name="Qwen/Qwen3-0.6B", num_steps=100)
    prompts = [{"messages": [...], "answer": "42"}, ...]

    def my_score_fn(sample):
        return Score(metrics=(Metric("correct", 1.0 if correct else 0.0, weight=1.0),))

    results = grpo_train(
        config=config,
        prompts=prompts,
        score_fn=my_score_fn,
        environment_cls=BasicEnvironment,
    )
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from ..dtypes import Environment, Score
    from ..training.types import Sample

# ──────────────────────── Sub-Configs (re-exported from shared) ───────────────

from ..training.configs import (  # noqa: E402
    CheckpointConfig,
    InferenceConfig,
    ModelConfig,
    OutputConfig,
    RolloutConfig,
    TrainerConfig,
)


def GRPOOutputConfig(  # noqa: N802 — factory, not a class
    experiment_name: str = "grpo",
    output_dir: str = "results/rl",
) -> OutputConfig:
    """Create OutputConfig with GRPO defaults (output_dir="results/rl")."""
    return OutputConfig(output_dir=output_dir, experiment_name=experiment_name)


# ──────────────────────── Composed Config ────────────────────────────────────


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO training.

    Composed from sub-configs that separate concerns cleanly.
    Use replace() for inheritance (see docs/code_style/experiment_config.md).

    Example:
        config = GRPOConfig(
            model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
            trainer=TrainerConfig(lr=1e-5),
            rollout=RolloutConfig(batch_size=4, n_samples_per_prompt=4),
        )

        # Derive a variant:
        fast = replace(config, trainer=replace(config.trainer, lr=1e-4))
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(output_dir="results/rl", experiment_name="grpo")
    )

    def save(self, path: Path | str) -> None:
        """Save config to JSON."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a JSON-serializable dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert config to JSON string (for remote runners)."""
        import json

        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> GRPOConfig:
        """Construct GRPOConfig from a dict (inverse of to_dict)."""
        assert isinstance(data, dict), f"data must be dict, got {type(data)}"

        model = ModelConfig(**data.get("model", {}))
        inference = InferenceConfig(**data.get("inference", {}))
        trainer = TrainerConfig(**data.get("trainer", {}))
        rollout = RolloutConfig(**data.get("rollout", {}))
        checkpoint = CheckpointConfig(**data.get("checkpoint", {}))
        output = OutputConfig(**data.get("output", {}))

        return GRPOConfig(
            model=model,
            inference=inference,
            trainer=trainer,
            rollout=rollout,
            checkpoint=checkpoint,
            output=output,
        )


# ──────────────────────── Training Function ──────────────────────────────────


def grpo_train(
    config: GRPOConfig,
    prompts: list[dict[str, Any]],
    score_fn: Callable[[Sample], Score],
    environment_cls: type[Environment],
    metadata_key: str | None = None,
) -> dict[str, Any]:
    """Run GRPO training.

    Args:
        config: Training configuration
        prompts: List of prompt dicts, each containing:
            - "messages": List of chat messages [{"role": "...", "content": "..."}]
            - Any metadata needed by score_fn (e.g., "answer", "expected_sorted")
        score_fn: Function (Sample) -> Score that computes reward
        environment_cls: Environment class (BasicEnvironment for no tools,
            CalculatorEnvironment for calculator, etc.)
        metadata_key: If set, extract this key from prompt dict to pass as metadata.
            If None, passes all non-"messages" keys as metadata.

    Returns:
        Dict with "metrics_history" list of per-step metrics

    Example:
        >>> from ..training.grpo import GRPOConfig, grpo_train
        >>> from ..environments.no_tools import BasicEnvironment
        >>>
        >>> config = GRPOConfig(model_name="Qwen/Qwen3-0.6B", num_steps=10)
        >>> prompts = [
        ...     {"messages": [{"role": "user", "content": "2+2=?"}], "answer": "4"},
        ... ]
        >>> results = grpo_train(config, prompts, my_score_fn, BasicEnvironment)
    """
    return trio.run(_grpo_train_async, config, prompts, score_fn, environment_cls, metadata_key)


# ──────────────────────── Training Helpers ────────────────────────────────────


def _setup_output_dir(config: GRPOConfig) -> tuple[Path, str]:
    """Setup output directory and run name.

    Priority:
    1. ROLLOUTS_OUTPUT_DIR env var (set by runner — single source of truth)
    2. ROLLOUTS_RUN_NAME env var + config.output.output_dir (legacy)
    3. Generate from config (local execution)

    Returns:
        Tuple of (output_dir, run_name)
    """
    import os
    from datetime import datetime, timezone

    explicit_dir = os.environ.get("ROLLOUTS_OUTPUT_DIR")
    if explicit_dir:
        output_dir = Path(explicit_dir)
        run_name = output_dir.name
    elif run_name_env := os.environ.get("ROLLOUTS_RUN_NAME"):
        run_name = run_name_env
        output_dir = Path(config.output.output_dir) / run_name
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_name = f"{config.output.experiment_name}_{timestamp}"
        output_dir = Path(config.output.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, run_name


def _create_inference_engine(
    config: GRPOConfig, output_dir: Path
) -> Any:  # SGLangEngine | VLLMEngine
    """Create single inference engine (legacy API, calls _create_inference_engines)."""
    engines = _create_inference_engines(config, output_dir)
    if len(engines) != 1:
        raise ValueError(
            f"_create_inference_engine expects 1 engine but config has {len(engines)}. "
            "Use _create_inference_engines() for multi-engine setup."
        )
    return engines[0]


def _create_inference_engines(
    config: GRPOConfig, output_dir: Path
) -> list[Any]:  # list[SGLangEngine | VLLMEngine | EngineV2Engine]
    """Create multiple inference engines for parallel rollout generation.

    Each engine runs on its own GPU(s) and port. More engines = more samples/second.

    Returns:
        List of inference engines (one per GPU or TP group)
    """
    from ..training.weight_sync import EngineV2Engine, SGLangEngine, VLLMEngine

    engines = []
    gpu_assignments = config.inference.gpu_assignments
    ports = config.inference.ports

    for _idx, (gpus, port) in enumerate(zip(gpu_assignments, ports, strict=False)):
        if config.inference.backend == "sglang":
            # NCCL weight sync uses HTTP API (init_weights_update_group),
            # not CLI flags. See weight_sync.py for implementation.
            engine = SGLangEngine(
                model_name=config.model.name,
                port=port,
                cuda_device_ids=gpus,
                output_dir=output_dir,
                dtype=config.model.dtype,
                mem_fraction=config.inference.mem_fraction,
            )
        elif config.inference.backend == "vllm":
            engine = VLLMEngine(
                model_name=config.model.name,
                port=port,
                cuda_device_ids=gpus,
                output_dir=output_dir,
                dtype=config.model.dtype,
                gpu_memory_utilization=config.inference.mem_fraction,
            )
        elif config.inference.backend == "engine_v2":
            # Rollouts native inference engine
            # max_batch_size = prompts * samples_per_prompt + headroom
            max_batch = config.rollout.batch_size * config.rollout.n_samples_per_prompt * 2
            engine = EngineV2Engine(
                model_name=config.model.name,
                port=port,
                cuda_device_ids=gpus,
                output_dir=output_dir,
                dtype=config.model.dtype,
                mem_fraction=config.inference.mem_fraction,
                max_batch_size=max_batch,
                max_seq_len=config.rollout.max_seq_len,
            )
        else:
            raise ValueError(f"Unknown inference backend: {config.inference.backend}")
        engines.append(engine)

    return engines


def _create_teacher_engine(
    config: GRPOConfig, output_dir: Path
) -> Any | None:  # SGLangEngine | None
    """Create teacher inference engine for OPD (On-Policy Distillation).

    The teacher model runs on a separate GPU and provides dense feedback
    by scoring student-generated tokens. Returns None if OPD is not enabled.

    Args:
        config: Training config with teacher settings
        output_dir: Directory for logs

    Returns:
        Teacher SGLangEngine, or None if not using OPD
    """
    # Only create teacher if using OPD
    if config.trainer.advantage_estimator != "opd":
        return None

    if config.trainer.teacher_model is None:
        raise ValueError(
            "advantage_estimator='opd' requires teacher_model to be set. "
            "Example: TrainerConfig(advantage_estimator='opd', teacher_model='Qwen/Qwen3-8B')"
        )

    from ..training.weight_sync import SGLangEngine

    # Teacher runs on a separate GPU (assumes GPU 1 for now)
    # TODO: Make teacher GPU configurable via DistributedConfig
    student_gpus = set(config.inference.cuda_device_ids)
    trainer_gpus = set(config.trainer.cuda_device_ids)
    used_gpus = student_gpus | trainer_gpus

    # Find first unused GPU for teacher (simple heuristic)
    # For 2-GPU setup: GPU 0 = student, GPU 1 = teacher
    teacher_gpu = max(used_gpus) + 1 if used_gpus else 1

    # Teacher uses higher mem fraction since it doesn't share with training
    teacher_engine = SGLangEngine(
        model_name=config.trainer.teacher_model,
        port=config.trainer.teacher_port,
        cuda_device_ids=(teacher_gpu,),
        output_dir=output_dir,
        dtype=config.model.dtype,
        mem_fraction=0.9,  # Teacher doesn't share GPU, can use more VRAM
    )

    return teacher_engine


def _make_loss_fn(
    trainer: TrainerConfig,
    vanilla_fn: Callable,
    clipped_fn: Callable,
    masked_fn: Callable,
    opd_fn: Callable | None = None,
) -> Callable:
    """Build the loss function from trainer config."""
    if trainer.loss_type == "vanilla":
        return vanilla_fn
    if trainer.loss_type == "clipped":
        return clipped_fn
    if trainer.loss_type == "masked":
        lo, hi = trainer.mask_ratio_low, trainer.mask_ratio_high

        def _masked(logits: Any, batch: Any) -> Any:
            return masked_fn(logits, batch, ratio_low=lo, ratio_high=hi)

        return _masked
    if trainer.loss_type == "opd":
        if opd_fn is None:
            raise ValueError("opd_fn must be provided for loss_type='opd'")
        return opd_fn
    raise ValueError(
        f"Unknown loss_type: {trainer.loss_type!r}. Use 'vanilla', 'clipped', 'masked', or 'opd'."
    )


def _setup_training_backend(
    config: GRPOConfig,
    output_dir: Path,
    inference_engine: Any,
    megatron_workers: list[Any] | None = None,
) -> tuple[Any, Any, Any, Callable[[], None] | None]:  # (backend, tokenizer, endpoint, cleanup)
    """Setup training backend, tokenizer, and endpoint.

    Args:
        config: GRPO training configuration.
        output_dir: Output directory for checkpoints.
        inference_engine: Inference engine for weight sync.
        megatron_workers: Pre-spawned megatron workers (fork before CUDA init).
            Required for megatron backend - workers must be forked before any
            CUDA context is created (e.g., before SGLang starts).

    Returns:
        Tuple of (backend, tokenizer, endpoint, cleanup).
        cleanup is an optional callable to run at shutdown (e.g., destroy process group).
    """
    # TODO: Remove HF transformers dependency. Use tokenizers library directly
    # or load tokenizer.json with custom wrapper.
    from transformers import AutoTokenizer

    from ..dtypes import Endpoint
    from ..training.backends.pytorch_factory import create_pytorch_backend, parse_dtype
    from ..training.losses import grpo_loss, grpo_loss_clipped, grpo_loss_masked, opd_loss

    # Select loss function based on config
    loss_fn = _make_loss_fn(
        config.trainer, grpo_loss, grpo_loss_clipped, grpo_loss_masked, opd_fn=opd_loss
    )

    cleanup: Callable[[], None] | None = None
    backend_name = config.trainer.backend

    if backend_name == "pytorch":
        gpu_rank = config.trainer.cuda_device_ids[0]
        backend = create_pytorch_backend(
            model_name=config.model.name,
            checkpoint_dir=output_dir,
            device_type="cuda",
            dtype=config.model.dtype,
            gpu_rank=gpu_rank,
            learning_rate=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
            loss_fn=loss_fn,
            num_minibatches=config.trainer.num_minibatches,
            max_grad_norm=config.trainer.max_grad_norm,
            use_lora=config.model.use_lora,
            lora_rank=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
        )
    elif backend_name == "nmoe":
        from ..training.backends.nmoe_backend import NmoeConfig, NmoeTrainingBackend

        gpu_rank = config.trainer.cuda_device_ids[0]
        nmoe_cfg = NmoeConfig(
            dtype=config.model.dtype,
            lr_dense=config.trainer.lr,
            lr_router=config.trainer.lr,
            lr_muon=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
        )
        backend = NmoeTrainingBackend(
            model_name=config.model.name,
            checkpoint_dir=output_dir,
            loss_fn=loss_fn,
            config=nmoe_cfg,
            device_type="cuda",
            gpu_rank=gpu_rank,
            num_minibatches=config.trainer.num_minibatches,
            max_grad_norm=config.trainer.max_grad_norm,
            use_lora=config.model.use_lora,
            lora_rank=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
        )
    elif backend_name in ("fsdp", "fsdp2"):
        if backend_name == "fsdp2":
            logging.getLogger(__name__).warning(
                "trainer.backend='fsdp2' selected; using FSDPTrainingBackend (fully_shard) "
                "bring-up path for now."
            )
        # Single-process FSDP bring-up path.
        # Multi-process/multi-node FSDP is orchestrated via rollouts.training.multi_node + fsdp_worker.
        import os
        import socket

        import torch
        import torch.distributed as dist
        from transformers import AutoModelForCausalLM

        from ..training.backends.fsdp import FSDPConfig, FSDPTrainingBackend

        trainer_gpu = config.trainer.cuda_device_ids[0]
        torch.cuda.set_device(trainer_gpu)

        if not dist.is_initialized():
            # Find an available port (avoid conflicts with weight sync ports / stale processes).
            def find_free_port(start_port: int, max_attempts: int = 100) -> int:
                for port in range(start_port, start_port + max_attempts):
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            s.bind(("", port))
                            return port
                    except OSError:
                        continue
                raise RuntimeError(
                    f"No free port found in range {start_port}-{start_port + max_attempts}"
                )

            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = find_free_port(config.checkpoint.nccl_master_port + 50)

            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=0,
                world_size=1,
            )

            def _cleanup_dist() -> None:
                if dist.is_initialized():
                    dist.destroy_process_group()

            cleanup = _cleanup_dist

        # Load model on CPU then let backend move it to the correct CUDA device.
        torch_dtype = parse_dtype(config.model.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Optimizer factory (called AFTER FSDP wrapping).
        def make_optimizer(fsdp_model: torch.nn.Module) -> torch.optim.Optimizer:
            return torch.optim.AdamW(
                fsdp_model.parameters(),
                lr=config.trainer.lr,
                weight_decay=config.trainer.weight_decay,
            )

        fsdp_config = FSDPConfig(
            sharding_strategy="FULL_SHARD",
            mixed_precision=(torch_dtype in (torch.bfloat16, torch.float16)),
            gradient_checkpointing=False,
            clip_grad=config.trainer.max_grad_norm,
        )

        backend = FSDPTrainingBackend(
            model=model,
            optimizer_fn=make_optimizer,
            loss_fn=loss_fn,
            checkpoint_dir=output_dir,
            config=fsdp_config,
            device=torch.device(f"cuda:{trainer_gpu}"),
        )
    elif backend_name == "megatron":
        # Megatron backend using miniray for multi-process orchestration.
        # Workers run megatron_worker.py and communicate via miniray IPC.
        #
        # IMPORTANT: Workers must be pre-spawned (forked) BEFORE any CUDA context
        # is created (e.g., before SGLang starts). This is because CUDA contexts
        # don't survive fork() - the child inherits a broken context.
        # See: docs/code_style/archive/domain/multiprocessing_heinrich.md
        from ..training.backends.megatron.remote_backend import (
            MegatronRemoteBackend,
            MegatronRemoteConfig,
        )

        if megatron_workers is None:
            raise ValueError(
                "megatron backend requires pre-spawned workers. "
                "Workers must be forked before CUDA initialization (before SGLang starts). "
                "Pass megatron_workers parameter from _grpo_train_async."
            )

        megatron_config = MegatronRemoteConfig(
            model_name=config.model.name,
            dtype=config.model.dtype,
            tensor_parallel_size=config.trainer.tensor_parallel_size,
            pipeline_parallel_size=config.trainer.pipeline_parallel_size,
            expert_parallel_size=config.trainer.expert_parallel_size,
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
            max_grad_norm=config.trainer.max_grad_norm,
            micro_batch_size=config.trainer.micro_batch_size or 1,
            global_batch_size=config.rollout.batch_size,
            seq_length=config.trainer.seq_length,
            master_port=config.checkpoint.nccl_master_port,
            inference_endpoints=[f"http://localhost:{config.inference.port}"],
        )

        backend = MegatronRemoteBackend(
            workers=megatron_workers,
            config=megatron_config,
            checkpoint_dir=output_dir,
        )
        backend.initialize()

        def _cleanup_megatron() -> None:
            backend.shutdown()

        cleanup = _cleanup_megatron
    elif backend_name == "torchtitan":
        # TorchTitan backend for GLM and other models with 4D parallelism
        import os
        import socket

        import torch
        import torch.distributed as dist

        from ..training.backends.torchtitan_backend import TorchTitanBackend, TorchTitanConfig

        # Import GLM to register with torchtitan
        from ..training.models import glm  # noqa: F401

        trainer_gpu = config.trainer.cuda_device_ids[0]
        torch.cuda.set_device(trainer_gpu)

        if not dist.is_initialized():

            def find_free_port(start_port: int, max_attempts: int = 100) -> int:
                for port in range(start_port, start_port + max_attempts):
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            s.bind(("", port))
                            return port
                    except OSError:
                        continue
                raise RuntimeError(
                    f"No free port found in range {start_port}-{start_port + max_attempts}"
                )

            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = find_free_port(config.checkpoint.nccl_master_port + 50)

            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=0,
                world_size=1,
            )

            def _cleanup_dist() -> None:
                if dist.is_initialized():
                    dist.destroy_process_group()

            cleanup = _cleanup_dist

        torchtitan_config = TorchTitanConfig(
            tp_degree=config.trainer.torchtitan_tp,
            cp_degree=config.trainer.torchtitan_cp,
            pp_degree=config.trainer.torchtitan_pp,
            seq_len=config.rollout.max_seq_len,
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
            max_grad_norm=config.trainer.max_grad_norm,
        )

        backend = TorchTitanBackend(
            model_name=config.trainer.torchtitan_model,
            model_size=config.trainer.torchtitan_model_size,
            checkpoint_dir=output_dir,
            loss_fn=loss_fn,
            config=torchtitan_config,
            hf_checkpoint=config.model.name,  # Load weights from HF
        )
    else:
        raise ValueError(
            f"Unknown trainer backend: {backend_name!r}. "
            "Use 'pytorch', 'fsdp', 'fsdp2', 'nmoe', 'megatron', or 'torchtitan'."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    endpoint = Endpoint(
        model=f"openai/{config.model.name}",
        base_url=inference_engine.api_base,  # Include /v1 for OpenAI SDK
        api_format="openai-completions",
        temperature=config.rollout.temperature,
        max_tokens=config.rollout.max_tokens,
        extra_params=config.rollout.extra_params or None,
    )

    return backend, tokenizer, endpoint, cleanup


def _create_generate_fn(
    config: GRPOConfig,
    endpoint: Any,
    tokenizer: Any,
    environment_cls: type[Environment],
    metadata_key: str | None,
    logger: logging.Logger,
) -> Callable:
    """Create the generate function for rollout generation.

    Uses the agent loop with standard providers. Providers now always request
    logprobs and populate Choice.token_ids, so token data flows through
    the normal agent path.

    For multi-turn agent rollouts (max_turns > 1), the agent loop handles
    tool execution. For single-turn (max_turns = 1), it's just one LLM call.
    """
    return _create_agent_generate_fn(
        config, endpoint, tokenizer, environment_cls, metadata_key, logger
    )


def _create_agent_generate_fn(
    config: GRPOConfig,
    endpoint: Any,
    tokenizer: Any,
    environment_cls: type[Environment],
    metadata_key: str | None,
    logger: logging.Logger,
) -> Callable:
    """Create standard agent rollout generate function."""
    from ..training.agent_integration import agent_rollout_to_sample

    async def generate_fn(batch_prompts: list[dict], **kwargs: Any) -> list:
        results = []
        for prompt_data in batch_prompts:
            messages = prompt_data["messages"]
            if metadata_key:
                metadata = {metadata_key: prompt_data.get(metadata_key)}
            else:
                metadata = {k: v for k, v in prompt_data.items() if k != "messages"}

            try:
                sample = await agent_rollout_to_sample(
                    prompt=messages,
                    environment_cls=environment_cls,
                    endpoint=endpoint,
                    tokenizer=tokenizer,
                    max_turns=config.rollout.max_turns,
                    metadata=metadata,
                )
                results.append(sample)
            except Exception as e:
                logger.warning(f"Rollout failed: {e}")

        return results

    return generate_fn


def _build_grpo_run_context(
    config: GRPOConfig,
    run_name: str,
    output_dir: Path,
    node_id: str | None = None,
    num_inference_engines: int | None = None,
) -> dict[str, Any]:
    """Build canonical per-run context fields for wide logging."""
    import os
    import socket

    return {
        "run_name": run_name,
        "experiment_name": config.output.experiment_name,
        "output_dir": str(output_dir),
        "model_name": config.model.name,
        "trainer_backend": config.trainer.backend,
        "inference_backend": config.inference.backend,
        "trainer_cuda_device_ids": tuple(config.trainer.cuda_device_ids),
        "inference_cuda_device_ids": tuple(config.inference.cuda_device_ids),
        "rollout_batch_size": config.rollout.batch_size,
        "n_samples_per_prompt": config.rollout.n_samples_per_prompt,
        "max_seq_len": config.rollout.max_seq_len,
        "max_tokens": config.rollout.max_tokens,
        "pipeline_mode": config.checkpoint.pipeline_mode,
        "advantage_estimator": config.trainer.advantage_estimator,
        "weight_sync_mode": config.checkpoint.weight_sync_mode,
        "num_inference_engines": num_inference_engines,
        "node_id": node_id or os.environ.get("ROLLOUTS_NODE_ID"),
        "hostname": socket.gethostname(),
    }


async def _process_training_step(
    step: int,
    batch: Any,
    config: GRPOConfig,
    backend: Any,
    tokenizer: Any,
    device: str,
    output_dir: Path,
    logger: logging.Logger,
    run_context: dict[str, Any],
    *,
    node_id: str | None = None,
) -> dict[str, Any] | None:
    """Process a single training step.

    Returns:
        Step metrics dict, or None if step was skipped
    """
    import json
    import time

    import torch
    import torch.distributed as dist

    from ..training.losses import compute_group_advantages

    step_start = time.perf_counter()

    if not batch.tokens:
        logger.warning(
            "No successful rollouts, step skipped",
            extra={
                "event": "grpo_step_skipped",
                "step": step + 1,
                "reason": "no_successful_rollouts",
                **run_context,
                "node_id": node_id or run_context.get("node_id"),
            },
        )
        return None

    # Save rollouts to JSONL
    rollouts_file = output_dir / "rollouts.jsonl"
    with open(rollouts_file, "a") as f:  # noqa: ASYNC230
        for sample in batch.samples:
            record = {
                "step": step + 1,
                "prompt": sample.prompt,
                "response": sample.response,
                "reward": sample.reward,
                "status": sample.status.value,
                "group_index": sample.group_index,
                "weight_version": getattr(sample, "weight_version", 0),
                "turns": sample.metadata.get("turns"),
                "stop_reason": sample.metadata.get("stop_reason"),
                "messages": sample.metadata.get("messages"),
                "metadata": {
                    k: v
                    for k, v in sample.metadata.items()
                    if k not in ("turns", "stop_reason", "messages")
                },
            }
            f.write(json.dumps(record) + "\n")
            logger.debug("rollout", extra={"event": "rollout", **record})

    # Compute advantages
    rewards = batch.rewards
    group_indices = batch.group_indices
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    num_groups = len(set(group_indices)) if group_indices else len(rewards)
    logger.info(f"Reward: {mean_reward:.3f} ({len(rewards)} samples, {num_groups} groups)")

    if group_indices and len(set(group_indices)) > 1:
        advantages = compute_group_advantages(rewards, group_indices).to(device)
    else:
        advantages = torch.tensor([r - mean_reward for r in rewards], device=device)

    # Prepare batch tensors
    prep_start = time.perf_counter()
    training_batch = _prepare_training_batch(batch, config, tokenizer, advantages, device)
    prep_ms = (time.perf_counter() - prep_start) * 1000

    # Training step - forward/backward
    fb_start = time.perf_counter()
    fb_future = backend.forward_backward(training_batch)
    fb_metrics = await fb_future.result()
    fb_ms = (time.perf_counter() - fb_start) * 1000

    # Optimizer step
    optim_start = time.perf_counter()
    optim_future = backend.optim_step()
    optim_metrics = await optim_future.result()
    optim_ms = (time.perf_counter() - optim_start) * 1000

    accumulated_metrics = {**fb_metrics, **optim_metrics}
    pg_loss = accumulated_metrics.get("pg_loss", 0.0)
    entropy = accumulated_metrics.get("entropy", 0.0)

    step_metrics = {
        "mean_reward": mean_reward,
        "num_samples": len(rewards),
        "num_groups": num_groups,
        **accumulated_metrics,
    }

    logger.debug("metrics", extra={"event": "step_metrics", "step": step + 1, **step_metrics})

    masked_frac = accumulated_metrics.get("masked_frac", 0.0)
    avg_ratio = accumulated_metrics.get("avg_ratio", 1.0)
    avg_advantage = accumulated_metrics.get("avg_advantage", 0.0)
    if (step + 1) % config.checkpoint.log_every == 0:
        # Include key diagnostic metrics for debugging loss issues
        logger.info(
            f"Step {step + 1}: reward={mean_reward:.3f} | "
            f"pg_loss={pg_loss:.4f} | entropy={entropy:.2f} | "
            f"masked={masked_frac:.2f} | ratio={avg_ratio:.3f} | adv={avg_advantage:.3f}"
        )

    step_total_ms = (time.perf_counter() - step_start) * 1000

    # Wide event: one structured log per step with all timing and context
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # Free CUDA memory after training step (QED-Nano pattern)
    torch.cuda.empty_cache()

    # Collect memory stats for wide event
    import psutil

    gpu_allocated_gb = torch.cuda.memory_allocated() / 1e9
    gpu_reserved_gb = torch.cuda.memory_reserved() / 1e9
    ram_gb = psutil.Process().memory_info().rss / 1e9

    logger.info(
        "step_complete",
        extra={
            **run_context,
            "event": "step_complete",
            "step": step + 1,
            "node_id": node_id or run_context.get("node_id"),
            "rank": rank,
            "world_size": world_size,
            # Timings (ms)
            "prep_ms": round(prep_ms, 1),
            "forward_backward_ms": round(fb_ms, 1),
            "optim_ms": round(optim_ms, 1),
            "step_total_ms": round(step_total_ms, 1),
            # Metrics
            "mean_reward": mean_reward,
            "pg_loss": pg_loss,
            "entropy": entropy,
            "num_samples": len(rewards),
            "num_groups": num_groups,
            # Memory (GB) - for debugging leaks
            "gpu_allocated_gb": round(gpu_allocated_gb, 3),
            "gpu_reserved_gb": round(gpu_reserved_gb, 3),
            "ram_gb": round(ram_gb, 3),
        },
    )

    return step_metrics


def _prepare_training_batch(
    batch: Any,
    config: GRPOConfig,
    tokenizer: Any,
    advantages: Any,
    device: str,
) -> dict[str, Any]:
    """Prepare tensors for training step."""
    import torch

    max_len = min(max(len(t) for t in batch.tokens), config.rollout.max_seq_len)

    batch_tokens = []
    batch_loss_masks = []
    batch_rollout_logprobs = []
    batch_teacher_logprobs = []
    has_rollout_logprobs = batch.rollout_log_probs is not None
    has_teacher_logprobs = batch.teacher_log_probs is not None

    for i, (toks, mask) in enumerate(zip(batch.tokens, batch.loss_masks, strict=True)):
        assert isinstance(toks, list) and (not toks or isinstance(toks[0], int)), (
            f"batch.tokens[{i}] must be list[int], got list with first element type {type(toks[0]) if toks else 'empty'}: {toks[:3] if toks else []}"
        )
        toks_truncated = toks[:max_len]
        mask_truncated = mask[:max_len]
        pad_len = max_len - len(toks_truncated)
        toks_padded = toks_truncated + [tokenizer.pad_token_id or 0] * pad_len
        mask_padded = mask_truncated + [0.0] * pad_len
        batch_tokens.append(toks_padded)
        batch_loss_masks.append(mask_padded)

        if has_rollout_logprobs:
            rlp = list(batch.rollout_log_probs[i][:max_len])
            rlp_padded = rlp + [0.0] * (max_len - len(rlp))
            batch_rollout_logprobs.append(rlp_padded)

        if has_teacher_logprobs:
            tlp = list(batch.teacher_log_probs[i][:max_len])
            tlp_padded = tlp + [0.0] * (max_len - len(tlp))
            batch_teacher_logprobs.append(tlp_padded)

    input_ids = torch.tensor(batch_tokens, device=device)
    # Shift labels left: labels[i] = input_ids[i+1] (causal LM prediction target)
    # logits[i] predicts token at position i+1, so labels[i] should be input_ids[i+1]
    labels = torch.cat([input_ids[:, 1:], torch.zeros_like(input_ids[:, :1])], dim=1)
    loss_mask = torch.tensor(batch_loss_masks, device=device)
    # Also shift loss_mask left to match shifted labels
    loss_mask = torch.cat([loss_mask[:, 1:], torch.zeros_like(loss_mask[:, :1])], dim=1)

    training_batch = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "advantages": advantages,
    }

    if has_rollout_logprobs:
        rollout_logprobs_tensor = torch.tensor(batch_rollout_logprobs, device=device)
        # Shift rollout_logprobs left to match shifted labels/loss_mask
        # After shift: position i has logprob for token i+1
        rollout_logprobs_tensor = torch.cat(
            [rollout_logprobs_tensor[:, 1:], torch.zeros_like(rollout_logprobs_tensor[:, :1])],
            dim=1,
        )
        seq_rollout_logprobs = (rollout_logprobs_tensor * loss_mask).sum(dim=1) / loss_mask.sum(
            dim=1
        ).clamp(min=1.0)
        training_batch["old_logprobs"] = seq_rollout_logprobs

    if has_teacher_logprobs:
        teacher_logprobs_tensor = torch.tensor(batch_teacher_logprobs, device=device)
        # Shift teacher_logprobs left to match shifted labels/loss_mask
        teacher_logprobs_tensor = torch.cat(
            [teacher_logprobs_tensor[:, 1:], torch.zeros_like(teacher_logprobs_tensor[:, :1])],
            dim=1,
        )
        training_batch["teacher_logprobs"] = teacher_logprobs_tensor

    return training_batch


async def _grpo_train_async(
    config: GRPOConfig,
    prompts: list[dict[str, Any]],
    score_fn: Callable[[Sample], Score],
    environment_cls: type[Environment],
    metadata_key: str | None = None,
) -> dict[str, Any]:
    """Async GRPO training implementation."""
    from .._logging import setup_logging
    from ..training.datasets.data_buffer import DataBuffer
    from ..training.metrics import JSONLLogger
    from ..training.rollout_gen.async_rollout_manager import AsyncRolloutManager
    from ..training.types import RolloutConfig

    # Setup output directory first (needed for log file path)
    output_dir, run_name = _setup_output_dir(config)

    # Setup logging: structured JSON to file, human-readable to stderr
    # This keeps tracebacks clean in stderr while structured logs go to training.jsonl
    setup_logging(
        level="INFO",
        use_json=False,  # stderr stays human-readable for tracebacks
        use_color=True,
        log_file=str(output_dir / "training.jsonl"),  # structured logs go here
        logger_levels={"httpx": "WARNING", "httpcore": "WARNING"},
    )
    logger = logging.getLogger(__name__)
    import os

    logger.info("=" * 60)
    logger.info(f"GRPO Training: {run_name}")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Backend: {config.inference.backend}")
    logger.info(f"Steps: {config.checkpoint.num_steps}")
    logger.info(
        f"Batch: {config.rollout.batch_size} prompts x {config.rollout.n_samples_per_prompt} samples"
    )
    logger.info(f"Output: {output_dir}")

    # Preflight check: validate config against hardware limits
    # This fails fast if config is likely to OOM
    # Only run on rank 0 to avoid duplicate checks in torchrun/DDP
    # Skip for torchtitan - the FSDP sharding estimation is tricky and we trust the manual calc
    from ..training.preflight import run_preflight_check

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend_name = getattr(config.trainer, "backend", "pytorch")
    if local_rank == 0 and backend_name != "torchtitan":
        try:
            # Detect GPU type from CUDA device
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                run_preflight_check(config, gpu_name)
            else:
                logger.warning("CUDA not available, skipping preflight check")
        except ValueError as e:
            logger.exception(f"Preflight check failed: {e}")
            raise
    elif backend_name == "torchtitan":
        logger.info("Skipping preflight check for torchtitan (FSDP sharding handled by backend)")

    config.save(output_dir / "config.json")
    metrics_logger = JSONLLogger(output_dir)

    # =========================================================================
    # CRITICAL: Spawn megatron workers BEFORE any CUDA initialization
    # =========================================================================
    # Workers must be forked before CUDA context is created (e.g., before SGLang
    # starts). CUDA contexts don't survive fork() - children inherit broken state.
    # See: docs/code_style/archive/domain/multiprocessing_heinrich.md
    megatron_workers: list[Any] | None = None
    if backend_name == "megatron":
        from .backends.megatron.remote_backend import (
            MegatronRemoteConfig,
            spawn_megatron_workers,
        )

        num_trainer_gpus = len(config.trainer.cuda_device_ids)
        logger.info(f"Spawning {num_trainer_gpus} megatron workers (before CUDA init)...")

        # Create config for worker spawning (full config passed at init time)
        megatron_config = MegatronRemoteConfig(
            model_name=config.model.name,
            dtype=config.model.dtype,
            tensor_parallel_size=config.trainer.tensor_parallel_size,
            pipeline_parallel_size=config.trainer.pipeline_parallel_size,
            expert_parallel_size=config.trainer.expert_parallel_size,
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
            max_grad_norm=config.trainer.max_grad_norm,
            micro_batch_size=config.trainer.micro_batch_size or 1,
            global_batch_size=config.rollout.batch_size,
            seq_length=config.trainer.seq_length,
            master_port=config.checkpoint.nccl_master_port,
            inference_endpoints=[f"http://localhost:{config.inference.port}"],
        )

        megatron_workers = spawn_megatron_workers(
            num_gpus=num_trainer_gpus,
            config=megatron_config,
        )
        logger.info(f"Spawned {len(megatron_workers)} megatron workers")

    # Launch inference engine(s) - multi-engine for higher throughput
    # NOTE: This is when CUDA gets initialized (SGLang loads model on GPU 0)
    inference_engines = _create_inference_engines(config, output_dir)
    num_engines = len(inference_engines)
    run_context = _build_grpo_run_context(
        config=config,
        run_name=run_name,
        output_dir=output_dir,
        node_id=os.environ.get("ROLLOUTS_NODE_ID"),
        num_inference_engines=num_engines,
    )

    if num_engines == 1:
        gpu_str = ",".join(str(g) for g in config.inference.cuda_device_ids)
        logger.info(f"Launching {inference_engines[0].name} on GPU {gpu_str}...")
    else:
        logger.info(f"Launching {num_engines} inference engines (PipelineRL-style)...")
        for i, engine in enumerate(inference_engines):
            gpu_str = ",".join(str(g) for g in config.inference.gpu_assignments[i])
            logger.info(f"  Engine {i}: {engine.name} on GPU {gpu_str}, port {engine.port}")

    for engine in inference_engines:
        engine.launch()
        engine.start_log_tailer()

    # Primary engine for backward compat (endpoint creation uses first engine's base_url)
    inference_engine = inference_engines[0]

    # Launch teacher engine for OPD (On-Policy Distillation)
    teacher_engine = _create_teacher_engine(config, output_dir)
    if teacher_engine is not None:
        teacher_gpu_str = ",".join(str(g) for g in teacher_engine.cuda_device_ids)
        logger.info(
            f"Launching teacher model ({config.trainer.teacher_model}) "
            f"on GPU {teacher_gpu_str}, port {teacher_engine.port}..."
        )
        teacher_engine.launch()
        teacher_engine.start_log_tailer()

    backend_cleanup: Callable[[], None] | None = None

    try:
        # Wait for all engines to be ready in parallel
        startup_timeout = config.inference.startup_timeout
        async with trio.open_nursery() as startup_nursery:
            for engine in inference_engines:
                startup_nursery.start_soon(engine.wait_until_ready, startup_timeout)
            if teacher_engine is not None:
                startup_nursery.start_soon(teacher_engine.wait_until_ready, startup_timeout)
        logger.info(f"All {num_engines} inference engine(s) ready")
        if teacher_engine is not None:
            logger.info("Teacher engine ready")

        # Setup training backend (pass pre-spawned workers for megatron)
        backend, tokenizer, endpoint, backend_cleanup = _setup_training_backend(
            config, output_dir, inference_engine, megatron_workers=megatron_workers
        )
        device = f"cuda:{config.trainer.cuda_device_ids[0]}"

        # Load checkpoint if provided (for SFT → RL pipeline)
        if config.model.checkpoint_path:
            ckpt_path = Path(config.model.checkpoint_path)
            if (ckpt_path / "pytorch_model.bin").exists():
                # Our checkpoint format
                logger.info(f"Loading checkpoint from {ckpt_path}")
                await backend.load_checkpoint(ckpt_path)
                logger.info("Checkpoint loaded successfully")
            elif (ckpt_path / "config.json").exists():
                # HuggingFace format - already loaded via model_name
                logger.info(f"Using HuggingFace checkpoint: {ckpt_path}")
            else:
                raise ValueError(
                    f"Invalid checkpoint path: {ckpt_path} (no pytorch_model.bin or config.json)"
                )

        # VRAM preflight: dry-run one forward+backward at worst-case seq_len
        if not config.trainer.skip_vram_check:
            from ..training.vram import preflight_vram_check

            preflight_vram_check(backend, config, device)
        else:
            logger.info("VRAM preflight check skipped (skip_vram_check=True)")

        # Initialize NCCL weight sync if enabled (PipelineRL-style in-flight updates).
        # Skip for true_pipeline mode - NCCLWeightSyncer handles NCCL init separately.
        if (
            config.checkpoint.weight_sync_mode == "nccl"
            and config.checkpoint.pipeline_mode != "true_pipeline"
        ):
            logger.info(f"Initializing NCCL weight sync with {num_engines} engine(s)...")
            init_fn = getattr(backend, "init_nccl_weight_sync", None)
            if init_fn is None:
                raise ValueError(
                    "weight_sync_mode='nccl' requires the selected trainer backend to implement "
                    "init_nccl_weight_sync()."
                )
            await init_fn(
                inference_endpoints=[e.base_url for e in inference_engines],
                master_port=config.checkpoint.nccl_master_port,
            )
            logger.info("NCCL weight sync initialized")

        # Setup data and rollout generation
        logger.info(f"Dataset: {len(prompts)} prompts")
        data_buffer = DataBuffer(prompts=prompts)
        generate_fn = _create_generate_fn(
            config, endpoint, tokenizer, environment_cls, metadata_key, logger
        )

        rollout_config = RolloutConfig(
            batch_size=config.rollout.batch_size,
            n_samples_per_prompt=config.rollout.n_samples_per_prompt,
            over_sampling_factor=1.0,
            generate_fn=generate_fn,
            score_fn=score_fn,
        )

        # Training loop (delegated to rollouts.training.train.train)
        from ..training.train import train as _train_loop

        # Step-level weight syncer (blocking). True PipelineRL uses non-blocking NCCLWeightSyncer instead.
        from ..training.weight_sync import BackendNCCLWeightSyncer, FilesystemWeightSyncer

        step_weight_syncer = None
        if config.checkpoint.pipeline_mode != "true_pipeline":
            if config.checkpoint.weight_sync_mode == "nccl":
                step_weight_syncer = BackendNCCLWeightSyncer(backend=backend, log=logger)
            elif config.checkpoint.weight_sync_mode == "disk":
                step_weight_syncer = FilesystemWeightSyncer(
                    backend=backend, engines=inference_engines
                )
            else:
                raise ValueError(
                    f"Unknown weight_sync_mode: {config.checkpoint.weight_sync_mode!r}. "
                    "Use 'disk' or 'nccl'."
                )

        async def _save_checkpoint(step: int, step_metrics: dict[str, Any]) -> Path:
            save_fn = getattr(backend, "save_checkpoint", None)
            if save_fn is None:
                raise ValueError(
                    "Checkpointing requires the selected trainer backend to implement "
                    "save_checkpoint(step, metrics) -> Path."
                )
            numeric_metrics = {
                k: float(v) for k, v in step_metrics.items() if isinstance(v, (int, float))
            }
            ckpt_dir = await save_fn(step, numeric_metrics)
            logger.info(f"Saved checkpoint: {ckpt_dir}")
            return ckpt_dir

        async def _sync_batches() -> AsyncIterator[Any]:
            # Synchronous training (default): generate batch, train, sync weights, repeat.
            async with AsyncRolloutManager(data_buffer, rollout_config) as rollout_manager:
                for _step in range(config.checkpoint.num_steps):
                    batch = await rollout_manager.generate_batch(score_fn=score_fn)

                    # Compute teacher logprobs for OPD (only supported in sync mode for now)
                    if teacher_engine is not None and batch.samples:
                        from ..training.opd import compute_teacher_logprobs_batch

                        teacher_url = teacher_engine.base_url
                        logger.info(
                            f"Computing teacher logprobs for {len(batch.samples)} samples..."
                        )
                        await compute_teacher_logprobs_batch(teacher_url, batch.samples)
                        # Update batch with teacher logprobs (samples were modified in-place)
                        # Filter out None values - all samples should have teacher_log_probs after compute
                        batch.teacher_log_probs = [
                            s.teacher_log_probs
                            for s in batch.samples
                            if s.teacher_log_probs is not None
                        ]
                        logger.info("Teacher logprobs computed")

                    yield batch

        if config.checkpoint.pipeline_mode == "true_pipeline":
            # True PipelineRL: both sampling AND weight sync are non-blocking.
            from ..training.rollout_gen.pipelined_rollout_manager import PipelinedRolloutManager
            from ..training.weight_sync import NCCLWeightSyncer

            weight_sync_manager = NCCLWeightSyncer(
                inference_endpoints=[e.base_url for e in inference_engines],
                max_lag=config.checkpoint.max_lag,
                nccl_master_port=config.checkpoint.nccl_master_port,
            )

            pipelined_manager = PipelinedRolloutManager(
                data_buffer=data_buffer,
                config=rollout_config,
                max_lag=config.checkpoint.max_lag,
                queue_size=config.checkpoint.pipeline_queue_size,
                # Pause sampling when weight sync is in progress (SGLang is blocked)
                sync_in_progress_fn=lambda: weight_sync_manager.sync_in_progress,
            )

            logger.info(
                f"Using TRUE PipelineRL mode (max_lag={config.checkpoint.max_lag}, "
                f"queue_size={config.checkpoint.pipeline_queue_size}, "
                f"engines={num_engines})"
            )
            logger.info("  - Background sampling: ON (inference never stops)")
            logger.info("  - Non-blocking weight sync: ON (training never waits)")
            if num_engines > 1:
                logger.info(f"  - Multi-engine: {num_engines} inference servers")

            async def _true_pipeline_batches() -> AsyncIterator[Any]:
                try:
                    # Initialize NCCL for non-blocking weight sync
                    await weight_sync_manager.init_nccl_group()

                    async with pipelined_manager:
                        async with trio.open_nursery() as nursery:
                            # Start background sampling
                            await pipelined_manager.start_sampling(
                                nursery=nursery,
                                initial_weight_version=weight_sync_manager.current_version,
                            )

                            for step in range(config.checkpoint.num_steps):
                                batch = await pipelined_manager.get_batch(
                                    current_weight_version=weight_sync_manager.current_version,
                                    score_fn=score_fn,
                                )
                                yield batch

                                # Non-blocking weight sync - spawns background task.
                                should_sync = (step + 1) % config.checkpoint.sync_weights_every == 0
                                if should_sync:
                                    logger.debug(
                                        f"Spawning async weight sync (v={weight_sync_manager.current_version + 1})"
                                    )
                                    await weight_sync_manager.broadcast_weights_async(
                                        backend.model, nursery
                                    )

                                # Update version in rollout manager
                                pipelined_manager.update_weight_version(
                                    weight_sync_manager.current_version
                                )

                            # Log pipeline stats
                            stats = pipelined_manager.stats()
                            logger.info(
                                f"Pipeline stats: generated={stats['samples_generated']}, "
                                f"discarded_stale={stats['samples_discarded_stale']} "
                                f"({stats['discard_rate']:.1f}%)"
                            )
                finally:
                    await weight_sync_manager.close()

            async def _process_batch(step: int, batch: Any, _backend: Any) -> dict[str, Any] | None:
                return await _process_training_step(
                    step,
                    batch,
                    config,
                    backend,
                    tokenizer,
                    device,
                    output_dir,
                    logger,
                    run_context,
                )

            train_result = await _train_loop(
                config=config.checkpoint,
                backend=backend,
                batch_iterator=_true_pipeline_batches(),
                process_batch=_process_batch,
                weight_syncer=None,
                save_checkpoint=_save_checkpoint,
                metrics_logger=metrics_logger,
                logger=logger,
            )

        elif config.checkpoint.pipeline_mode == "async":
            # Async sampling but blocking weight sync (training waits for sync).
            from ..training.rollout_gen.pipelined_rollout_manager import PipelinedRolloutManager

            pipelined_manager = PipelinedRolloutManager(
                data_buffer=data_buffer,
                config=rollout_config,
                max_lag=config.checkpoint.max_lag,
                queue_size=config.checkpoint.pipeline_queue_size,
            )

            logger.info(
                f"Using async pipeline (max_lag={config.checkpoint.max_lag}, "
                f"queue_size={config.checkpoint.pipeline_queue_size})"
            )

            async def _async_pipeline_batches() -> AsyncIterator[Any]:
                async with pipelined_manager:
                    async with trio.open_nursery() as nursery:
                        await pipelined_manager.start_sampling(
                            nursery=nursery,
                            initial_weight_version=backend.weight_version,
                        )

                        for _step in range(config.checkpoint.num_steps):
                            batch = await pipelined_manager.get_batch(
                                current_weight_version=backend.weight_version,
                                score_fn=score_fn,
                            )
                            yield batch
                            pipelined_manager.update_weight_version(backend.weight_version)

                        # Log pipeline stats
                        stats = pipelined_manager.stats()
                        logger.info(
                            f"Pipeline stats: generated={stats['samples_generated']}, "
                            f"discarded_stale={stats['samples_discarded_stale']} "
                            f"({stats['discard_rate']:.1f}%)"
                        )

            async def _process_batch(step: int, batch: Any, _backend: Any) -> dict[str, Any] | None:
                return await _process_training_step(
                    step,
                    batch,
                    config,
                    backend,
                    tokenizer,
                    device,
                    output_dir,
                    logger,
                    run_context,
                )

            train_result = await _train_loop(
                config=config.checkpoint,
                backend=backend,
                batch_iterator=_async_pipeline_batches(),
                process_batch=_process_batch,
                weight_syncer=step_weight_syncer,
                save_checkpoint=_save_checkpoint,
                metrics_logger=metrics_logger,
                logger=logger,
            )

        else:

            async def _process_batch(step: int, batch: Any, _backend: Any) -> dict[str, Any] | None:
                return await _process_training_step(
                    step,
                    batch,
                    config,
                    backend,
                    tokenizer,
                    device,
                    output_dir,
                    logger,
                    run_context,
                )

            train_result = await _train_loop(
                config=config.checkpoint,
                backend=backend,
                batch_iterator=_sync_batches(),
                process_batch=_process_batch,
                weight_syncer=step_weight_syncer,
                save_checkpoint=_save_checkpoint,
                metrics_logger=metrics_logger,
                logger=logger,
            )

        metrics_history = train_result.metrics_history

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("Training Complete")
        logger.info("=" * 60)

        if metrics_history:
            first_reward = metrics_history[0]["mean_reward"]
            last_reward = metrics_history[-1]["mean_reward"]
            first_loss = metrics_history[0].get("pg_loss", 0.0)
            last_loss = metrics_history[-1].get("pg_loss", 0.0)
            logger.info(f"First: reward={first_reward:.3f}, pg_loss={first_loss:.4f}")
            logger.info(f"Last:  reward={last_reward:.3f}, pg_loss={last_loss:.4f}")

        return {"metrics_history": metrics_history}

    finally:
        # Cleanup NCCL weight sync if it was initialized
        if config.checkpoint.weight_sync_mode == "nccl":
            try:
                cleanup_fn = getattr(backend, "cleanup_nccl_weight_sync", None)
                if cleanup_fn is not None:
                    await cleanup_fn()
            except NameError:
                pass  # backend not yet created
            except Exception as e:
                logger.warning(f"NCCL cleanup failed: {e}")

        if backend_cleanup is not None:
            try:
                backend_cleanup()
            except Exception as e:
                logger.warning(f"Backend cleanup failed: {e}")

        for engine in inference_engines:
            logger.info(f"Shutting down {engine.name} (port={engine.port})...")
            engine.shutdown()
            logger.info(f"Logs: {engine.log_path}")

        # Shutdown teacher engine if it was launched
        if teacher_engine is not None:
            logger.info("Shutting down teacher engine...")
            teacher_engine.shutdown()


# ──────────────────────── TI/TO Helpers ───────────────────────────────────────

# Moved to rollouts.training.tito (kept as aliases for backward compatibility).
