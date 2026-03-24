"""GRPO Training Loop.

Shared training infrastructure for GRPO (Group Relative Policy Optimization).
Each task provides prompts, a scorer, and environment_cls - this module handles
the rest: SGLang server, training backend, rollout generation, gradient updates.

Usage:
    from ..training.grpo import GRPOConfig, grpo_train

    config = GRPOConfig(model_name="Qwen/Qwen3-0.6B", num_steps=100)
    prompts = [{"messages": [...], "answer": "42"}, ...]

    class MyScorer:
        async def score(self, result, context):
            return Score(metrics=(Metric("correct", 1.0 if correct else 0.0, weight=1.0),))

    results = grpo_train(
        config=config,
        prompts=prompts,
        scorer=MyScorer(),
        environment_cls=BasicEnvironment,
    )
"""

from __future__ import annotations

import logging
import socket
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from ..core import Environment
    from ..training.types import Scorer

# ──────────────────────── Sub-Configs (re-exported from shared) ───────────────

from ..resource_watchdog import ResourceWatchdog, ResourceWatchdogConfig
from ..run_logger import RunLogger
from ..training.configs import (  # noqa: E402
    CheckpointConfig,
    DepsConfig,
    InferenceConfig,
    ModelConfig,
    OutputConfig,
    RolloutConfig,
    TrainerConfig,
    deps_config_from_data,
    megatron_overrides_from_data,
)
from ..training.inference_runtime_factory import create_inference_backend_runtime
from ..training.runtime_factory import (
    build_megatron_lowering,
    create_training_backend_runtime,
    resolve_megatron_batch_realization,
)
from ..training.types import RolloutRuntime


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

    TODO(worker-topology): `trainer` + `inference` are the current compatibility
    surface. The cleaner denotation is:
    - allocate hardware once
    - realize named inference/training workers on that allocation
    - bind semantic roles like actor/judge/teacher to those workers
    Keep new shared authoring surfaces additive until GRPO can consume the
    worker graph natively without hiding launch ownership in runtime glue.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    runtime_watchdog: ResourceWatchdogConfig = field(default_factory=ResourceWatchdogConfig)
    output: OutputConfig = field(
        default_factory=lambda: OutputConfig(output_dir="results/rl", experiment_name="grpo")
    )
    service_runtime_layout: str = "shared_env"

    def __post_init__(self) -> None:
        if self.trainer.backend == "megatron" and self.checkpoint.weight_sync_mode != "nccl":
            raise ValueError(
                "Megatron training currently supports only checkpoint.weight_sync_mode='nccl'. "
                "The Megatron backend performs direct weight sync to inference and does not "
                "implement disk checkpoint sync semantics for per-step sampler updates yet."
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
        inference_data = dict(data.get("inference", {}))
        trainer_data = dict(data.get("trainer", {}))
        inference_data["deps"] = deps_config_from_data(inference_data.get("deps"))
        trainer_data["deps"] = deps_config_from_data(trainer_data.get("deps"))
        trainer_data["megatron_overrides"] = megatron_overrides_from_data(
            trainer_data.get("megatron_overrides")
        )
        inference = InferenceConfig(**inference_data)
        trainer = TrainerConfig(**trainer_data)
        rollout = RolloutConfig(**data.get("rollout", {}))
        checkpoint = CheckpointConfig(**data.get("checkpoint", {}))
        runtime_watchdog = ResourceWatchdogConfig(**data.get("runtime_watchdog", {}))
        output = OutputConfig(**data.get("output", {}))

        return GRPOConfig(
            model=model,
            inference=inference,
            trainer=trainer,
            rollout=rollout,
            checkpoint=checkpoint,
            runtime_watchdog=runtime_watchdog,
            output=output,
            service_runtime_layout=data.get("service_runtime_layout", "shared_env"),
        )

    def trainer_deps(self) -> DepsConfig | None:
        """Runtime deps owned by the trainer service."""
        return self.trainer.deps

    def inference_deps(self) -> DepsConfig | None:
        """Runtime deps owned by the inference service."""
        return self.inference.deps


def _megatron_lowering(config: GRPOConfig) -> Any:
    return build_megatron_lowering(config.trainer, training_mode="rl")


def _rl_training_global_batch_size(config: GRPOConfig) -> int:
    return config.rollout.batch_size * config.rollout.n_samples_per_prompt


# ──────────────────────── Training Function ──────────────────────────────────


def grpo_train(
    config: GRPOConfig,
    prompts: list[dict[str, Any]],
    scorer: Scorer | None = None,
    environment_cls: Callable[[], Environment] | type[Environment] | None = None,
    metadata_key: str | None = None,
    environment_factory: Callable[[dict[str, Any]], Any] | None = None,
    run_logger: Any | None = None,
) -> dict[str, Any]:
    """Run GRPO training.

    Args:
        config: Training configuration
        prompts: List of prompt dicts, each containing:
            - "messages": List of chat messages [{"role": "...", "content": "..."}]
            - Any metadata needed by the scorer (e.g., "answer", "expected_sorted")
        scorer: Explicit scoring stage over raw attempt results.
        environment_cls: Zero-arg environment constructor for simple cases
            (BasicEnvironment, CalculatorEnvironment, factory function, etc.).
        environment_factory: Optional per-sample environment factory. Receives the
            original prompt/sample dict and may be sync or async.
        run_logger: Optional structured run logger from the outer runner. GRPO
            does not consume it yet directly; this exists so config wrappers can
            forward runner kwargs without lying about the call boundary.
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
        >>> results = grpo_train(config, prompts, MyScorer(), BasicEnvironment)
    """
    if scorer is None:
        raise ValueError("grpo_train requires an explicit scorer")
    return trio.run(
        _grpo_train_async,
        config,
        prompts,
        scorer,
        environment_cls,
        metadata_key,
        environment_factory,
        run_logger,
    )


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
        disable_cuda_graph=config.inference.disable_cuda_graph,
        max_total_tokens=config.inference.max_total_tokens,
        max_prefill_tokens=config.inference.max_prefill_tokens,
        max_running_requests=config.inference.max_running_requests,
        chunked_prefill_size=config.inference.chunked_prefill_size,
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
    emit_phase: Callable[[str], None] | None = None,
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
    from ..training.losses import grpo_loss, grpo_loss_clipped, grpo_loss_masked, opd_loss

    # Select loss function based on config
    loss_fn = _make_loss_fn(
        config.trainer, grpo_loss, grpo_loss_clipped, grpo_loss_masked, opd_fn=opd_loss
    )

    backend, cleanup = create_training_backend_runtime(
        model=config.model,
        trainer=config.trainer,
        checkpoint=config.checkpoint,
        output_dir=output_dir,
        seq_len=config.rollout.max_seq_len,
        global_batch_size=_rl_training_global_batch_size(config),
        loss_fn=loss_fn,
        training_mode="rl",
        megatron_workers=megatron_workers,
        megatron_inference_endpoints=(f"http://localhost:{config.inference.port}",),
        emit_phase=emit_phase,
    )

    tokenizer, endpoint = _build_training_client_surface(config, inference_engine)
    return backend, tokenizer, endpoint, cleanup


def _create_generate_fn(
    config: GRPOConfig,
    endpoint: Any,
    tokenizer: Any,
    environment_cls: Callable[[], Environment] | type[Environment] | None,
    environment_factory: Callable[[dict[str, Any]], Any] | None,
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
        config, endpoint, tokenizer, environment_cls, environment_factory, metadata_key, logger
    )


def _create_agent_generate_fn(
    config: GRPOConfig,
    endpoint: Any,
    tokenizer: Any,
    environment_cls: Callable[[], Environment] | type[Environment] | None,
    environment_factory: Callable[[dict[str, Any]], Any] | None,
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
                    environment_factory=environment_factory,
                    sample_data=prompt_data,
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
        "inference_realization": config.inference.realization or config.inference.backend,
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


def _build_training_preflight_datum(config: GRPOConfig, device: str) -> Any:
    """Build a cheap synthetic datum for backend health checks.

    This is not a VRAM worst-case probe. It exists to exercise the real
    backend/model/provider/optimizer path before we pay inference startup cost.
    """
    import torch

    from ..training.contracts import ModelInput, TrainableParameterPolicy, TrainingDatum

    del device

    micro_batch_size = config.trainer.micro_batch_size or 1
    seq_len = min(config.rollout.max_seq_len, 32)
    vocab_size = 1024

    tokens = torch.randint(0, vocab_size, (micro_batch_size, seq_len))
    labels = tokens.clone()
    loss_mask = torch.ones(micro_batch_size, seq_len)
    advantages = torch.ones(micro_batch_size)
    group_ids = torch.arange(micro_batch_size, dtype=torch.long)

    return TrainingDatum(
        model_input=ModelInput(tokens=tokens),
        objective_inputs={
            "labels": labels,
            "loss_mask": loss_mask,
            "advantages": advantages,
            "group_ids": group_ids,
        },
        trainable_parameter_policy=TrainableParameterPolicy.full_weight(),
        metadata={"synthetic": True, "preflight": "training_backend"},
    )


def _build_megatron_preflight_batch(config: GRPOConfig) -> dict[str, Any]:
    """Build a backend-native Megatron synthetic batch."""
    import torch

    micro_batch_size = config.trainer.micro_batch_size or 1
    seq_len = min(config.rollout.max_seq_len, 32)
    vocab_size = 1024

    input_ids = torch.randint(0, vocab_size, (micro_batch_size, seq_len))
    batch = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "loss_mask": torch.ones(micro_batch_size, seq_len),
        "advantages": torch.ones(micro_batch_size),
        "group_ids": torch.arange(micro_batch_size, dtype=torch.long),
    }
    if config.trainer.loss_type in {"clipped", "masked"}:
        batch["old_logprobs"] = torch.zeros(micro_batch_size)
    if config.trainer.loss_type == "opd":
        batch["teacher_logprobs"] = torch.zeros(micro_batch_size, seq_len)
    return batch


def _build_training_client_surface(config: GRPOConfig, inference_engine: Any) -> tuple[Any, Any]:
    """Build tokenizer + inference endpoint for training rollouts."""
    from transformers import AutoTokenizer

    from ..dtypes import Endpoint

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    endpoint = Endpoint(
        model=f"openai/{config.model.name}",
        base_url=inference_engine.api_base,
        api_format="openai-completions",
        temperature=config.rollout.temperature,
        max_tokens=config.rollout.max_tokens,
        extra_params=config.rollout.extra_params or None,
    )
    return tokenizer, endpoint


def _megatron_worker_snapshot(backend: Any) -> list[dict[str, Any]]:
    workers = getattr(backend, "workers", None)
    if not isinstance(workers, list):
        return []
    snapshot: list[dict[str, Any]] = []
    for worker in workers:
        pid = getattr(worker, "pid", None)
        try:
            alive = bool(worker.is_alive())
        except Exception as exc:
            alive = False
            snapshot.append({
                "pid": pid,
                "alive": alive,
                "state_error": f"{type(exc).__name__}: {exc}",
            })
            continue
        snapshot.append({"pid": pid, "alive": alive})
    return snapshot


async def _abort_failed_megatron_witness_backend(
    backend: Any,
    logger: logging.Logger,
    run_context: dict[str, Any],
    *,
    reason: str,
) -> None:
    import os
    import signal

    workers = getattr(backend, "workers", None)
    if not isinstance(workers, list) or not workers:
        logger.warning(
            "training_preflight_weight_sync_witness_abort_skipped",
            extra={
                "event": "training_preflight_weight_sync_witness_abort_skipped",
                **run_context,
                "reason": reason,
                "abort_kind": "no_workers",
            },
        )
        return

    before = _megatron_worker_snapshot(backend)
    logger.warning(
        "training_preflight_weight_sync_witness_abort_start",
        extra={
            "event": "training_preflight_weight_sync_witness_abort_start",
            **run_context,
            "reason": reason,
            "workers_before": before,
        },
    )

    try:
        shutdown = getattr(backend, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as exc:
        logger.warning(
            "training_preflight_weight_sync_witness_abort_shutdown_failed",
            extra={
                "event": "training_preflight_weight_sync_witness_abort_shutdown_failed",
                **run_context,
                "reason": reason,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

    for worker in workers:
        pid = getattr(worker, "pid", None)
        if pid is None:
            continue
        try:
            if worker.is_alive():
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    await trio.sleep(0.5)

    for worker in workers:
        pid = getattr(worker, "pid", None)
        if pid is None:
            continue
        try:
            if worker.is_alive():
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            worker.close()
        except Exception:
            pass

    await trio.sleep(0.1)
    after = _megatron_worker_snapshot(backend)
    logger.warning(
        "training_preflight_weight_sync_witness_abort_complete",
        extra={
            "event": "training_preflight_weight_sync_witness_abort_complete",
            **run_context,
            "reason": reason,
            "workers_before": before,
            "workers_after": after,
        },
    )


async def _run_training_preflight(
    config: GRPOConfig,
    output_dir: Path,
    logger: logging.Logger,
    *,
    megatron_workers: list[Any] | None = None,
    node_id: str | None = None,
    run_context: dict[str, Any] | None = None,
    run_logger: RunLogger | None = None,
) -> tuple[Any | None, Callable[[], None] | None]:
    """Initialize the training backend and run one synthetic step.

    This is a backend-health preflight, not a full startup or VRAM truth probe.
    Its job is to fail before expensive inference startup if training is broken.
    """
    from types import SimpleNamespace

    from ..training.contract_witnesses import rl_contract_loss

    rc = run_context or {}
    runtime_run_logger = run_logger if isinstance(run_logger, RunLogger) else None
    resolved_node_id = node_id or rc.get("node_id")
    backend_name = config.trainer.backend

    def _emit_preflight_event(event: str, **data: Any) -> None:
        logger.info(
            event,
            extra={
                "event": event,
                **rc,
                "node_id": resolved_node_id,
                "backend": backend_name,
                **data,
            },
        )
        if runtime_run_logger is not None:
            event_payload = {
                **rc,
                "backend": backend_name,
                **data,
            }
            if "node_id" not in event_payload:
                event_payload["node_id"] = resolved_node_id
            runtime_run_logger.event(
                event,
                **event_payload,
            )

    _emit_preflight_event("training_preflight_start")

    dummy_engine = SimpleNamespace(api_base=f"http://127.0.0.1:{config.inference.port}/v1")
    preflight_output_dir = output_dir / "_training_preflight"
    preflight_output_dir.mkdir(parents=True, exist_ok=True)

    backend: Any | None = None
    cleanup: Callable[[], None] | None = None
    reusable_backend: Any | None = None
    reusable_cleanup: Callable[[], None] | None = None
    try:
        _emit_preflight_event("training_preflight_backend_init_start")
        backend, _tokenizer, _endpoint, cleanup = _setup_training_backend(
            config,
            preflight_output_dir,
            dummy_engine,
            megatron_workers=megatron_workers,
            emit_phase=_emit_preflight_event,
        )

        _emit_preflight_event("training_preflight_backend_init_ok")

        if backend_name == "megatron":
            if config.trainer.validate_inference_export_in_preflight:
                validate_inference_export = getattr(backend, "validate_inference_export", None)
                assert callable(validate_inference_export), (
                    "Megatron backend must expose validate_inference_export()"
                )
                _emit_preflight_event("training_preflight_inference_export_start")
                export_validation = await validate_inference_export().result()
                _emit_preflight_event(
                    "training_preflight_inference_export_ok",
                    tensor_count=export_validation.get("tensor_count"),
                )
            else:
                _emit_preflight_event("training_preflight_inference_export_skipped")
            preflight_step = getattr(backend, "preflight_step", None)
            assert callable(preflight_step), "Megatron backend must expose preflight_step()"
            _emit_preflight_event("training_preflight_synthetic_step_start")
            fb_result = await preflight_step(_build_megatron_preflight_batch(config)).result()
            optim_result = None
            if hasattr(backend, "checkpoint_dir"):
                backend.checkpoint_dir = output_dir
            reusable_backend = backend
            reusable_cleanup = cleanup
            cleanup = None
        else:
            device = f"cuda:{config.trainer.cuda_device_ids[0]}"
            datum = _build_training_preflight_datum(config, device)
            _emit_preflight_event("training_preflight_synthetic_step_start")
            fb_future = backend.forward_backward(datum, loss_fn=rl_contract_loss)
            fb_result = await fb_future.result()
            optim_future = backend.optim_step()
            optim_result = await optim_future.result()
            if backend_name == "torchtitan":
                reusable_backend = backend
                reusable_cleanup = cleanup
                cleanup = None

        _emit_preflight_event(
            "training_preflight_synthetic_step_ok",
            losses=getattr(fb_result, "losses", {}),
            optim=optim_result,
        )
    finally:
        if cleanup is not None:
            cleanup()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return reusable_backend, reusable_cleanup


def _attach_runtime_observability(
    batch: Any,
    rollout_manager: Any,
    scorer: Any,
    environment_factory: Any = None,
) -> None:
    """Attach explicit runtime stats to batch metadata for downstream logging."""
    if hasattr(rollout_manager, "stats"):
        batch.metadata["rollout_stats"] = rollout_manager.stats()
    if scorer is not None and hasattr(scorer, "stats"):
        batch.metadata["scorer_stats"] = scorer.stats()
    if environment_factory is not None and hasattr(environment_factory, "stats"):
        batch.metadata["environment_stats"] = environment_factory.stats()


async def _maybe_start_environment_factory(
    environment_factory: Any, logger: logging.Logger
) -> None:
    if environment_factory is None:
        return
    start = getattr(environment_factory, "start", None)
    if callable(start):
        logger.info("Starting rollout environment resources...")
        await start()


async def _maybe_stop_environment_factory(environment_factory: Any, logger: logging.Logger) -> None:
    if environment_factory is None:
        return
    stop = getattr(environment_factory, "stop", None)
    if callable(stop):
        logger.info("Stopping rollout environment resources...")
        await stop()


async def _pause_pipeline_admissions(pipelined_manager: Any, logger: logging.Logger) -> None:
    """Pause new rollout admissions before a blocking weight-sync boundary."""
    pipelined_manager.pause_new_admissions("weight_sync")
    logger.debug("Paused new rollout admissions for weight sync")
    await trio.lowlevel.checkpoint()


async def _resume_pipeline_admissions(pipelined_manager: Any, logger: logging.Logger) -> None:
    """Resume rollout admissions after a blocking weight-sync boundary."""
    pipelined_manager.resume_new_admissions()
    logger.debug("Resumed rollout admissions after weight sync")
    await trio.lowlevel.checkpoint()


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

    from ..training.contracts import VersionedRolloutBatch
    from ..training.losses import compute_group_advantages
    from ..training.observability import flatten_numeric_stats

    step_start = time.perf_counter()
    batch_weight_version = None
    batch_version_lag = None
    if isinstance(batch, VersionedRolloutBatch):
        batch_weight_version = batch.weight_version
        batch_version_lag = batch.version_lag
        batch = batch.batch

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

    if config.trainer.advantage_estimator == "opd":
        raise NotImplementedError(
            "Contract-native GRPO training step does not yet implement OPD teacher semantics. "
            "Lower OPD onto an explicit distillation-aware objective before using this path."
        )

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

    from ..training.contracts import StepResult

    # Prepare batch tensors
    prep_start = time.perf_counter()
    training_batch = _prepare_training_batch(batch, config, tokenizer, advantages, device)
    prep_ms = (time.perf_counter() - prep_start) * 1000

    # Training step - forward/backward
    fb_start = time.perf_counter()
    if config.trainer.backend == "megatron":
        fb_future = backend.forward_backward(training_batch)
    else:
        from ..training.contract_witnesses import rl_contract_loss

        fb_future = backend.forward_backward(training_batch, loss_fn=rl_contract_loss)
    fb_result = await fb_future.result()
    fb_ms = (time.perf_counter() - fb_start) * 1000

    # Optimizer step
    optim_start = time.perf_counter()
    optim_future = backend.optim_step()
    optim_metrics = await optim_future.result()
    optim_ms = (time.perf_counter() - optim_start) * 1000

    if isinstance(fb_result, StepResult):
        accumulated_metrics = {**fb_result.losses, **fb_result.other_metrics, **optim_metrics}
    else:
        accumulated_metrics = {**fb_result, **optim_metrics}
    pg_loss = accumulated_metrics.get("pg_loss", 0.0)
    entropy = accumulated_metrics.get("entropy", 0.0)

    step_metrics = {
        "mean_reward": mean_reward,
        "num_samples": len(rewards),
        "num_groups": num_groups,
        **accumulated_metrics,
    }
    if batch_weight_version is not None:
        step_metrics["batch_weight_version"] = float(batch_weight_version)
    if batch_version_lag is not None:
        step_metrics["batch_version_lag"] = float(batch_version_lag)
    rollout_stats = batch.metadata.get("rollout_stats")
    rollout_observability: dict[str, float] = {}
    if isinstance(rollout_stats, dict):
        rollout_observability = flatten_numeric_stats(rollout_stats, prefix="rollout_")
        step_metrics.update(rollout_observability)
    scorer_stats = batch.metadata.get("scorer_stats")
    scorer_observability: dict[str, float] = {}
    if isinstance(scorer_stats, dict):
        scorer_observability = flatten_numeric_stats(scorer_stats, prefix="scorer_")
        step_metrics.update(scorer_observability)
    environment_stats = batch.metadata.get("environment_stats")
    environment_observability: dict[str, float] = {}
    if isinstance(environment_stats, dict):
        environment_observability = flatten_numeric_stats(environment_stats, prefix="environment_")
        step_metrics.update(environment_observability)

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
            **rollout_observability,
            **scorer_observability,
        },
    )

    return step_metrics


def _prepare_training_batch(
    batch: Any,
    config: GRPOConfig,
    tokenizer: Any,
    advantages: Any,
    device: str,
) -> Any:
    """Prepare tensors for training step."""
    import torch

    from ..training.contracts import ModelInput, TrainableParameterPolicy, TrainingDatum

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

    objective_inputs: dict[str, Any] = {
        "labels": labels,
        "loss_mask": loss_mask,
        "advantages": advantages,
        "group_ids": torch.tensor(batch.group_indices, device=device, dtype=torch.long),
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
        objective_inputs["old_logprobs"] = seq_rollout_logprobs

    if has_teacher_logprobs:
        teacher_logprobs_tensor = torch.tensor(batch_teacher_logprobs, device=device)
        # Shift teacher_logprobs left to match shifted labels/loss_mask
        teacher_logprobs_tensor = torch.cat(
            [teacher_logprobs_tensor[:, 1:], torch.zeros_like(teacher_logprobs_tensor[:, :1])],
            dim=1,
        )
        objective_inputs["teacher_logprobs"] = teacher_logprobs_tensor

    return TrainingDatum(
        model_input=ModelInput(tokens=input_ids),
        objective_inputs=objective_inputs,
        trainable_parameter_policy=TrainableParameterPolicy.full_weight(),
        metadata=batch.metadata,
    )


async def _grpo_train_async(
    config: GRPOConfig,
    prompts: list[dict[str, Any]],
    scorer: Scorer,
    environment_cls: Callable[[], Environment] | type[Environment] | None,
    metadata_key: str | None = None,
    environment_factory: Callable[[dict[str, Any]], Any] | None = None,
    run_logger: Any | None = None,
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
    logger.info(f"Inference: {config.inference.realization or config.inference.backend}")
    logger.info(f"Steps: {config.checkpoint.num_steps}")
    logger.info(
        f"Batch: {config.rollout.batch_size} prompts x {config.rollout.n_samples_per_prompt} samples"
    )
    logger.info(f"Output: {output_dir}")

    # Preflight check: validate config against hardware limits
    # This fails fast if config is likely to OOM
    # Only run on rank 0 to avoid duplicate checks in torchrun/DDP
    # Skip for torchtitan/megatron - they have their own sharding
    from ..training.preflight import run_preflight_check

    def _get_gpu_name_subprocess() -> str | None:
        """Get GPU name in subprocess to avoid polluting main process CUDA context.

        CUDA contexts don't survive fork() - if we init CUDA here then fork workers,
        the workers will crash with 'Cannot re-initialize CUDA in forked subprocess'.
        Running in a subprocess lets us probe the GPU safely.
        """
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-c",
                "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend_name = getattr(config.trainer, "backend", "pytorch")
    if local_rank == 0 and backend_name != "torchtitan":
        try:
            # Detect GPU type from CUDA device in subprocess to avoid polluting
            # main process CUDA context (which would break fork-based workers)
            gpu_name = _get_gpu_name_subprocess()
            if gpu_name:
                run_preflight_check(config, gpu_name)
            else:
                logger.warning("CUDA not available, skipping preflight check")
        except ValueError as e:
            logger.exception(f"Preflight check failed: {e}")
            raise
    elif backend_name == "torchtitan":
        from .preflight import preflight_torchtitan_runtime

        runtime_preflight = preflight_torchtitan_runtime()
        if runtime_preflight.ok:
            logger.info(
                "TorchTitan runtime preflight passed: stage=%s torch=%s",
                runtime_preflight.stage,
                runtime_preflight.details.get("torch_version"),
            )
        else:
            details = runtime_preflight.details or {}
            logger.error(
                "TorchTitan runtime preflight failed: stage=%s error=%s details=%s",
                runtime_preflight.stage,
                runtime_preflight.error,
                details,
            )
            runtime_preflight.require_ok()

    config.save(output_dir / "config.json")
    metrics_logger = JSONLLogger(output_dir)
    runtime_run_logger = (
        run_logger if isinstance(run_logger, RunLogger) else RunLogger(text_logger=logger)
    )
    base_run_context = {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "model_name": config.model.name,
        "trainer_backend": config.trainer.backend,
        "inference_backend": config.inference.backend,
        "inference_realization": config.inference.realization or config.inference.backend,
        "node_id": os.environ.get("ROLLOUTS_NODE_ID"),
        "hostname": socket.gethostname(),
    }

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
        runtime_run_logger.event(
            "megatron_worker_spawn_start",
            **base_run_context,
            num_workers=num_trainer_gpus,
        )

        lowering = _megatron_lowering(config)
        megatron_overrides = config.trainer.megatron_overrides
        sequence_parallel = config.trainer.sequence_parallel
        if megatron_overrides is not None and megatron_overrides.sequence_parallel is not None:
            sequence_parallel = megatron_overrides.sequence_parallel
        global_batch_size = _rl_training_global_batch_size(config)
        micro_batch_size, num_microbatches = resolve_megatron_batch_realization(
            config.trainer,
            global_batch_size=global_batch_size,
        )

        # Create config for worker spawning (full config passed at init time)
        megatron_config = MegatronRemoteConfig(
            model_name=config.model.name,
            dtype=config.model.dtype,
            checkpoint_path=config.model.checkpoint_path,
            lowering=lowering,
            sequence_parallel=sequence_parallel,
            megatron_overrides=megatron_overrides,
            lr=config.trainer.lr,
            weight_decay=config.trainer.weight_decay,
            max_grad_norm=config.trainer.max_grad_norm,
            loss_type=config.trainer.loss_type,
            mask_ratio_low=config.trainer.mask_ratio_low,
            mask_ratio_high=config.trainer.mask_ratio_high,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            num_microbatches=num_microbatches,
            seq_length=config.trainer.seq_length,
            master_port=config.checkpoint.nccl_master_port,
            inference_endpoints=[f"http://localhost:{config.inference.port}"],
            cuda_device_ids=config.trainer.cuda_device_ids,
        )

        megatron_workers = spawn_megatron_workers(
            num_gpus=num_trainer_gpus,
            config=megatron_config,
        )
        logger.info(f"Spawned {len(megatron_workers)} megatron workers")
        runtime_run_logger.event(
            "megatron_worker_spawn_ok",
            **base_run_context,
            num_workers=len(megatron_workers),
        )

    preflight_backend: Any | None = None
    preflight_backend_cleanup: Callable[[], None] | None = None
    resource_watchdog = ResourceWatchdog(
        config=config.runtime_watchdog,
        run_logger=runtime_run_logger,
        run_context=base_run_context,
    )
    resource_watchdog.start()

    # Training preflight: initialize the backend and run one synthetic step
    # before paying inference startup cost. This is a backend-health check, not
    # a VRAM truth probe.
    resource_watchdog.set_phase("training_preflight")
    preflight_backend, preflight_backend_cleanup = await _run_training_preflight(
        config,
        output_dir,
        logger,
        megatron_workers=megatron_workers,
        node_id=os.environ.get("ROLLOUTS_NODE_ID"),
        run_context=base_run_context,
        run_logger=runtime_run_logger,
    )

    # Launch inference engine(s) - multi-engine for higher throughput
    # NOTE: This is when CUDA gets initialized (SGLang loads model on GPU 0)
    inference_runtime = create_inference_backend_runtime(
        model=config.model,
        inference=config.inference,
        rollout=config.rollout,
        checkpoint=config.checkpoint,
        output_dir=output_dir,
    )
    inference_engines = list(inference_runtime.engines)
    num_engines = len(inference_engines)
    run_context = _build_grpo_run_context(
        config=config,
        run_name=run_name,
        output_dir=output_dir,
        node_id=os.environ.get("ROLLOUTS_NODE_ID"),
        num_inference_engines=num_engines,
    )
    logger.info(
        "inference startup",
        extra={
            **run_context,
            "event": "inference_startup_start",
            "num_engines": num_engines,
            "ports": list(config.inference.ports),
            "gpu_assignments": [list(gpus) for gpus in config.inference.gpu_assignments],
            "inference_backend": config.inference.backend,
            "inference_realization": inference_runtime.realization.name,
            "inference_sync_realization": (
                inference_runtime.sync_realization.name
                if inference_runtime.sync_realization is not None
                else config.checkpoint.inference_sync_realization
            ),
            "inference_mem_fraction": config.inference.mem_fraction,
            "inference_tensor_parallel_size": config.inference.tensor_parallel_size,
            "inference_startup_timeout": config.inference.startup_timeout,
            "trainer_cuda_device_ids": list(config.trainer.cuda_device_ids),
            "inference_cuda_device_ids": list(config.inference.cuda_device_ids),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    runtime_run_logger.event(
        "inference_startup_start",
        **run_context,
        num_engines=num_engines,
        ports=list(config.inference.ports),
        gpu_assignments=[list(gpus) for gpus in config.inference.gpu_assignments],
        inference_sync_realization=(
            inference_runtime.sync_realization.name
            if inference_runtime.sync_realization is not None
            else config.checkpoint.inference_sync_realization
        ),
        inference_mem_fraction=config.inference.mem_fraction,
        inference_tensor_parallel_size=config.inference.tensor_parallel_size,
        inference_startup_timeout=config.inference.startup_timeout,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    resource_watchdog.set_phase(
        "inference_startup",
        num_engines=num_engines,
        ports=list(config.inference.ports),
    )

    if num_engines == 1:
        gpu_str = ",".join(str(g) for g in config.inference.cuda_device_ids)
        logger.info(f"Launching {inference_engines[0].name} on GPU {gpu_str}...")
    else:
        logger.info(f"Launching {num_engines} inference engines (PipelineRL-style)...")
        for i, engine in enumerate(inference_engines):
            gpu_str = ",".join(str(g) for g in config.inference.gpu_assignments[i])
            logger.info(f"  Engine {i}: {engine.name} on GPU {gpu_str}, port {engine.port}")

    for idx, engine in enumerate(inference_engines):
        launch_cmd = engine.build_launch_cmd() if hasattr(engine, "build_launch_cmd") else None
        session_name = getattr(engine, "session_name", None)
        log_path = str(getattr(engine, "log_path", "")) if hasattr(engine, "log_path") else None
        logger.info(
            "inference engine launch",
            extra={
                **run_context,
                "event": "inference_engine_launch",
                "engine_index": idx,
                "engine_name": engine.name,
                "engine_port": engine.port,
                "engine_cuda_device_ids": list(engine.cuda_device_ids),
                "engine_launch_cmd": launch_cmd,
                "engine_session_name": session_name,
                "engine_log_path": log_path,
                "engine_mem_fraction": getattr(engine, "mem_fraction", None),
                "engine_gpu_memory_utilization": getattr(engine, "gpu_memory_utilization", None),
            },
        )
        runtime_run_logger.event(
            "inference_engine_launch",
            **run_context,
            engine_index=idx,
            engine_name=engine.name,
            engine_port=engine.port,
            engine_cuda_device_ids=list(engine.cuda_device_ids),
            engine_launch_cmd=launch_cmd,
            engine_session_name=session_name,
            engine_log_path=log_path,
            engine_mem_fraction=getattr(engine, "mem_fraction", None),
            engine_gpu_memory_utilization=getattr(engine, "gpu_memory_utilization", None),
        )
        engine.launch()
        engine.start_log_tailer()

    # Primary engine for backward compat (endpoint creation uses first engine's base_url).
    # TODO(inference-routing): launching multiple inference engines is not the same
    # as routing rollout requests across them. Replace this single-engine binding
    # with an explicit multi-endpoint inference client/runtime plan.
    inference_engine = inference_engines[0]

    # Launch teacher engine for OPD (On-Policy Distillation)
    teacher_engine = _create_teacher_engine(config, output_dir)
    if teacher_engine is not None:
        teacher_gpu_str = ",".join(str(g) for g in teacher_engine.cuda_device_ids)
        logger.info(
            f"Launching teacher model ({config.trainer.teacher_model}) "
            f"on GPU {teacher_gpu_str}, port {teacher_engine.port}..."
        )
        logger.info(
            "teacher engine launch",
            extra={
                **run_context,
                "event": "teacher_inference_engine_launch",
                "engine_name": teacher_engine.name,
                "engine_port": teacher_engine.port,
                "engine_cuda_device_ids": list(teacher_engine.cuda_device_ids),
                "teacher_model": config.trainer.teacher_model,
            },
        )
        runtime_run_logger.event(
            "teacher_inference_engine_launch",
            **run_context,
            engine_name=teacher_engine.name,
            engine_port=teacher_engine.port,
            engine_cuda_device_ids=list(teacher_engine.cuda_device_ids),
            teacher_model=config.trainer.teacher_model,
        )
        teacher_engine.launch()
        teacher_engine.start_log_tailer()

    backend_cleanup: Callable[[], None] | None = None

    try:
        # Wait for all engines to be ready in parallel
        startup_timeout = config.inference.startup_timeout
        logger.info(
            "inference healthcheck start",
            extra={
                **run_context,
                "event": "inference_healthcheck_start",
                "startup_timeout": startup_timeout,
                "num_engines": num_engines,
                "teacher_engine": teacher_engine is not None,
            },
        )
        runtime_run_logger.event(
            "inference_healthcheck_start",
            **run_context,
            startup_timeout=startup_timeout,
            num_engines=num_engines,
            teacher_engine=teacher_engine is not None,
        )
        async with trio.open_nursery() as startup_nursery:
            for engine in inference_engines:
                startup_nursery.start_soon(engine.wait_until_ready, startup_timeout)
            if teacher_engine is not None:
                startup_nursery.start_soon(teacher_engine.wait_until_ready, startup_timeout)
        logger.info(f"All {num_engines} inference engine(s) ready")
        logger.info(
            "inference startup ready",
            extra={
                **run_context,
                "event": "inference_ready",
                "num_engines": num_engines,
                "teacher_engine": teacher_engine is not None,
            },
        )
        runtime_run_logger.event(
            "inference_ready",
            **run_context,
            num_engines=num_engines,
            teacher_engine=teacher_engine is not None,
        )
        if teacher_engine is not None:
            logger.info("Teacher engine ready")

        await _maybe_start_environment_factory(environment_factory, logger)

        if preflight_backend is not None:
            backend = preflight_backend
            tokenizer, endpoint = _build_training_client_surface(config, inference_engine)
            backend_cleanup = preflight_backend_cleanup
        else:
            # Setup training backend (pass pre-spawned workers for megatron)
            backend, tokenizer, endpoint, backend_cleanup = _setup_training_backend(
                config, output_dir, inference_engine, megatron_workers=megatron_workers
            )
        device = f"cuda:{config.trainer.cuda_device_ids[0]}"

        # Load checkpoint if provided (for SFT → RL pipeline)
        if config.model.checkpoint_path:
            ckpt_path = Path(config.model.checkpoint_path)
            megatron_tracker = ckpt_path / "latest_checkpointed_iteration.txt"
            if config.trainer.backend == "megatron" and megatron_tracker.exists():
                logger.info(
                    "Megatron checkpoint restore is owned by backend initialization: %s",
                    ckpt_path,
                )
            elif (ckpt_path / "pytorch_model.bin").exists():
                # Our checkpoint format
                logger.info(f"Loading checkpoint from {ckpt_path}")
                load_result = backend.load_checkpoint(ckpt_path)
                if hasattr(load_result, "result") and not hasattr(load_result, "__await__"):
                    await load_result.result()
                else:
                    await load_result
                logger.info("Checkpoint loaded successfully")
            elif (ckpt_path / "config.json").exists():
                # HuggingFace format - already loaded via model_name
                logger.info(f"Using HuggingFace checkpoint: {ckpt_path}")
            else:
                raise ValueError(
                    f"Invalid checkpoint path: {ckpt_path} (no pytorch_model.bin or config.json)"
                )

        # VRAM preflight: dry-run one forward+backward at worst-case seq_len.
        # This is backend-specific for now. The generic TrainingBackend
        # protocol does not promise direct access to `.optimizer` / `.loss_fn`,
        # so operational dry-runs must follow each backend's real surface.
        if not config.trainer.skip_vram_check and config.trainer.backend != "megatron":
            if config.trainer.backend == "torchtitan":
                from ..training.preflight import preflight_torchtitan_vram_check

                preflight_torchtitan_vram_check(backend, config, device)
            else:
                from ..training.vram_pytorch_like import preflight_vram_check

                preflight_vram_check(backend, config, device)
        else:
            reason = (
                "backend=megatron"
                if config.trainer.backend == "megatron"
                else "skip_vram_check=True"
            )
            logger.info(f"VRAM preflight check skipped ({reason})")

        # Initialize direct NCCL weight sync if enabled.
        # Skip for true_pipeline mode - NCCLWeightSyncer handles NCCL init separately.
        if (
            config.checkpoint.weight_sync_mode == "nccl"
            and config.checkpoint.pipeline_mode != "true_pipeline"
        ):
            init_fn = getattr(backend, "init_nccl_weight_sync", None)
            witness_fn = getattr(backend, "sync_weights_nccl_witness", None)
            megatron_uses_isolated_runtime_sync = config.trainer.backend == "megatron"
            if init_fn is None and not megatron_uses_isolated_runtime_sync:
                raise ValueError(
                    "weight_sync_mode='nccl' requires the selected trainer backend to implement "
                    "init_nccl_weight_sync()."
                )
            runtime_init_after_witness = False
            if megatron_uses_isolated_runtime_sync:
                logger.info(
                    "Skipping direct Megatron NCCL runtime init; live sync uses the "
                    f"isolated sender helper for {num_engines} engine(s)."
                )
            else:
                logger.info(f"Initializing NCCL weight sync with {num_engines} engine(s)...")
                assert init_fn is not None
                await init_fn(
                    inference_endpoints=[e.base_url for e in inference_engines],
                    # Weight sync uses its own rendezvous group. Reusing the Megatron
                    # training port is a real port collision, not a backend quirk.
                    # Start probing above the training port so the second distributed
                    # effect stays disjoint from the main process-group rendezvous.
                    master_port=config.checkpoint.nccl_master_port + 50,
                )
                logger.info("NCCL weight sync initialized")
            if megatron_uses_isolated_runtime_sync and callable(witness_fn):
                logger.info(
                    "training_preflight_weight_sync_runtime_init_skipped",
                    extra={
                        "event": "training_preflight_weight_sync_runtime_init_skipped",
                        **run_context,
                        "node_id": run_context.get("node_id"),
                        "backend": config.trainer.backend,
                        "reason": "megatron_isolated_runtime_sync",
                    },
                )
            if callable(witness_fn):
                inference_log_paths = [
                    str(getattr(engine, "log_path", ""))
                    for engine in inference_engines
                    if hasattr(engine, "log_path")
                ]
                inference_log_tails = {}
                resource_watchdog.set_phase("weight_sync_witness", tensor_limit=1)
                logger.info(
                    "training_preflight_weight_sync_witness_start",
                    extra={
                        "event": "training_preflight_weight_sync_witness_start",
                        **run_context,
                        "node_id": run_context.get("node_id"),
                        "backend": config.trainer.backend,
                        "tensor_limit": 1,
                        "inference_log_paths": inference_log_paths,
                    },
                )
                try:
                    await witness_fn(tensor_limit=1)
                except Exception as exc:
                    for engine in inference_engines:
                        log_path = getattr(engine, "log_path", None)
                        if log_path is None:
                            continue
                        try:
                            path = Path(log_path)
                            if path.exists():
                                tail = path.read_text(errors="replace").splitlines()[-20:]
                                inference_log_tails[str(path)] = (
                                    "\n".join(tail) if tail else "<log file empty>"
                                )
                            else:
                                inference_log_tails[str(path)] = "<log file not found>"
                        except Exception as tail_exc:
                            inference_log_tails[str(log_path)] = (
                                f"<failed to read log tail: {type(tail_exc).__name__}: {tail_exc}>"
                            )
                    if config.trainer.backend == "megatron":
                        await _abort_failed_megatron_witness_backend(
                            backend,
                            logger,
                            run_context,
                            reason="weight_sync_witness_failed",
                        )
                    logger.exception(
                        "training_preflight_weight_sync_witness_failed",
                        extra={
                            "event": "training_preflight_weight_sync_witness_failed",
                            **run_context,
                            "node_id": run_context.get("node_id"),
                            "backend": config.trainer.backend,
                            "tensor_limit": 1,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "inference_log_paths": inference_log_paths,
                            "inference_log_tails": inference_log_tails,
                        },
                    )
                    raise
                logger.info(
                    "training_preflight_weight_sync_witness_ok",
                    extra={
                        "event": "training_preflight_weight_sync_witness_ok",
                        **run_context,
                        "node_id": run_context.get("node_id"),
                        "backend": config.trainer.backend,
                        "tensor_limit": 1,
                        "inference_log_paths": inference_log_paths,
                    },
                )
                if runtime_init_after_witness:
                    assert init_fn is not None
                    resource_watchdog.set_phase("weight_sync_runtime_init")
                    logger.info(
                        "training_preflight_weight_sync_runtime_init_start",
                        extra={
                            "event": "training_preflight_weight_sync_runtime_init_start",
                            **run_context,
                            "node_id": run_context.get("node_id"),
                            "backend": config.trainer.backend,
                            "inference_log_paths": inference_log_paths,
                        },
                    )
                    await init_fn(
                        inference_endpoints=[e.base_url for e in inference_engines],
                        master_port=config.checkpoint.nccl_master_port + 50,
                    )
                    logger.info(
                        "training_preflight_weight_sync_runtime_init_ok",
                        extra={
                            "event": "training_preflight_weight_sync_runtime_init_ok",
                            **run_context,
                            "node_id": run_context.get("node_id"),
                            "backend": config.trainer.backend,
                            "inference_log_paths": inference_log_paths,
                        },
                    )
        # Setup data and rollout generation
        logger.info(f"Dataset: {len(prompts)} prompts")
        runtime_run_logger.event(
            "dataset_setup_start",
            **run_context,
            prompt_count=len(prompts),
        )
        resource_watchdog.set_phase("dataset_setup", prompt_count=len(prompts))
        data_buffer = DataBuffer(prompts=prompts)
        _base_generate_fn = _create_generate_fn(
            config,
            endpoint,
            tokenizer,
            environment_cls,
            environment_factory,
            metadata_key,
            logger,
        )

        async def generate_fn(*args: Any, **kwargs: Any) -> Any:
            resource_watchdog.set_phase("rollout_generation")
            try:
                return await _base_generate_fn(*args, **kwargs)
            finally:
                resource_watchdog.set_phase("rollout_loop")

        rollout_config = RolloutConfig(
            batch_size=config.rollout.batch_size,
            n_samples_per_prompt=config.rollout.n_samples_per_prompt,
            over_sampling_factor=1.0,
        )
        rollout_runtime = RolloutRuntime(
            generate_fn=generate_fn,
            scorer=scorer,
        )
        assert rollout_runtime.scorer is not None, "scorer must be resolved"

        # Training loop (delegated to rollouts.training.train.train)
        from ..training.contracts import (
            AdmissionPolicy,
            OverloadPolicy,
            PipelineRuntimeState,
            StalenessPolicy,
            VersionedRolloutBatch,
            WeightVisibilityPolicy,
        )
        from ..training.train import train as _train_loop

        # Architectural note:
        # This `pipeline_mode` switch is still a GRPO-owned orchestration state
        # machine. The requested mode is not yet lowered into validated trainer
        # and inference runtime capabilities the way training backends are.
        # `true_pipeline` in particular should eventually be realized through an
        # explicit runtime plan for both sides, not treated as "GRPO does a
        # different branch and hopes the selected backends can keep up".
        if config.checkpoint.pipeline_mode == "sync":
            staleness_policy = StalenessPolicy.synchronous()
            weight_visibility_policy = WeightVisibilityPolicy.synchronous(
                publish_mode=config.checkpoint.weight_sync_mode
            )
            admission_policy = AdmissionPolicy.synchronous()
            overload_policy = OverloadPolicy()
        elif config.checkpoint.pipeline_mode == "async":
            staleness_policy = StalenessPolicy(
                max_version_lag=config.checkpoint.max_lag,
                drop_stale=True,
                require_exact_version=(config.checkpoint.max_lag == 0),
            )
            weight_visibility_policy = WeightVisibilityPolicy(
                publish_mode=config.checkpoint.weight_sync_mode,
                atomic_visibility=True,
                drain_before_publish=False,
            )
            admission_policy = AdmissionPolicy.stream_style_default()
            overload_policy = OverloadPolicy(
                queue_pressure_threshold=(
                    config.checkpoint.pipeline_queue_size
                    if config.checkpoint.pipeline_queue_size > 0
                    else None
                ),
                block_generation_as_last_resort=config.checkpoint.pipeline_queue_size > 0,
            )
        elif config.checkpoint.pipeline_mode == "true_pipeline":
            staleness_policy = StalenessPolicy(
                max_version_lag=config.checkpoint.max_lag,
                drop_stale=True,
                require_exact_version=False,
            )
            weight_visibility_policy = WeightVisibilityPolicy(
                publish_mode=config.checkpoint.weight_sync_mode,
                atomic_visibility=True,
                drain_before_publish=False,
            )
            admission_policy = AdmissionPolicy.stream_style_default()
            overload_policy = OverloadPolicy(
                cancel_stale_inflight=True,
                queue_pressure_threshold=(
                    config.checkpoint.pipeline_queue_size
                    if config.checkpoint.pipeline_queue_size > 0
                    else None
                ),
                block_generation_as_last_resort=config.checkpoint.pipeline_queue_size > 0,
            )
        else:
            raise ValueError(
                f"Unknown pipeline_mode: {config.checkpoint.pipeline_mode!r}. "
                "Use 'sync', 'async', or 'true_pipeline'."
            )
        pipeline_state = PipelineRuntimeState(
            current_train_version=getattr(backend, "weight_version", 0),
            current_serving_version=getattr(backend, "weight_version", 0),
            sync_in_progress=False,
            admissions_paused=False,
            inflight_batches=0,
        )
        logger.info(
            "Pipeline semantics: "
            f"mode={config.checkpoint.pipeline_mode}, "
            f"staleness_max_lag={staleness_policy.max_version_lag}, "
            f"require_exact_version={staleness_policy.require_exact_version}, "
            f"publish_mode={weight_visibility_policy.publish_mode}, "
            f"drain_before_publish={weight_visibility_policy.drain_before_publish}, "
            f"pause_on_sync={admission_policy.pause_on_sync}, "
            f"queue_pressure_threshold={overload_policy.queue_pressure_threshold}"
        )
        runtime_run_logger.event(
            "rollout_loop_started",
            **run_context,
            staleness_max_lag=staleness_policy.max_version_lag,
            require_exact_version=staleness_policy.require_exact_version,
            publish_mode=weight_visibility_policy.publish_mode,
            drain_before_publish=weight_visibility_policy.drain_before_publish,
            pause_on_sync=admission_policy.pause_on_sync,
            queue_pressure_threshold=overload_policy.queue_pressure_threshold,
        )
        resource_watchdog.set_phase("rollout_loop", pipeline_mode=config.checkpoint.pipeline_mode)

        def _add_pipeline_policy_metrics(step_metrics: dict[str, Any]) -> dict[str, Any]:
            step_metrics["staleness_max_version_lag"] = float(staleness_policy.max_version_lag)
            step_metrics["staleness_require_exact_version"] = (
                1.0 if staleness_policy.require_exact_version else 0.0
            )
            step_metrics["weight_visibility_atomic"] = (
                1.0 if weight_visibility_policy.atomic_visibility else 0.0
            )
            step_metrics["weight_visibility_drain_before_publish"] = (
                1.0 if weight_visibility_policy.drain_before_publish else 0.0
            )
            step_metrics["admission_pause_on_sync"] = 1.0 if admission_policy.pause_on_sync else 0.0
            step_metrics["overload_cancel_stale_inflight"] = (
                1.0 if overload_policy.cancel_stale_inflight else 0.0
            )
            step_metrics["overload_spill_to_disk"] = 1.0 if overload_policy.spill_to_disk else 0.0
            step_metrics["overload_block_generation"] = (
                1.0 if overload_policy.block_generation_as_last_resort else 0.0
            )
            step_metrics["overload_queue_pressure_threshold"] = float(
                overload_policy.queue_pressure_threshold or 0
            )
            return step_metrics

        def _wrap_versioned_batch(
            batch: Any, *, created_at_step: int, weight_version: int
        ) -> VersionedRolloutBatch:
            return VersionedRolloutBatch(
                batch=batch,
                weight_version=weight_version,
                created_at_step=created_at_step,
                version_lag=max(pipeline_state.current_train_version - weight_version, 0),
            )

        def _reject_stale_batch(step: int, batch: Any) -> bool:
            if not isinstance(batch, VersionedRolloutBatch):
                return False
            if staleness_policy.allows_version(
                batch_version=batch.weight_version,
                current_train_version=pipeline_state.current_train_version,
            ):
                return False
            logger.warning(
                "Dropping stale rollout batch",
                extra={
                    **run_context,
                    "event": "stale_rollout_batch_dropped",
                    "step": step + 1,
                    "batch_weight_version": batch.weight_version,
                    "current_train_version": pipeline_state.current_train_version,
                    "version_lag": batch.version_lag,
                },
            )
            return True

        def _update_pipeline_state(
            *,
            train_version: int | None = None,
            serving_version: int | None = None,
            sync_in_progress: bool | None = None,
            admissions_paused: bool | None = None,
            inflight_batches: int | None = None,
        ) -> None:
            nonlocal pipeline_state
            pipeline_state = PipelineRuntimeState(
                current_train_version=(
                    pipeline_state.current_train_version if train_version is None else train_version
                ),
                current_serving_version=(
                    pipeline_state.current_serving_version
                    if serving_version is None
                    else serving_version
                ),
                sync_in_progress=(
                    pipeline_state.sync_in_progress
                    if sync_in_progress is None
                    else sync_in_progress
                ),
                admissions_paused=(
                    pipeline_state.admissions_paused
                    if admissions_paused is None
                    else admissions_paused
                ),
                inflight_batches=(
                    pipeline_state.inflight_batches
                    if inflight_batches is None
                    else inflight_batches
                ),
            )

        def _annotate_pipeline_metrics(step_metrics: dict[str, Any]) -> dict[str, Any]:
            step_metrics["pipeline_current_train_version"] = float(
                pipeline_state.current_train_version
            )
            step_metrics["pipeline_current_serving_version"] = float(
                pipeline_state.current_serving_version
            )
            step_metrics["pipeline_sync_in_progress"] = (
                1.0 if pipeline_state.sync_in_progress else 0.0
            )
            step_metrics["pipeline_admissions_paused"] = (
                1.0 if pipeline_state.admissions_paused else 0.0
            )
            step_metrics["pipeline_inflight_batches"] = float(pipeline_state.inflight_batches)
            return _add_pipeline_policy_metrics(step_metrics)

        # Step-level weight syncer (blocking). True PipelineRL uses non-blocking NCCLWeightSyncer instead.
        from ..training.weight_sync import (
            BackendNCCLWeightSyncer,
            FilesystemWeightSyncer,
            ManagedChannelWeightSyncer,
            ManagedWeightUpdateChannel,
        )
        from ..training.weight_sync_protocol import InferenceWeightUpdate, WeightSyncPolicy

        step_weight_syncer = None
        if config.checkpoint.pipeline_mode != "true_pipeline":
            raw_step_syncer = None
            if config.checkpoint.weight_sync_mode == "nccl":
                raw_step_syncer = BackendNCCLWeightSyncer(backend=backend, log=logger)
            elif config.checkpoint.weight_sync_mode == "disk":
                raw_step_syncer = FilesystemWeightSyncer(
                    backend=backend,
                    engines=inference_engines,
                    inference_sync_realization=config.checkpoint.inference_sync_realization,
                )
            else:
                raise ValueError(
                    f"Unknown weight_sync_mode: {config.checkpoint.weight_sync_mode!r}. "
                    "Use 'disk' or 'nccl'."
                )
            sync_policy = WeightSyncPolicy(
                blocking=config.checkpoint.pipeline_mode != "true_pipeline",
                sync_every=config.checkpoint.sync_weights_every,
                realization=config.checkpoint.inference_sync_realization or "",
                max_version_lag=0
                if config.checkpoint.pipeline_mode == "sync"
                else config.checkpoint.max_lag,
            )

            async def _publish_blocking_weight_update(
                _update: InferenceWeightUpdate,
                *,
                _raw_step_syncer: Any = raw_step_syncer,
            ) -> dict[str, Any]:
                await _raw_step_syncer.sync()
                return {
                    "success": True,
                    "version": _update.version,
                    "realization": _update.realization,
                }

            def _build_blocking_weight_update() -> InferenceWeightUpdate:
                current_version = getattr(backend, "weight_version", None)
                next_version = current_version + 1 if isinstance(current_version, int) else None
                return InferenceWeightUpdate(
                    version=next_version,
                    realization=config.checkpoint.inference_sync_realization,
                    metadata={
                        "pipeline_mode": config.checkpoint.pipeline_mode,
                        "weight_sync_mode": config.checkpoint.weight_sync_mode,
                    },
                )

            step_weight_syncer = ManagedChannelWeightSyncer(
                channel=ManagedWeightUpdateChannel(
                    inference=inference_engines[0],
                    policy=sync_policy,
                    publish_impl=_publish_blocking_weight_update,
                ),
                syncer=raw_step_syncer,
                update_factory=_build_blocking_weight_update,
            )

        def _log_update_channel_state(event: str) -> None:
            if step_weight_syncer is None:
                return
            state = getattr(step_weight_syncer, "state", None)
            if state is None:
                return
            logger.info(
                event,
                extra={
                    **run_context,
                    "event": event,
                    "channel_ready": state.channel_ready,
                    "quiescing_for_update": state.quiescing_for_update,
                    "update_in_progress": state.update_in_progress,
                    "last_published_version": state.last_published_version,
                    "serving_resumed": state.serving_resumed,
                    "pipeline_mode": config.checkpoint.pipeline_mode,
                    "weight_sync_mode": config.checkpoint.weight_sync_mode,
                    "inference_sync_realization": config.checkpoint.inference_sync_realization,
                },
            )

        def _log_post_weight_sync_state() -> None:
            if step_weight_syncer is None:
                return
            state = getattr(step_weight_syncer, "state", None)
            if state is None:
                return
            event = (
                "weight_update_channel_after_sync"
                if not state.update_in_progress and state.serving_resumed
                else "weight_update_channel_after_sync_failed"
            )
            _log_update_channel_state(event)

        if step_weight_syncer is not None:
            _log_update_channel_state("weight_update_channel_created")

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
            save_result = save_fn(step, numeric_metrics)
            if hasattr(save_result, "result") and not hasattr(save_result, "__await__"):
                ckpt_result = await save_result.result()
            else:
                ckpt_result = await save_result
            ckpt_dir = ckpt_result
            logger.info(f"Saved checkpoint: {ckpt_dir}")
            return ckpt_dir

        async def _sync_batches() -> AsyncIterator[Any]:
            # Synchronous training (default): generate batch, train, sync weights, repeat.
            async with AsyncRolloutManager(
                data_buffer,
                rollout_config,
                runtime=rollout_runtime,
            ) as rollout_manager:
                for _step in range(config.checkpoint.num_steps):
                    batch = await rollout_manager.generate_batch(scorer=rollout_runtime.scorer)
                    _attach_runtime_observability(
                        batch,
                        rollout_manager,
                        rollout_runtime.scorer,
                        environment_factory,
                    )

                    # TODO(async-design-decisions.md): Teacher / judge scoring
                    # should be a general scoring stage, not a sync-path special
                    # case. This inline OPD branch bakes in "scoring is cheap and
                    # adjacent to rollout batching", which does not match the
                    # target architecture.
                    # See rollouts/training/async-design-decisions.md decisions 15 and 16.
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
                    yield _wrap_versioned_batch(
                        batch,
                        created_at_step=_step + 1,
                        weight_version=pipeline_state.current_serving_version,
                    )

        # TODO: split this pipeline_mode branch into smaller helpers. It is
        # currently carrying too much nested orchestration, policy wiring, and
        # transport-specific control flow in one place.
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
                runtime=rollout_runtime,
                max_lag=config.checkpoint.max_lag,
                queue_size=config.checkpoint.pipeline_queue_size,
                # Current inference backend still blocks while NCCL receives weights.
                # Keep this explicit: no new prompt admissions while sync is active.
                sync_in_progress_fn=lambda: weight_sync_manager.sync_in_progress,
            )

            # TODO(async-design-decisions.md): This mode is still modeled as a
            # bounded in-memory queue with pause-on-sync behavior. Our agreed
            # target is stream-style production semantics with explicit safety
            # valves, plus drain-then-sync as the clean default sync semantic.
            # See rollouts/training/async-design-decisions.md decisions 2, 9, 12, 14.
            logger.info(
                f"Using experimental in-flight sync pipeline mode (max_lag={config.checkpoint.max_lag}, "
                f"queue_size={config.checkpoint.pipeline_queue_size}, "
                f"engines={num_engines})"
            )
            logger.info("  - Background sampling: ON")
            logger.info("  - Weight sync transport: async NCCL")
            logger.info("  - New admissions pause while inference is blocked by sync")
            if num_engines > 1:
                logger.info(f"  - Multi-engine: {num_engines} inference servers")

            async def _true_pipeline_batches() -> AsyncIterator[Any]:
                try:
                    # Initialize NCCL for experimental trainer-side overlap.
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
                                    scorer=rollout_runtime.scorer,
                                )
                                _attach_runtime_observability(
                                    batch,
                                    pipelined_manager,
                                    rollout_runtime.scorer,
                                    environment_factory,
                                )
                                _update_pipeline_state(
                                    serving_version=weight_sync_manager.current_version
                                )
                                yield _wrap_versioned_batch(
                                    batch,
                                    created_at_step=step + 1,
                                    weight_version=weight_sync_manager.current_version,
                                )

                                # Trainer-side non-blocking publication. Current
                                # direct receive/load realizations may still pause
                                # new admissions while sync is in progress.
                                #
                                # Architectural note:
                                # This reaches through the backend abstraction to
                                # `backend.model`, which means `true_pipeline`
                                # currently depends on a PyTorch-shaped training
                                # backend capability that is not represented in
                                # `TrainingBackend`. A real lowering should ask
                                # the training backend for an explicit async
                                # publication capability instead of assuming the
                                # concrete model object is available here.
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
                if _reject_stale_batch(step, batch):
                    return None
                resource_watchdog.set_phase("train_step", step=step + 1)
                _update_pipeline_state(
                    train_version=getattr(
                        _backend, "weight_version", pipeline_state.current_train_version
                    ),
                    inflight_batches=1,
                )
                step_metrics = await _process_training_step(
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
                _update_pipeline_state(
                    train_version=getattr(
                        _backend, "weight_version", pipeline_state.current_train_version
                    ),
                    inflight_batches=0,
                )
                if step_metrics is None:
                    return None
                resource_watchdog.set_phase("rollout_loop", step=step + 1)
                return _annotate_pipeline_metrics(step_metrics)

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
                runtime=rollout_runtime,
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
                                scorer=rollout_runtime.scorer,
                            )
                            _attach_runtime_observability(
                                batch,
                                pipelined_manager,
                                rollout_runtime.scorer,
                                environment_factory,
                            )
                            _update_pipeline_state(
                                serving_version=getattr(backend, "weight_version", 0)
                            )
                            yield _wrap_versioned_batch(
                                batch,
                                created_at_step=_step + 1,
                                weight_version=getattr(backend, "weight_version", 0),
                            )
                            pipelined_manager.update_weight_version(backend.weight_version)

                        # Log pipeline stats
                        stats = pipelined_manager.stats()
                        logger.info(
                            f"Pipeline stats: generated={stats['samples_generated']}, "
                            f"discarded_stale={stats['samples_discarded_stale']} "
                            f"({stats['discard_rate']:.1f}%)"
                        )

            async def _process_batch(step: int, batch: Any, _backend: Any) -> dict[str, Any] | None:
                if _reject_stale_batch(step, batch):
                    return None
                resource_watchdog.set_phase("train_step", step=step + 1)
                _update_pipeline_state(
                    train_version=getattr(
                        _backend, "weight_version", pipeline_state.current_train_version
                    ),
                    inflight_batches=1,
                )
                step_metrics = await _process_training_step(
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
                _update_pipeline_state(
                    train_version=getattr(
                        _backend, "weight_version", pipeline_state.current_train_version
                    ),
                    inflight_batches=0,
                )
                if step_metrics is None:
                    return None
                return _annotate_pipeline_metrics(step_metrics)

            async def _before_async_weight_sync() -> None:
                _log_update_channel_state("weight_update_channel_before_sync")
                _update_pipeline_state(
                    train_version=getattr(
                        backend, "weight_version", pipeline_state.current_train_version
                    ),
                    sync_in_progress=True,
                    admissions_paused=admission_policy.pause_on_sync,
                )
                if admission_policy.pause_on_sync:
                    await _pause_pipeline_admissions(pipelined_manager, logger)

            async def _after_async_weight_sync() -> None:
                current_version = getattr(
                    backend, "weight_version", pipeline_state.current_train_version
                )
                _log_post_weight_sync_state()
                _update_pipeline_state(
                    train_version=current_version,
                    serving_version=current_version,
                    sync_in_progress=False,
                    admissions_paused=False,
                )
                if admission_policy.pause_on_sync:
                    await _resume_pipeline_admissions(pipelined_manager, logger)

            train_result = await _train_loop(
                config=config.checkpoint,
                backend=backend,
                batch_iterator=_async_pipeline_batches(),
                process_batch=_process_batch,
                weight_syncer=step_weight_syncer,
                save_checkpoint=_save_checkpoint,
                metrics_logger=metrics_logger,
                logger=logger,
                before_weight_sync=_before_async_weight_sync,
                after_weight_sync=_after_async_weight_sync,
            )

        else:

            async def _process_batch(step: int, batch: Any, _backend: Any) -> dict[str, Any] | None:
                if _reject_stale_batch(step, batch):
                    return None
                _update_pipeline_state(
                    train_version=getattr(
                        _backend, "weight_version", pipeline_state.current_train_version
                    ),
                    inflight_batches=1,
                )
                step_metrics = await _process_training_step(
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
                if step_metrics is None:
                    _update_pipeline_state(
                        train_version=getattr(
                            _backend, "weight_version", pipeline_state.current_train_version
                        ),
                        inflight_batches=0,
                    )
                    resource_watchdog.set_phase("rollout_loop", step=step + 1)
                    return None
                _update_pipeline_state(
                    train_version=getattr(
                        _backend, "weight_version", pipeline_state.current_train_version
                    ),
                    inflight_batches=0,
                )
                resource_watchdog.set_phase("rollout_loop", step=step + 1)
                return _annotate_pipeline_metrics(step_metrics)

            async def _before_weight_sync() -> None:
                _log_update_channel_state("weight_update_channel_before_sync")
                _update_pipeline_state(
                    train_version=getattr(
                        backend, "weight_version", pipeline_state.current_train_version
                    ),
                    sync_in_progress=True,
                    admissions_paused=admission_policy.pause_on_sync,
                )

            async def _after_weight_sync() -> None:
                current_version = getattr(
                    backend, "weight_version", pipeline_state.current_train_version
                )
                _log_post_weight_sync_state()
                _update_pipeline_state(
                    train_version=current_version,
                    serving_version=current_version,
                    sync_in_progress=False,
                    admissions_paused=False,
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
                before_weight_sync=_before_weight_sync,
                after_weight_sync=_after_weight_sync,
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

    except Exception as e:
        logger.exception(
            "inference startup or training failed",
            extra={
                **run_context,
                "event": "inference_or_training_failed",
                "error_type": type(e).__name__,
                "error": str(e),
                "num_engines": num_engines,
                "teacher_engine": teacher_engine is not None,
            },
        )
        raise
    finally:
        resource_watchdog.stop()
        try:
            await _maybe_stop_environment_factory(environment_factory, logger)
        except Exception as e:
            logger.warning(f"Environment resource cleanup failed: {e}")

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
