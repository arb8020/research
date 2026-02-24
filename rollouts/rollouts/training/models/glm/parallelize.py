"""Parallelization for GLM models.

Applies TP, FSDP, activation checkpointing, and torch.compile.

Note: This is a simplified implementation. For full torchtitan features
(expert parallelism, pipeline parallelism, etc.), use the torchtitan
training loop directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from torchtitan.config import JobConfig
    from torchtitan.distributed import ParallelDims


def parallelize_glm(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:
    """Apply parallelization to GLM model.

    This is a simplified version that applies:
    1. FSDP (if enabled)
    2. Activation checkpointing (if enabled)
    3. torch.compile (if enabled)

    For full 4D parallelism (TP + CP + PP + FSDP), use torchtitan's
    training loop with the appropriate job config.
    """
    import torch
    from torch.distributed._composable.fsdp import fully_shard
    from torchtitan.distributed.activation_checkpoint import apply_ac
    from torchtitan.tools.logging import logger

    model_compile_enabled = job_config.compile.enable and "model" in job_config.compile.components

    # Activation checkpointing
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            base_folder=job_config.job.dump_folder,
        )
        logger.info("Applied activation checkpointing to GLM model")

    # torch.compile per-block
    if model_compile_enabled:
        layers = model.layers
        for layer_id, layer in layers.items():
            layers[layer_id] = torch.compile(layer, fullgraph=True)
        logger.info("Applied torch.compile to GLM model")

    # FSDP
    if parallel_dims.fsdp_enabled:
        dp_mesh = parallel_dims.get_mesh("fsdp")

        # Shard embedding
        fully_shard(model.tok_embeddings, mesh=dp_mesh)

        # Shard each transformer block
        for layer in model.layers.values():
            fully_shard(layer, mesh=dp_mesh)

        # Shard output
        fully_shard(model.output, mesh=dp_mesh)

        # Shard the whole model
        fully_shard(model, mesh=dp_mesh)

        logger.info("Applied FSDP to GLM model")

    return model
