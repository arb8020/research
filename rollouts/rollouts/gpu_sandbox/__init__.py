"""GPU Sandbox Pool for distributed kernel scoring.

This module provides a SandboxPool abstraction for running kernel evaluation
on remote GPU instances. Designed to:

1. Work with multiple providers (Modal, RunPod, SSH targets)
2. Scale from 0 sandboxes (local subprocess scoring) to N remote sandboxes
3. Be provider-agnostic: all sandboxes run the same scoring worker
4. Support future multi-node training/inference separation

Architecture (single node, external scoring):

    Training Node                      Sandbox Pool (remote GPUs)
    ├── SGLang (inference)             ├── Sandbox 0 (Modal A100)
    ├── Trainer (GRPO)                 ├── Sandbox 1 (Modal A100)
    └── SandboxPool.score_batch() ────▶├── Sandbox 2 (RunPod H100)
                                       └── ...

Architecture (multi-node, future):

    Training Nodes                     Inference Nodes              Scoring Sandboxes
    ├── Node 0 (FSDP rank 0)          ├── Node 0 (SGLang)          ├── Sandbox 0
    ├── Node 1 (FSDP rank 1)          └── Node 1 (SGLang)          └── Sandbox 1
    └── ...                                    │
              ▲                                │
              └── weight sync (miniray) ───────┘

The key abstraction is SandboxPool, which:
- Manages a set of SandboxWorker connections (via miniray RemoteWorker)
- Provisions sandboxes on demand from configured providers
- Distributes scoring work across available workers
- Handles health checks and worker recovery

Usage:
    from rollouts.gpu_sandbox import SandboxPool, ModalSandboxConfig

    # Create pool (sandboxes provisioned lazily on start())
    pool = SandboxPool([
        ModalSandboxConfig(gpu="A100", count=2),
    ])

    # Start sandboxes
    await pool.start()

    # Score a batch of samples (distributes across workers)
    scores = await pool.score_batch(samples, score_fn)

    # Cleanup
    await pool.stop()

For local scoring (no remote sandboxes):
    pool = SandboxPool([])  # empty config = local subprocess
"""

from rollouts.gpu_sandbox.config import (
    BrokerSandboxConfig,
    ExistingInstanceConfig,
    LocalSandboxConfig,
    # Legacy aliases
    ModalSandboxConfig,
    RunPodSandboxConfig,
    SandboxConfig,
    SSHSandboxConfig,
)
from rollouts.gpu_sandbox.pool import SandboxPool
from rollouts.gpu_sandbox.worker import (
    BrokerSandboxWorker,
    LocalSandboxWorker,
    SandboxWorker,
)

__all__ = [
    "SandboxPool",
    "SandboxConfig",
    "BrokerSandboxConfig",
    "ExistingInstanceConfig",
    "LocalSandboxConfig",
    # Legacy aliases
    "ModalSandboxConfig",
    "RunPodSandboxConfig",
    "SSHSandboxConfig",
    # Workers
    "SandboxWorker",
    "LocalSandboxWorker",
    "BrokerSandboxWorker",
]
