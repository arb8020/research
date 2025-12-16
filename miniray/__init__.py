"""MiniRay: Simple distributed system built on TCP + Heinrich's pattern.

MiniRay is a minimal distributed computing system that provides:
- Multi-node worker spawning (via TCP + SSH)
- Same API as local Worker pattern
- NCCL setup for distributed training
- Simple semantics (no Ray complexity)

Example:
    >>> # Define cluster
    >>> cluster = Cluster(nodes=[
    ...     NodeConfig("node1.local", num_workers=4),
    ...     NodeConfig("node2.local", num_workers=4),
    ... ])
    >>>
    >>> # Launch workers
    >>> workers = cluster.start(work_fn="module.train_fn")
    >>>
    >>> # Use them (same API as local Worker!)
    >>> for worker in workers:
    ...     worker.send({"cmd": "train", "batch": batch})
    >>> results = [w.recv() for w in workers]

Tiger Style: Explicit control, simple primitives.
Heinrich Kuttler: TCP + fork > Ray's complexity.
"""

from miniray.cluster import Cluster, NodeConfig
from miniray.gpu_affinity import get_gpu_numa_node, set_gpu_affinity
from miniray.nccl import NCCLConfig, create_nccl_configs, restore_nccl_env, setup_nccl_env
from miniray.remote_worker import RemoteWorker
from miniray.worker import Worker, wait_any

__all__ = [
    "Worker",
    "Cluster",
    "NodeConfig",
    "NCCLConfig",
    "RemoteWorker",
    "wait_any",
    "setup_nccl_env",
    "restore_nccl_env",
    "create_nccl_configs",
    "set_gpu_affinity",
    "get_gpu_numa_node",
]
