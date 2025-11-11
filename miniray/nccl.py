"""NCCL configuration for multi-node distributed training.

Provides utilities for setting up torch.distributed with NCCL backend
across multiple nodes. Handles environment variable configuration needed
for NCCL initialization.

Tiger Style: Explicit NCCL setup, no hidden magic.
PyTorch: NCCL requires specific env vars for multi-node.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class NCCLConfig:
    """Configuration for NCCL multi-node training.

    torch.distributed needs these environment variables for multi-node:
    - MASTER_ADDR: IP of rank 0 node (coordination)
    - MASTER_PORT: Port for coordination
    - WORLD_SIZE: Total number of processes
    - RANK: Global rank of this process (0 to world_size-1)
    - LOCAL_RANK: Rank within this node (0 to num_gpus_per_node-1)

    Example:
        >>> # For 2 nodes, 4 GPUs each (8 total processes)
        >>> # Node 1, GPU 0:
        >>> config = NCCLConfig(
        ...     master_addr="192.168.1.10",
        ...     master_port=29500,
        ...     world_size=8,
        ...     rank=0,
        ...     local_rank=0,
        ... )
        >>>
        >>> # Node 2, GPU 3:
        >>> config = NCCLConfig(
        ...     master_addr="192.168.1.10",  # Same master
        ...     master_port=29500,
        ...     world_size=8,
        ...     rank=7,  # 4 workers on node 1, this is 4th on node 2
        ...     local_rank=3,  # 4th GPU on this node
        ... )
    """

    master_addr: str
    """IP or hostname of rank 0 node (master for coordination)"""

    master_port: int = 29500
    """Port for NCCL coordination (default: 29500)"""

    world_size: int = 1
    """Total number of processes across all nodes"""

    rank: int = 0
    """Global rank of this process (0 to world_size-1)"""

    local_rank: int = 0
    """Local rank within this node (0 to num_gpus_per_node-1)"""

    def __post_init__(self):
        """Validate configuration."""
        # Tiger Style: Assert preconditions
        assert self.master_addr, "master_addr cannot be empty"
        assert 1024 <= self.master_port <= 65535, (
            f"master_port must be in range [1024, 65535], got {self.master_port}"
        )
        assert self.world_size > 0, f"world_size must be > 0, got {self.world_size}"
        assert 0 <= self.rank < self.world_size, (
            f"rank must be in [0, {self.world_size}), got {self.rank}"
        )
        assert self.local_rank >= 0, (
            f"local_rank must be >= 0, got {self.local_rank}"
        )


def setup_nccl_env(config: NCCLConfig) -> dict[str, Optional[str]]:
    """Set up environment variables for NCCL.

    Call this BEFORE torch.distributed.init_process_group().

    Args:
        config: NCCL configuration

    Returns:
        Dictionary of old environment variable values (for restoration)

    Side effects:
        - Sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK
        - Sets CUDA_VISIBLE_DEVICES to local_rank

    Example:
        >>> def training_work_fn(handle):
        ...     # Receive config from coordinator
        ...     config = handle.recv(max_size=1024)
        ...     config = NCCLConfig(**config)
        ...
        ...     # Set up NCCL environment (save old values)
        ...     old_env = setup_nccl_env(config)
        ...
        ...     # Initialize torch.distributed
        ...     import torch.distributed as dist
        ...     dist.init_process_group(backend="nccl", init_method="env://")
        ...
        ...     # ... training ...
        ...
        ...     # Restore old environment (optional)
        ...     restore_nccl_env(old_env)
    """
    # Tiger Style: Assert preconditions
    assert config is not None, "config cannot be None"
    assert isinstance(config, NCCLConfig), f"config must be NCCLConfig, got {type(config)}"

    # Save old values for restoration
    env_vars = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"]
    old_env = {}
    for var in env_vars:
        old_env[var] = os.environ.get(var)

    # Set new values
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    os.environ["WORLD_SIZE"] = str(config.world_size)
    os.environ["RANK"] = str(config.rank)
    os.environ["LOCAL_RANK"] = str(config.local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.local_rank)

    # Tiger Style: Assert postconditions
    assert os.environ["MASTER_ADDR"] == config.master_addr, \
        f"MASTER_ADDR mismatch: {os.environ['MASTER_ADDR']} != {config.master_addr}"
    assert os.environ["RANK"] == str(config.rank), \
        f"RANK mismatch: {os.environ['RANK']} != {config.rank}"

    return old_env


def restore_nccl_env(old_env: dict[str, Optional[str]]) -> None:
    """Restore environment variables to previous state.

    Args:
        old_env: Dictionary returned from setup_nccl_env()

    Example:
        >>> old_env = setup_nccl_env(config)
        >>> # ... do training ...
        >>> restore_nccl_env(old_env)
    """
    # Tiger Style: Assert preconditions
    assert old_env is not None, "old_env cannot be None"
    assert isinstance(old_env, dict), f"old_env must be dict, got {type(old_env)}"

    for var, value in old_env.items():
        if value is None:
            # Variable didn't exist before, remove it
            if var in os.environ:
                del os.environ[var]
        else:
            # Restore old value
            os.environ[var] = value


def create_nccl_configs(
    master_addr: str,
    nodes: list[tuple[str, int]],  # [(hostname, num_gpus), ...]
    master_port: int = 29500,
) -> list[NCCLConfig]:
    """Create NCCL configs for all workers in a cluster.

    Args:
        master_addr: IP/hostname of rank 0 node
        nodes: List of (hostname, num_gpus) for each node
        master_port: Port for coordination

    Returns:
        List of NCCLConfig, one per worker (GPU)

    Example:
        >>> # 2 nodes: node1 has 4 GPUs, node2 has 4 GPUs
        >>> configs = create_nccl_configs(
        ...     master_addr="192.168.1.10",
        ...     nodes=[
        ...         ("node1", 4),
        ...         ("node2", 4),
        ...     ],
        ... )
        >>> # configs[0] = rank 0, local_rank 0 (node1, GPU 0)
        >>> # configs[3] = rank 3, local_rank 3 (node1, GPU 3)
        >>> # configs[4] = rank 4, local_rank 0 (node2, GPU 0)
        >>> # configs[7] = rank 7, local_rank 3 (node2, GPU 3)
    """
    # Tiger Style: Assert preconditions
    assert master_addr, "master_addr cannot be empty"
    assert len(master_addr) < 256, f"master_addr too long: {len(master_addr)}"
    assert nodes, "nodes cannot be empty"
    assert len(nodes) > 0, "Must have at least one node"
    assert len(nodes) < 1000, f"Too many nodes: {len(nodes)}"
    assert 1024 <= master_port <= 65535, \
        f"master_port must be in range [1024, 65535], got {master_port}"

    # Validate each node
    for i, node in enumerate(nodes):
        assert node is not None, f"nodes[{i}] is None"
        assert isinstance(node, tuple), f"nodes[{i}] must be tuple, got {type(node)}"
        assert len(node) == 2, f"nodes[{i}] must be (name, num_gpus), got {len(node)} elements"

        node_name, num_gpus = node
        assert isinstance(node_name, str), f"nodes[{i}] name must be str, got {type(node_name)}"
        assert isinstance(num_gpus, int), f"nodes[{i}] num_gpus must be int, got {type(num_gpus)}"
        assert num_gpus > 0, f"nodes[{i}] num_gpus must be > 0, got {num_gpus}"
        assert num_gpus <= 16, f"nodes[{i}] num_gpus too large: {num_gpus} (max 16)"

    configs = []
    global_rank = 0
    world_size = sum(num_gpus for _, num_gpus in nodes)

    # Tiger Style: Assert world_size is valid
    assert world_size > 0, "world_size must be > 0"
    assert world_size < 10000, f"world_size too large: {world_size}"

    for node_name, num_gpus in nodes:
        for local_rank in range(num_gpus):
            config = NCCLConfig(
                master_addr=master_addr,
                master_port=master_port,
                world_size=world_size,
                rank=global_rank,
                local_rank=local_rank,
            )
            configs.append(config)
            global_rank += 1

    return configs


def print_nccl_config(config: NCCLConfig):
    """Pretty-print NCCL configuration.

    Args:
        config: NCCL configuration

    Example:
        >>> print_nccl_config(config)
        NCCL Configuration:
          Master:     192.168.1.10:29500
          World Size: 8
          Rank:       3/8
          Local Rank: 3
          GPU:        cuda:3
    """
    print("NCCL Configuration:")
    print(f"  Master:     {config.master_addr}:{config.master_port}")
    print(f"  World Size: {config.world_size}")
    print(f"  Rank:       {config.rank}/{config.world_size}")
    print(f"  Local Rank: {config.local_rank}")
    print(f"  GPU:        cuda:{config.local_rank}")


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    """Example: Generate NCCL configs for a cluster"""

    print("Example: 2-node cluster, 4 GPUs per node\n")

    # Create configs for all workers
    configs = create_nccl_configs(
        master_addr="192.168.1.10",
        nodes=[
            ("node1", 4),  # 4 GPUs on node1
            ("node2", 4),  # 4 GPUs on node2
        ],
    )

    # Print configs for each worker
    for i, config in enumerate(configs):
        print(f"\n=== Worker {i} ===")
        print_nccl_config(config)

    print("\n\nUsage in training script:")
    print("""
    def training_work_fn(handle):
        # Receive config from coordinator
        config_dict = handle.recv()
        config = NCCLConfig(**config_dict)

        # Set up NCCL environment
        setup_nccl_env(config)

        # Initialize torch.distributed
        dist.init_process_group(backend="nccl", init_method="env://")

        # Now FSDP works across nodes!
        model = FSDP(model, ...)
    """)
