"""Multi-node FSDP training launcher.

Orchestrates distributed training across multiple GPU nodes using:
- bifrost: Provisions GPU instances (RunPod, etc.)
- miniray: Coordinates workers via SSH
- FSDP: Shards model across GPUs within and across nodes

Architecture (2 nodes × 8 GPUs example):
    Node 0 (master):
        GPU 0-1: Inference engines (2 servers, ports 30000-30001)
        GPU 2-7: FSDP trainer (ranks 0-5)
    Node 1:
        GPU 0-1: Inference engines (2 servers, ports 30000-30001)
        GPU 2-7: FSDP trainer (ranks 6-11)

NCCL groups:
    1. FSDP training group: All trainer GPUs (12 total)
    2. Weight sync group: Trainer rank 0 + all inference GPUs (5 total)

Tiger Style: Explicit orchestration, clear state transitions.
PipelineRL: In-flight weight updates, inference never stops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiNodeConfig:
    """Configuration for multi-node distributed training.

    Specifies how to provision and use multiple GPU nodes for training.
    Combines hardware provisioning (bifrost) with distributed strategy.

    Example:
        # 2 nodes × 8 H100s: 4 inference GPUs + 12 trainer GPUs
        config = MultiNodeConfig(
            num_nodes=2,
            gpus_per_node=8,
            inference_gpus_per_node=2,
            gpu_type="H100",
        )

        # Resulting allocation:
        # - 4 inference engines total (2 per node)
        # - 12 FSDP ranks total (6 per node)
        # - NCCL weight sync: trainer rank 0 → all 4 inference engines
    """

    # Node provisioning
    num_nodes: int = 2
    gpus_per_node: int = 8
    gpu_type: str = "H100"
    provider: Literal["runpod", "lambdalabs", "vast", "local"] = "runpod"
    container_disk_gb: int = 100
    persistent_volume_id: str | None = None
    persistent_volume_mount_path: str = "/workspace"
    persistent_volume_location: str | None = None

    # GPU allocation per node
    inference_gpus_per_node: int = 2  # First N GPUs for inference
    # Remaining GPUs (gpus_per_node - inference_gpus_per_node) are for training

    # Inference settings
    inference_tp: int = 1  # Tensor parallel per inference engine
    inference_base_port: int = 30000

    # NCCL settings
    nccl_master_port: int = 29500
    nccl_weight_sync_port: int = 29501

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.num_nodes >= 1, f"num_nodes must be >= 1, got {self.num_nodes}"
        assert self.gpus_per_node >= 2, f"gpus_per_node must be >= 2, got {self.gpus_per_node}"
        assert self.inference_gpus_per_node >= 1, (
            f"inference_gpus_per_node must be >= 1, got {self.inference_gpus_per_node}"
        )
        assert self.inference_gpus_per_node < self.gpus_per_node, (
            f"inference_gpus_per_node ({self.inference_gpus_per_node}) must be < "
            f"gpus_per_node ({self.gpus_per_node})"
        )
        assert self.inference_gpus_per_node % self.inference_tp == 0, (
            f"inference_gpus_per_node ({self.inference_gpus_per_node}) must be "
            f"divisible by inference_tp ({self.inference_tp})"
        )

    @property
    def trainer_gpus_per_node(self) -> int:
        """Number of trainer GPUs per node."""
        return self.gpus_per_node - self.inference_gpus_per_node

    @property
    def total_inference_gpus(self) -> int:
        """Total inference GPUs across all nodes."""
        return self.inference_gpus_per_node * self.num_nodes

    @property
    def total_trainer_gpus(self) -> int:
        """Total trainer GPUs across all nodes (FSDP world size)."""
        return self.trainer_gpus_per_node * self.num_nodes

    @property
    def inference_engines_per_node(self) -> int:
        """Number of inference engines per node."""
        return self.inference_gpus_per_node // self.inference_tp

    @property
    def total_inference_engines(self) -> int:
        """Total inference engines across all nodes."""
        return self.inference_engines_per_node * self.num_nodes


@dataclass
class NodeAllocation:
    """GPU allocation for a single node.

    Computed from MultiNodeConfig. Describes which GPUs run inference
    vs training on a specific node.
    """

    node_rank: int
    hostname: str
    public_ip: str

    # Inference allocation
    inference_gpus: tuple[int, ...]  # e.g., (0, 1)
    inference_ports: tuple[int, ...]  # e.g., (30000, 30001)

    # Trainer allocation
    trainer_gpus: tuple[int, ...]  # e.g., (2, 3, 4, 5, 6, 7)
    trainer_fsdp_ranks: tuple[int, ...]  # Global FSDP ranks, e.g., (0, 1, 2, 3, 4, 5)


@dataclass
class ClusterAllocation:
    """Complete GPU allocation for all nodes.

    Computed from MultiNodeConfig. Contains all information needed
    to launch inference engines and FSDP trainers.
    """

    config: MultiNodeConfig
    nodes: list[NodeAllocation] = field(default_factory=list)

    # Master node info (node 0)
    master_addr: str = ""
    master_port: int = 29500

    @property
    def all_inference_endpoints(self) -> list[str]:
        """All inference engine endpoints (for weight sync)."""
        endpoints = []
        for node in self.nodes:
            for port in node.inference_ports:
                endpoints.append(f"http://{node.public_ip}:{port}")
        return endpoints

    @property
    def fsdp_world_size(self) -> int:
        """Total FSDP ranks (trainer GPUs)."""
        return sum(len(node.trainer_fsdp_ranks) for node in self.nodes)


def compute_cluster_allocation(
    config: MultiNodeConfig,
    node_ips: list[str],
) -> ClusterAllocation:
    """Compute GPU allocation for all nodes.

    Args:
        config: Multi-node configuration
        node_ips: Public IPs of provisioned nodes (in order)

    Returns:
        Complete cluster allocation with per-node GPU assignments

    Example:
        >>> config = MultiNodeConfig(num_nodes=2, gpus_per_node=8, inference_gpus_per_node=2)
        >>> allocation = compute_cluster_allocation(config, ["1.2.3.4", "5.6.7.8"])
        >>> allocation.nodes[0].inference_gpus
        (0, 1)
        >>> allocation.nodes[0].trainer_gpus
        (2, 3, 4, 5, 6, 7)
        >>> allocation.nodes[0].trainer_fsdp_ranks
        (0, 1, 2, 3, 4, 5)
    """
    assert len(node_ips) == config.num_nodes, (
        f"Expected {config.num_nodes} node IPs, got {len(node_ips)}"
    )

    nodes = []
    fsdp_rank_offset = 0

    for node_rank, public_ip in enumerate(node_ips):
        # Inference GPUs: first N GPUs
        inference_gpus = tuple(range(config.inference_gpus_per_node))

        # Inference ports: one per engine (accounting for TP)
        num_engines = config.inference_engines_per_node
        inference_ports = tuple(config.inference_base_port + i for i in range(num_engines))

        # Trainer GPUs: remaining GPUs
        trainer_gpus = tuple(range(config.inference_gpus_per_node, config.gpus_per_node))

        # FSDP ranks: global ranks for this node's trainer GPUs
        num_trainer_gpus = len(trainer_gpus)
        trainer_fsdp_ranks = tuple(range(fsdp_rank_offset, fsdp_rank_offset + num_trainer_gpus))
        fsdp_rank_offset += num_trainer_gpus

        node = NodeAllocation(
            node_rank=node_rank,
            hostname=f"node-{node_rank}",
            public_ip=public_ip,
            inference_gpus=inference_gpus,
            inference_ports=inference_ports,
            trainer_gpus=trainer_gpus,
            trainer_fsdp_ranks=trainer_fsdp_ranks,
        )
        nodes.append(node)

    return ClusterAllocation(
        config=config,
        nodes=nodes,
        master_addr=node_ips[0],
        master_port=config.nccl_master_port,
    )


async def provision_nodes(
    config: MultiNodeConfig,
) -> list[tuple[Any, Any]]:  # list[(BifrostClient, GPUInstance)]
    """Provision GPU nodes via bifrost.

    Args:
        config: Multi-node configuration

    Returns:
        List of (client, instance) tuples for each node

    Side effects:
        - Provisions GPU instances on cloud provider
        - SSH keys are configured for access
    """
    from bifrost import GPUQuery, acquire_node

    nodes = []
    for i in range(config.num_nodes):
        logger.info(f"Provisioning node {i + 1}/{config.num_nodes}...")

        client, instance = await acquire_node(
            provision=GPUQuery(
                type=config.gpu_type,
                count=config.gpus_per_node,
                provider=config.provider,
                container_disk_gb=config.container_disk_gb,
                persistent_volume_id=config.persistent_volume_id,
                persistent_volume_mount_path=config.persistent_volume_mount_path,
                persistent_volume_location=config.persistent_volume_location,
            )
        )

        logger.info(
            f"  Node {i}: {instance.public_ip} ({config.gpu_type} × {config.gpus_per_node})"
        )
        nodes.append((client, instance))

    return nodes


async def deploy_to_nodes(
    nodes: list[tuple[Any, Any]],
    allow_dirty: bool = False,
) -> list[str]:
    """Deploy code to all nodes.

    Uses the same bifrost API as run.py for consistency.

    Args:
        nodes: List of (bifrost_client, instance) from provision_nodes
        allow_dirty: Allow deploying uncommitted changes

    Returns:
        List of workspace paths (one per node)

    Side effects:
        - Git syncs code to each node
        - Installs uv and dependencies
        - Installs sglang
    """
    workspaces: list[str] = []

    # Bootstrap steps (same as run.py)
    bootstrap_steps = [
        ("Installing system deps", "apt-get update && apt-get install -y tmux libnuma1 || true"),
        (
            "Installing uv",
            "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env",
        ),
        (
            "Syncing Python deps",
            "~/.local/bin/uv python install 3.12 && ~/.local/bin/uv sync --python 3.12 --package rollouts",
        ),
        (
            "Installing ML packages",
            "~/.local/bin/uv pip install --upgrade torch datasets accelerate curl_cffi peft"
            " 'sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python'"
            " && ~/.local/bin/uv pip install --upgrade 'transformers>=5.0.0' 'huggingface_hub>=1.4.0'",
        ),
    ]

    def deploy_one(client: Any, instance: Any, idx: int) -> str:
        logger.info(f"Deploying to node {idx}: {instance.public_ip}...")

        # Push code via git sync
        workspace = client.push("~/.bifrost/workspaces/rollouts-rl", allow_dirty=allow_dirty)
        logger.info(f"  Node {idx}: Code synced to {workspace}")

        # Run bootstrap steps
        for label, cmd in bootstrap_steps:
            logger.info(f"  Node {idx}: {label}...")
            client.exec(cmd, working_dir=workspace)

        logger.info(f"  Node {idx}: Deployed")
        return workspace

    # Deploy sequentially (bifrost client is sync)
    for i, (client, instance) in enumerate(nodes):
        ws = deploy_one(client, instance, i)
        workspaces.append(ws)

    return workspaces


def generate_worker_commands(
    allocation: ClusterAllocation,
    config_path: str,
    work_fn: str = "rollouts.training.fsdp_worker.fsdp_train_worker",
) -> dict[str, list[str]]:
    """Generate launch commands for each node.

    Returns dict mapping node IP to list of commands to run.
    Each node runs:
    1. N inference engine processes
    2. M trainer processes (one per GPU)

    Args:
        allocation: Cluster allocation from compute_cluster_allocation
        config_path: Path to training config file
        work_fn: Python path to worker function

    Returns:
        Dict of {node_ip: [cmd1, cmd2, ...]}
    """
    commands: dict[str, list[str]] = {}

    for node in allocation.nodes:
        node_commands = []

        # Inference engine commands
        for i, (gpus_start, port) in enumerate(
            zip(
                range(0, len(node.inference_gpus), allocation.config.inference_tp),
                node.inference_ports,
                strict=False,
            )
        ):
            gpu_ids = ",".join(
                str(node.inference_gpus[gpus_start + j])
                for j in range(allocation.config.inference_tp)
            )
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu_ids} "
                f"python -m sglang.launch_server "
                f"--port {port} "
                f"--model-path $MODEL_PATH "
                f"--tp {allocation.config.inference_tp} "
                f"--trust-remote-code"
            )
            node_commands.append(cmd)

        # FSDP trainer commands (one per GPU)
        for local_idx, fsdp_rank in enumerate(node.trainer_fsdp_ranks):
            local_gpu = node.trainer_gpus[local_idx]
            cmd = (
                f"CUDA_VISIBLE_DEVICES={local_gpu} "
                f"MASTER_ADDR={allocation.master_addr} "
                f"MASTER_PORT={allocation.master_port} "
                f"WORLD_SIZE={allocation.fsdp_world_size} "
                f"RANK={fsdp_rank} "
                f"LOCAL_RANK=0 "
                f"python -m {work_fn} "
                f"--config {config_path} "
                f"--is-rank-0 {1 if fsdp_rank == 0 else 0}"
            )
            node_commands.append(cmd)

        commands[node.public_ip] = node_commands

    return commands


async def launch_multi_node_training(
    config: MultiNodeConfig,
    train_config: Any,  # GRPOConfig
    branch: str = "main",
) -> ClusterAllocation:
    """Launch multi-node distributed training.

    Orchestrates the full multi-node training pipeline:
    1. Provision nodes via bifrost
    2. Deploy code to all nodes
    3. Compute GPU allocation
    4. Launch inference engines on all nodes
    5. Launch FSDP trainers on all nodes
    6. Return allocation for monitoring

    Args:
        config: Multi-node configuration
        train_config: Training configuration (GRPOConfig)
        branch: Git branch to deploy

    Returns:
        ClusterAllocation with all node information

    Example:
        >>> config = MultiNodeConfig(num_nodes=2, gpus_per_node=8)
        >>> allocation = await launch_multi_node_training(config, train_config)
        >>> print(f"Training on {allocation.fsdp_world_size} GPUs")
    """
    logger.info("=" * 60)
    logger.info("Multi-Node FSDP Training")
    logger.info("=" * 60)
    logger.info(f"Nodes: {config.num_nodes}")
    logger.info(f"GPUs per node: {config.gpus_per_node}")
    logger.info(f"Inference GPUs per node: {config.inference_gpus_per_node}")
    logger.info(f"Trainer GPUs per node: {config.trainer_gpus_per_node}")
    logger.info(f"Total FSDP ranks: {config.total_trainer_gpus}")

    # Phase 1: Provision nodes
    logger.info("\n--- Phase 1: Provisioning nodes ---")
    nodes = await provision_nodes(config)

    # Phase 2: Deploy code
    logger.info("\n--- Phase 2: Deploying code ---")
    workspaces = await deploy_to_nodes(nodes)

    # Phase 3: Compute allocation
    logger.info("\n--- Phase 3: Computing GPU allocation ---")
    node_ips = [instance.public_ip for _, instance in nodes]
    allocation = compute_cluster_allocation(config, node_ips)

    for node in allocation.nodes:
        logger.info(f"  Node {node.node_rank} ({node.public_ip}):")
        logger.info(f"    Inference: GPUs {node.inference_gpus}, ports {node.inference_ports}")
        logger.info(f"    Training: GPUs {node.trainer_gpus}, FSDP ranks {node.trainer_fsdp_ranks}")

    # Phase 4: Save config to nodes
    logger.info("\n--- Phase 4: Saving config to nodes ---")
    config_json = train_config.to_json() if hasattr(train_config, "to_json") else "{}"
    for (client, _), workspace in zip(nodes, workspaces, strict=False):
        config_path = f"{workspace}/rollouts/config.json"
        result = client.exec(f"echo '{config_json}' > {config_path}")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to save config: {result.stderr}")

    # Phase 5: Launch workers on all nodes
    logger.info("\n--- Phase 5: Launching workers ---")
    execute_commands_on_nodes(
        nodes=nodes,
        workspaces=workspaces,
        allocation=allocation,
        model_path=train_config.model.name,
    )

    logger.info("\n" + "=" * 60)
    logger.info(
        f"Cluster ready: {len(allocation.nodes)} nodes, {allocation.fsdp_world_size} FSDP ranks"
    )
    logger.info(f"Inference endpoints: {allocation.all_inference_endpoints}")
    logger.info("=" * 60)

    return allocation


def execute_commands_on_nodes(
    nodes: list[tuple[Any, Any]],
    workspaces: list[str],
    allocation: ClusterAllocation,
    model_path: str,
) -> None:
    """Execute inference and trainer commands on all nodes.

    Launches processes in tmux sessions so they persist.

    Args:
        nodes: List of (client, instance) from provision_nodes
        workspaces: List of workspace paths (one per node)
        allocation: Cluster allocation with GPU assignments
        model_path: HuggingFace model path
    """
    # Python command using uv (same as run.py)
    python_cmd = "~/.local/bin/uv run python"

    def launch_on_node(
        client: Any,
        instance: Any,
        workspace: str,
        node_alloc: NodeAllocation,
    ) -> None:
        """Launch all processes on a single node."""
        node_ip = instance.public_ip
        rollouts_dir = f"{workspace}/rollouts"

        # Launch inference engines
        for i, port in enumerate(node_alloc.inference_ports):
            gpu_start = i * allocation.config.inference_tp
            gpu_ids = ",".join(
                str(node_alloc.inference_gpus[gpu_start + j])
                for j in range(allocation.config.inference_tp)
            )

            session_name = f"inference_{i}"
            cmd = (
                f"cd {rollouts_dir} && "
                f"CUDA_VISIBLE_DEVICES={gpu_ids} "
                f"{python_cmd} -m sglang.launch_server "
                f"--model-path {model_path} "
                f"--port {port} "
                f"--tp {allocation.config.inference_tp} "
                f"--mem-fraction-static 0.85 "
                f"--trust-remote-code"
            )

            logger.info(f"  Node {node_ip}: Launching inference engine {i} on GPU {gpu_ids}")
            tmux_cmd = f"tmux new-session -d -s {session_name} '{cmd}'"
            result = client.exec(tmux_cmd)
            if result.exit_code != 0:
                logger.warning(f"Failed to launch inference {i}: {result.stderr}")

        # Launch FSDP trainers
        for local_idx, fsdp_rank in enumerate(node_alloc.trainer_fsdp_ranks):
            local_gpu = node_alloc.trainer_gpus[local_idx]

            session_name = f"trainer_{fsdp_rank}"
            inference_endpoints = ",".join(allocation.all_inference_endpoints)

            cmd = (
                f"cd {rollouts_dir} && "
                f"CUDA_VISIBLE_DEVICES={local_gpu} "
                f"MASTER_ADDR={allocation.master_addr} "
                f"MASTER_PORT={allocation.master_port} "
                f"WORLD_SIZE={allocation.fsdp_world_size} "
                f"RANK={fsdp_rank} "
                f"LOCAL_RANK=0 "
                f"INFERENCE_ENDPOINTS={inference_endpoints} "
                f"{python_cmd} -m rollouts.training.fsdp_worker "
                f"--config config.json "
                f"--is-rank-0 {1 if fsdp_rank == 0 else 0}"
            )

            logger.info(f"  Node {node_ip}: Launching FSDP rank {fsdp_rank} on GPU {local_gpu}")
            tmux_cmd = f"tmux new-session -d -s {session_name} '{cmd}'"
            result = client.exec(tmux_cmd)
            if result.exit_code != 0:
                logger.warning(f"Failed to launch trainer {fsdp_rank}: {result.stderr}")

    # Launch on all nodes sequentially (bifrost client is sync)
    for (client, instance), workspace, node_alloc in zip(
        nodes, workspaces, allocation.nodes, strict=False
    ):
        launch_on_node(client, instance, workspace, node_alloc)
