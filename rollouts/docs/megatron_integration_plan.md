# Megatron Integration Plan

## Goal

Add Megatron-Core support for training large MoE models like GLM-5 (700B-A40B) with full tensor/pipeline/expert parallelism.

Test path:
1. GLM-4.7-Flash (30B-A3B MoE) - small enough for quick iteration
2. GLM-5 (700B-A40B MoE) - production target

## Architecture Overview

**Key insight:** Use miniray for multi-node orchestration (not Ray like SLIME).

miniray provides:
- `Cluster` + `NodeConfig` for multi-node SSH management
- `Worker.send()` / `Worker.recv()` for coordination
- `wait_any(workers)` for async-style waiting
- Already used for log streaming in rollouts

```
rollouts/training/
├── megatron_worker.py           # NEW: miniray work function
├── backends/
│   ├── megatron_backend.py      # TrainingBackend impl (exists, needs factory)
│   └── megatron/                # NEW: Megatron utilities
│       ├── __init__.py
│       ├── initialize.py        # Process group setup
│       ├── model.py             # Model/optimizer factory
│       ├── weight_sync.py       # NCCL weight sync to inference
│       └── weight_conversion/   # Megatron → HF state dict
│           ├── __init__.py
│           ├── deepseekv3.py    # GLM-4.7-Flash (glm4moelite)
│           └── processors.py    # Common transforms
```

## Orchestration with miniray

### Why miniray instead of Ray?

SLIME uses Ray for:
1. Launching actors on remote nodes
2. `ray.get()` / `.remote()` for RPC
3. `ray.put()` for data distribution
4. Log aggregation

miniray does the same but lighter:
1. SSH + `WorkerServer` for remote launch
2. `worker.send()` / `worker.recv()` for RPC
3. Direct send for data (or NCCL broadcast)
4. Already have `LogsServer` for logs

### Multi-node flow

```python
# grpo.py (coordinator)
from miniray import Cluster, NodeConfig

# Launch Megatron workers on all nodes
cluster = Cluster(nodes=[
    NodeConfig("node1", num_workers=8, base_port=10000),  # 8 GPUs
    NodeConfig("node2", num_workers=8, base_port=10000),
])
workers = cluster.start(work_fn="rollouts.training.megatron_worker.train")

# Initialize all workers with their rank/config
for rank, worker in enumerate(workers):
    worker.send({
        "cmd": "init",
        "rank": rank,
        "world_size": len(workers),
        "config": megatron_config,
    })

# Training loop
for step in range(num_steps):
    # Get rollouts from inference
    rollouts = await inference.generate(prompts)

    # Send batch to rank 0 (it broadcasts via NCCL)
    workers[0].send({"cmd": "train_step", "batch": rollouts})

    # Wait for metrics from rank 0
    metrics = workers[0].recv()

    # Weight sync (rank 0 pushes to inference)
    if step % sync_interval == 0:
        workers[0].send({"cmd": "sync_weights"})
        workers[0].recv()  # Wait for completion

cluster.stop()
```

### megatron_worker.py

```python
def train(handle):
    """Miniray work function for Megatron training."""

    # Phase 1: Initialize
    init_msg = handle.recv()
    rank = init_msg["rank"]
    world_size = init_msg["world_size"]
    config = init_msg["config"]

    # Initialize Megatron process groups
    init_megatron(
        rank=rank,
        world_size=world_size,
        tensor_parallel_size=config["tp"],
        pipeline_parallel_size=config["pp"],
        expert_parallel_size=config["ep"],
    )

    # Create model/optimizer
    model, optimizer, scheduler = setup_megatron_model(
        model_name=config["model_name"],
        config=config,
    )

    backend = MegatronTrainingBackend(
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        config=config,
    )

    # Phase 2: Training loop
    while True:
        msg = handle.recv()

        if msg["cmd"] == "shutdown":
            break

        elif msg["cmd"] == "train_step":
            batch = msg["batch"]

            # Rank 0 broadcasts batch to other ranks via NCCL
            if rank == 0:
                broadcast_batch(batch)
            else:
                batch = receive_batch()

            # All ranks call forward_backward together
            metrics = backend.forward_backward(batch)
            backend.optim_step()

            # Only rank 0 reports metrics
            if rank == 0:
                handle.send(metrics)

        elif msg["cmd"] == "sync_weights":
            # Gather weights and sync to inference
            sync_weights_to_inference(backend, config["inference_endpoints"])
            if rank == 0:
                handle.send({"status": "synced"})
```

## Key Components

### 1. Process Group Initialization (`initialize.py`)

Megatron requires specific process group topology:
- Tensor parallel (TP): split attention/MLP across GPUs
- Pipeline parallel (PP): split layers across GPUs
- Expert parallel (EP): distribute MoE experts
- Data parallel (DP): replicate model across GPUs

```python
def init_megatron(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    *,
    master_addr: str,
    master_port: int,
    rank: int,
    world_size: int,
) -> None:
    """Initialize Megatron distributed groups.

    Must be called once per process before any Megatron operations.
    """
```

**From SLIME `initialize.py`:**
- `mpu.initialize_model_parallel()` with TP/PP/EP sizes
- Random seed setup per rank
- Tokenizer init
- Microbatch calculator init

### 2. Model/Optimizer Factory (`model.py`)

```python
def setup_megatron_model(
    model_name: str,
    config: MegatronConfig,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Create Megatron model wrapped in DDP with optimizer.

    Args:
        model_name: HuggingFace model name (e.g., "zai-org/GLM-4.7-Flash")
        config: MegatronConfig with parallelism settings
        role: "actor" or "critic" for RL

    Returns:
        (model_chunks, optimizer, scheduler)
    """
```

**Key insight from SLIME:**
SLIME uses `megatron.bridge.AutoBridge` which automatically converts HF models to Megatron format:

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=False)
provider.tensor_model_parallel_size = tp_size
provider.finalize()

model = get_model(provider.provide, ModelType.encoder_or_decoder)
```

This is the cleanest path - no need to manually create model provider functions for each architecture.

**From SLIME `model.py`:**
- `get_model()` from `megatron.training.training`
- `AutoBridge` for HF → Megatron conversion (supports GLM-4.7-Flash!)
- `get_megatron_optimizer()` with OptimizerConfig
- Checkpoint loading

### 3. Weight Conversion (`weight_conversion/`)

Megatron state dict keys differ from HuggingFace. For weight sync to SGLang, we need to convert.

```python
def convert_megatron_to_hf(
    model_name: str,
    megatron_state_dict: dict[str, Tensor],
    *,
    vocab_size: int,
    num_layers: int,
    num_attention_heads: int,
    hidden_size: int,
) -> dict[str, Tensor]:
    """Convert Megatron state dict to HuggingFace format.

    Handles:
    - Key name mapping (decoder.layers.N.* → model.layers.N.*)
    - Tensor reshaping (GLU split, QKV split)
    - TP gather (if needed)
    """
```

**Key mappings for GLM-4.7-Flash (deepseekv3 arch):**
- `module.module.embedding.word_embeddings.weight` → `model.embed_tokens.weight`
- `module.module.decoder.layers.N.mlp.linear_fc1.weight` → split to `gate_proj` + `up_proj`
- `module.module.decoder.layers.N.mlp.experts.M.linear_fc1.weightK` → `model.layers.N.mlp.experts.K.gate_proj.weight`

### 4. Weight Sync (`weight_sync.py`)

NCCL broadcast from trainer to SGLang inference workers.

```python
class MegatronWeightSyncer:
    """Sync weights from Megatron trainer to SGLang inference.

    Flow:
    1. Gather TP-sharded params on rank 0
    2. Convert to HF format
    3. NCCL broadcast to inference workers
    """

    def sync(self) -> None:
        """Push current weights to inference engine."""
```

**From SLIME:**
- `all_gather_param()` for TP gather
- `UpdateWeightFromDistributed` for NCCL broadcast
- Handles expert parallelism (EP) separately

### 5. Coordinator in `grpo.py`

For Megatron, `grpo.py` becomes a coordinator that talks to miniray workers:

```python
elif backend_name == "megatron":
    from miniray import Cluster, NodeConfig

    # Launch workers on all training nodes
    cluster = Cluster(nodes=[
        NodeConfig(
            host=node.host,
            num_workers=node.num_gpus,
            base_port=10000,
        )
        for node in config.trainer.nodes
    ])
    workers = cluster.start(work_fn="rollouts.training.megatron_worker.train")

    # Initialize all workers
    for rank, worker in enumerate(workers):
        worker.send({
            "cmd": "init",
            "rank": rank,
            "world_size": len(workers),
            "tp": config.trainer.tensor_parallel_size,
            "pp": config.trainer.pipeline_parallel_size,
            "ep": config.trainer.expert_parallel_size,
            "model_name": config.model.name,
            "inference_endpoints": inference_endpoints,
        })

    # Wrapper that talks to workers instead of local backend
    class MegatronRemoteBackend:
        def forward_backward(self, batch):
            workers[0].send({"cmd": "train_step", "batch": batch})
            return workers[0].recv()

        def sync_weights(self):
            workers[0].send({"cmd": "sync_weights"})
            return workers[0].recv()

    backend = MegatronRemoteBackend()
```

This keeps the training loop in `grpo.py` unchanged - it just calls `backend.forward_backward()` like before.

## Code Style Refactoring Notes

Apply principles from docs/code_style/:

### Tiger Style
- Assert function args/return values
- Explicit control flow, no hidden recursion
- Functions ≤70 lines
- Fail fast with clear error messages

### Semantic Compression (Casey)
- Don't abstract until we have 2+ use cases
- Inline code first, extract when pattern emerges
- "Shared stack frame" for related state (dataclasses)

### System Design (Sean)
- Minimize stateful components
- One module owns each piece of state
- Boring, well-tested patterns over clever tricks

### Specific Refactorings

**SLIME uses Ray actors heavily** - we'll remove Ray dependency, use plain Python with torch.distributed.

**SLIME has many magic strings** - we'll use enums/constants:
```python
# Before (SLIME)
if "linear_fc1" in name:

# After
class MegatronParamType(Enum):
    LINEAR_FC1 = "linear_fc1"
    LINEAR_FC2 = "linear_fc2"
    ...
```

**SLIME has long functions** - we'll split into focused helpers:
```python
# Before: 100+ line train_one_step
# After:
def _zero_grads(model, optimizer): ...
def _run_forward_backward(model, data_iterator, forward_step): ...
def _finalize_and_step(model, optimizer, scheduler): ...
```

## Test Config: GLM-4.7-Flash

```python
# examples/training_architecture/test_megatron_glm47flash.py

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=4,  # 2x inference (TP=2) + 2x trainer (TP=2)
    provider="runpod",
)

config = GRPOConfig(
    model=ModelConfig(name="zai-org/GLM-4.7-Flash"),
    trainer=TrainerConfig(
        backend="megatron",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        cuda_device_ids=(2, 3),
    ),
    inference=InferenceConfig(
        backend="sglang",
        cuda_device_ids=(0, 1),
        tensor_parallel_size=2,
    ),
    checkpoint=CheckpointConfig(
        num_steps=10,
        sync_weights_every=5,
        weight_sync_mode="nccl",
    ),
)
```

## Dependencies

Megatron-Core:
```bash
pip install megatron-core
# or
pip install git+https://github.com/NVIDIA/Megatron-LM.git
```

Note: Requires CUDA, NCCL, and compatible GPU drivers.

## Risks & Mitigations

1. **Megatron version compatibility**: Pin to specific Megatron-Core version in requirements
2. **Model provider mapping**: GLM-4.7-Flash may not have official Megatron model provider - may need custom
3. **Weight format mismatches**: Extensive testing of conversion with known checkpoints
4. **Multi-node complexity**: Start with single-node multi-GPU, validate before multi-node

## Implementation Order

1. ✅ Directory structure
2. `megatron_worker.py` - miniray work function (skeleton)
3. `initialize.py` - Megatron process group setup
4. `model.py` - model/optimizer factory using AutoBridge
5. Wire into `grpo.py` - coordinator that talks to miniray workers
6. Test basic forward/backward with GLM-4.7-Flash (single node, 2 GPUs)
7. `weight_conversion/deepseekv3.py` - HF format conversion
8. `weight_sync.py` - sync weights to SGLang inference
9. Test full loop with GLM-4.7-Flash (weight sync working)
10. Checkpoint save/load
11. Test multi-node with GLM-4.7-Flash
12. Test with GLM-5 (multi-node, full scale)
