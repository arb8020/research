# Simple RL Architecture: Training + Inference Separation

**TL;DR**: YES! Use 4 GPUs for training, 4 GPUs for inference. Much simpler than what I described. No multi-node FSDP needed for first version.

---

## The Confusion: What I Said vs What You Actually Need

### What I Overcomplicated

I described this:

```
Training Cluster (2 nodes, 16 GPUs)
├─ Node 1: 8 GPUs (FSDP training)
├─ Node 2: 8 GPUs (FSDP training)

Inference Cluster (4 nodes, 32 GPUs)
├─ Node 3: 8 GPUs (rollouts)
├─ Node 4: 8 GPUs (rollouts)
├─ Node 5: 8 GPUs (rollouts)
├─ Node 6: 8 GPUs (rollouts)
```

**This is overkill for getting started!**

---

### What You Actually Want (Much Simpler!)

```
Single Node (8 GPUs)
├─ Training: GPUs 0-3 (4 GPUs for FSDP)
└─ Inference: GPUs 4-7 (4 GPUs for rollout generation)
```

**OR**

```
Two Nodes (8 GPUs each)
├─ Node 1: 8 GPUs for training (FSDP)
└─ Node 2: 8 GPUs for inference (rollout generation)
```

**This is what Slime does with `--colocate`!**

---

## Slime's Actual Architecture (Looking Closer)

### Colocated Mode (Default)

From Slime's test script:

```bash
python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \                     # ← Training + inference on SAME GPUs!
   --rollout-num-gpus 8             # Use all 8 GPUs for inference too
```

**What `--colocate` does:**
- Training uses GPUs 0-7 (FSDP)
- Inference ALSO uses GPUs 0-7 (time-multiplexed)
- They share memory/compute

**Architecture:**
```
Node 1 (8 GPUs)
├─ GPUs 0-7: FSDP training
└─ GPUs 0-7: SGLang inference engines (shared!)
```

**Workflow:**
1. Pause training
2. Generate rollouts (use all GPUs for inference)
3. Resume training with new rollouts

---

### Distributed Mode (What You Want!)

From Slime's FSDP test:

```bash
python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \    # 2 GPUs for training
   --train-backend fsdp \
   --rollout-num-gpus 2             # 2 GPUs for inference
```

**What this means:**
```
Node 1 (8 GPUs)
├─ GPUs 0-1: FSDP training (2 GPUs)
└─ GPUs 2-3: SGLang inference (2 GPUs)
```

**OR if you want full separation:**

```
Node 1 (8 GPUs) - Training only
└─ GPUs 0-7: FSDP training

Node 2 (8 GPUs) - Inference only
└─ GPUs 0-7: SGLang inference
```

---

## Three Simple Architectures (Pick One!)

### Architecture 1: Single-Node, Split GPUs (Simplest!)

**Hardware:** 1 node, 8 GPUs

**Setup:**
```python
# Training on GPUs 0-3
training_backend = PyTorchTrainingBackend(
    model_name="Qwen/Qwen2.5-0.5B",
    world_size=4,
    gpu_ids=[0, 1, 2, 3],  # First 4 GPUs
    strategy="fsdp",
)

# Inference on GPUs 4-7
inference_engines = [
    SGLangEngine(model_name="Qwen/Qwen2.5-0.5B", gpu_id=i)
    for i in [4, 5, 6, 7]  # Last 4 GPUs
]

# Run RL loop
await run_rl_training(
    backend=training_backend,
    rollout_manager=rollout_manager,
    inference_engines=inference_engines,
    config=config,
)
```

**Pros:**
- ✅ Simple (one machine)
- ✅ No SSH/network needed
- ✅ Training and inference don't interfere

**Cons:**
- ❌ Limited scale (max 8 GPUs total)
- ❌ Training limited to 4 GPUs

**When to use:** Getting started, small models (< 7B)

---

### Architecture 2: Two Nodes, Full Separation (Best for Development!)

**Hardware:** 2 nodes, 8 GPUs each

**Setup:**
```python
# Node 1: Training (8 GPUs, single-node FSDP)
training_backend = PyTorchTrainingBackend(
    model_name="Qwen/Qwen2.5-7B",
    world_size=8,
    gpu_ids=list(range(8)),  # All 8 GPUs on node 1
    strategy="fsdp",
)

# Node 2: Inference (8 GPUs, using MiniRay!)
from miniray import Cluster, NodeConfig

inference_cluster = Cluster(nodes=[
    NodeConfig("node2", num_workers=8),
])
inference_workers = inference_cluster.start(work_fn="module.inference_worker")

# Run RL loop
await run_rl_training(
    backend=training_backend,
    rollout_manager=rollout_manager,
    inference_workers=inference_workers,  # Remote workers!
    config=config,
)
```

**Pros:**
- ✅ Full 8 GPUs for training
- ✅ Full 8 GPUs for inference
- ✅ No resource contention
- ✅ Still uses single-node FSDP (no multi-node complexity!)

**Cons:**
- ❌ Need 2 nodes
- ❌ Training still limited to 8 GPUs

**When to use:** Medium models (7B-13B), want max throughput

---

### Architecture 3: Multi-Node Training + Dedicated Inference (Production)

**Hardware:** 4 nodes total (2 for training, 2 for inference)

**Setup:**
```python
from miniray import Cluster, NodeConfig, create_nccl_configs

# Nodes 1-2: Multi-node FSDP training (16 GPUs total)
training_cluster = Cluster(nodes=[
    NodeConfig("train-node1", num_workers=8),
    NodeConfig("train-node2", num_workers=8),
])
training_workers = training_cluster.start(work_fn="module.train_worker")

# Setup NCCL for multi-node FSDP
nccl_configs = create_nccl_configs(
    master_addr="train-node1",
    nodes=[("train-node1", 8), ("train-node2", 8)],
)
for worker, config in zip(training_workers, nccl_configs):
    worker.send({"cmd": "init_nccl", "config": config.__dict__})

# Nodes 3-4: Inference (16 GPUs total)
inference_cluster = Cluster(nodes=[
    NodeConfig("inf-node1", num_workers=8),
    NodeConfig("inf-node2", num_workers=8),
])
inference_workers = inference_cluster.start(work_fn="module.inference_worker")

# Run RL loop
await run_rl_training(
    backend=MultiNodeFSDPBackend(training_workers),
    rollout_manager=rollout_manager,
    inference_workers=inference_workers,
    config=config,
)
```

**Pros:**
- ✅ 16 GPUs for training (larger models)
- ✅ 16 GPUs for inference (high throughput)
- ✅ Scales to 30B+ models

**Cons:**
- ❌ Need 4 nodes
- ❌ Complex setup (multi-node FSDP)

**When to use:** Large models (30B+), production

---

## What Slime Actually Does (The Real Answer)

Looking at their test scripts more carefully:

### Test 1: Colocated (Single Node)
```bash
# test-qwen2.5-0.5B-gsm8k.sh
ray start --head --num-gpus 4

python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \                      # ← SAME GPUs for training + inference
   --rollout-num-gpus-per-engine 1
```

**Architecture:**
```
Node 1 (4 GPUs)
└─ GPUs 0-3: Training (Megatron) + Inference (SGLang)
   - Time-multiplexed: generate rollouts, then train
```

### Test 2: Distributed (Separate GPUs, Single Node)
```bash
# test_qwen3-0.6B_fsdp_distributed.sh
ray start --head --num-gpus 4

python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \    # ← 2 GPUs for training
   --train-backend fsdp \
   --rollout-num-gpus 2             # ← 2 GPUs for inference
```

**Architecture:**
```
Node 1 (4 GPUs)
├─ GPUs 0-1: FSDP training
└─ GPUs 2-3: SGLang inference
```

**This is what you want!** ☝️

---

## Recommended Starting Point: Architecture 2 (Two Nodes)

**Why:**
1. ✅ **No multi-node FSDP needed** - Training uses single-node FSDP (already working!)
2. ✅ **Full GPU utilization** - 8 GPUs training, 8 GPUs inference
3. ✅ **Simple with MiniRay** - Just launch inference workers on node 2
4. ✅ **Easy to test** - Start with 1 node (split GPUs), scale to 2 nodes later

**Concrete Example:**

```python
# examples/train_rl_two_nodes.py
import asyncio
from pathlib import Path

from miniray import Cluster, NodeConfig
from rollouts.training.backends.pytorch import PyTorchTrainingBackend
from rollouts.training.loops.rl_loop import run_rl_training
from rollouts.training.types import RLTrainingConfig

async def main():
    # ════════════════════════════════════════════════════════════════
    # NODE 1: Training (8 GPUs, single-node FSDP)
    # ════════════════════════════════════════════════════════════════
    training_backend = PyTorchTrainingBackend(
        model_name="Qwen/Qwen2.5-0.5B",
        world_size=8,
        strategy="fsdp",
    )

    # ════════════════════════════════════════════════════════════════
    # NODE 2: Inference (8 GPUs, MiniRay)
    # ════════════════════════════════════════════════════════════════
    inference_cluster = Cluster(nodes=[
        NodeConfig("inference-node", num_workers=8),
    ])
    inference_workers = inference_cluster.start(
        work_fn="examples.inference_worker.run"
    )

    # Initialize inference engines on all workers
    for i, worker in enumerate(inference_workers):
        worker.send({
            "cmd": "init_engine",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "gpu_id": i,
        })

    # ════════════════════════════════════════════════════════════════
    # RL TRAINING LOOP
    # ════════════════════════════════════════════════════════════════
    config = RLTrainingConfig(
        num_steps=1000,
        sync_every=10,
    )

    # Create rollout manager that uses remote workers
    rollout_manager = RemoteRolloutManager(
        workers=inference_workers,
        data_buffer=data_buffer,
        config=rollout_config,
    )

    # Run training!
    metrics = await run_rl_training(
        backend=training_backend,
        data_buffer=data_buffer,
        rollout_manager=rollout_manager,
        inference_engines=[],  # Using workers instead
        config=config,
    )

    # Cleanup
    inference_cluster.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

**Inference worker (on node 2):**
```python
# examples/inference_worker.py
def run(handle, rank, world_size):
    """Inference worker function (runs on node 2)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Set GPU
    torch.cuda.set_device(rank)

    # Wait for init
    msg = handle.recv(max_size=1024)
    assert msg["cmd"] == "init_engine"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        msg["model_name"],
        torch_dtype=torch.bfloat16,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(msg["model_name"])

    # Inference loop
    while True:
        msg = handle.recv(max_size=10 * 1024 * 1024)

        if msg["cmd"] == "generate":
            # Generate completions
            prompts = msg["prompts"]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=1024)
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            handle.send({"responses": responses})

        elif msg["cmd"] == "reload_weights":
            # Load new checkpoint
            model.load_state_dict(torch.load(msg["checkpoint_path"]))
            handle.send({"status": "reloaded"})

        elif msg["cmd"] == "shutdown":
            break
```

---

## Updated Gap Analysis (Much Simpler!)

### ✅ What We Have (Already Working)

1. ✅ Single-node FSDP training (rollouts/training/backends/fsdp.py)
2. ✅ MiniRay for launching remote workers
3. ✅ Weight sync system
4. ✅ Training loop structure

### ❌ What We Need (Simpler List!)

1. **GRPO loss** (1 week) - Still need this
2. **Reward models** (3-5 days) - Still need this
3. **Remote rollout manager** (2-3 days) - NEW: Coordinate inference across nodes
4. **Config cleanup** (2 days) - Still helpful

### ❌ What We DON'T Need (Yet!)

1. ~~Multi-node FSDP~~ - Not needed! Use single-node FSDP on node 1
2. ~~Complex cluster orchestration~~ - MiniRay already handles this
3. ~~Placement groups~~ - Ray-specific, we don't need

---

## Comparison: What I Said vs What You Need

| What I Said | What You Actually Need |
|-------------|------------------------|
| Multi-node FSDP (2-4 nodes for training) | ❌ Single-node FSDP (8 GPUs) |
| Multi-node inference cluster (4-8 nodes) | ❌ Single node for inference (8 GPUs) |
| Complex NCCL setup across nodes | ❌ Standard FSDP (already works) |
| 3 weeks of work | ✅ 1-2 weeks (much simpler!) |

---

## Revised Timeline (1-2 Weeks)

### Week 1: Core RL + Remote Inference

**Day 1-2: GRPO Loss**
```python
# rollouts/training/rl_losses.py
def grpo_loss(...):
    # PPO clipping + KL
    pass
```

**Day 3-4: Reward Models**
```python
# rollouts/training/reward_models.py
class MathRewardModel:
    def score(prompt, response) -> float:
        pass
```

**Day 5-7: Remote Rollout Manager**
```python
# rollouts/training/rollout_gen/remote_rollout_manager.py
class RemoteRolloutManager:
    """Rollout manager using MiniRay workers."""

    def __init__(self, workers: list[RemoteWorker], ...):
        self.workers = workers

    async def generate_batch(self) -> RolloutBatch:
        # Send prompts to all workers
        prompts = self.data_buffer.sample(batch_size)
        batch_size_per_worker = len(prompts) // len(self.workers)

        for i, worker in enumerate(self.workers):
            worker_prompts = prompts[i*batch_size_per_worker:(i+1)*batch_size_per_worker]
            worker.send({"cmd": "generate", "prompts": worker_prompts})

        # Collect responses
        all_responses = []
        for worker in self.workers:
            result = worker.recv(max_size=10 * 1024 * 1024)
            all_responses.extend(result["responses"])

        return RolloutBatch(...)
```

### Week 2: Integration + Testing

**Day 8-9: End-to-end integration**
- Wire up GRPO loss in training backend
- Test two-node setup (1 training, 1 inference)

**Day 10: Debugging + Polish**
- Fix issues
- Add logging

**Done!** 🎉

---

## Summary: You Were Right!

**Your question:**
> Couldn't we use one node for training and one for inference?

**Answer:** YES! That's exactly what you should do:

1. **Node 1**: 8 GPUs for training (single-node FSDP - already working!)
2. **Node 2**: 8 GPUs for inference (MiniRay workers - already working!)

**No multi-node FSDP needed!** Just:
- GRPO loss (1 week)
- Reward models (3-5 days)
- Remote rollout manager (2-3 days)

**Total: 1-2 weeks, not 3 weeks.**

I overcomplicated it by thinking about scaling to 30B+ models immediately. For getting started (< 7B models), your simple architecture is perfect! 🚀
