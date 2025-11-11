# RL Loop Gap Analysis: Slime vs Our Setup

**TL;DR**: We have most pieces! Missing: (1) GRPO implementation, (2) Reward model integration, (3) Multi-node backends, (4) Config unification. ~2-3 weeks to working RL.

---

## Full Slime Configuration Breakdown

### What Slime's Bash Scripts Show

From `test-qwen2.5-0.5B-gsm8k.sh`:

```bash
# ════════════════════════════════════════════════════════════════
# INFRASTRUCTURE
# ════════════════════════════════════════════════════════════════
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars": {...}}'

# ════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (from qwen2.5-0.5B.sh)
# ════════════════════════════════════════════════════════════════
MODEL_ARGS=(
   --swiglu
   --num-layers 24
   --hidden-size 896
   --ffn-hidden-size 4864
   --num-attention-heads 14
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --group-query-attention
   --num-query-groups 2
   --vocab-size 151936
)

# ════════════════════════════════════════════════════════════════
# CHECKPOINTS
# ════════════════════════════════════════════════════════════════
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct/        # Initial weights
   --ref-load /root/Qwen2.5-0.5B-Instruct_torch_dist/  # Reference model (for KL)
)

# ════════════════════════════════════════════════════════════════
# ROLLOUT GENERATION (CRITICAL!)
# ════════════════════════════════════════════════════════════════
ROLLOUT_ARGS=(
   # Data
   --prompt-data gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   # Reward Model
   --rm-type math                         # ← CRITICAL: Reward model type

   # Generation config
   --num-rollout 3000                     # Total rollouts to generate
   --rollout-batch-size 32                # Prompts per batch
   --n-samples-per-prompt 8               # GRPO: 8 completions per prompt
   --rollout-max-response-len 1024
   --rollout-temperature 0.8

   # Dynamic sampling (SLIME-specific)
   --over-sampling-batch-size 64          # Generate more, filter to best
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std

   # Training batch
   --global-batch-size 256                # Actual training batch size
)

# ════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data gsm8k gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
)

# ════════════════════════════════════════════════════════════════
# PERFORMANCE / PARALLELISM (Megatron-specific)
# ════════════════════════════════════════════════════════════════
PERF_ARGS=(
   --tensor-model-parallel-size 1         # TP (within node)
   --sequence-parallel                     # Sequence parallel
   --pipeline-model-parallel-size 1       # PP (across nodes)
   --context-parallel-size 1              # CP
   --expert-model-parallel-size 1         # EP (MoE)
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size               # Pack sequences efficiently
   --max-tokens-per-gpu 9216              # Memory limit
)

# ════════════════════════════════════════════════════════════════
# GRPO ALGORITHM
# ════════════════════════════════════════════════════════════════
GRPO_ARGS=(
   --advantage-estimator grpo             # ← CRITICAL: GRPO algorithm

   # KL divergence (from reference model)
   --use-kl-loss
   --kl-loss-coef 0.00                    # Weight for KL term
   --kl-loss-type low_var_kl

   # Entropy regularization
   --entropy-coef 0.00

   # PPO clipping
   --eps-clip 0.2                         # Lower clip
   --eps-clip-high 0.28                   # Upper clip
)

# ════════════════════════════════════════════════════════════════
# OPTIMIZER
# ════════════════════════════════════════════════════════════════
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-test
   --wandb-group test-qwen2.5-0.5B-gsm8k
)

# ════════════════════════════════════════════════════════════════
# INFERENCE ENGINE (SGLang)
# ════════════════════════════════════════════════════════════════
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1        # GPUs per inference engine
   --sglang-mem-fraction-static 0.7       # VRAM allocation
)

# ════════════════════════════════════════════════════════════════
# MISC (Megatron)
# ════════════════════════════════════════════════════════════════
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ════════════════════════════════════════════════════════════════
# EXECUTION
# ════════════════════════════════════════════════════════════════
python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \                            # Training + inference on same GPUs
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
```

---

## What We Have vs What We Need

### ✅ What We Have (Working)

#### 1. ✅ Training Infrastructure
```python
# rollouts/rollouts/training/backends/pytorch.py
class PyTorchTrainingBackend:
    async def forward_backward(self, batch) -> TrainFuture[Dict[str, float]]
    async def optim_step() -> TrainFuture[Dict[str, float]]
    async def save_checkpoint(step, metrics) -> Path
```

**Status**: Working for SFT, needs GRPO loss

#### 2. ✅ RL Training Loop Structure
```python
# rollouts/rollouts/training/loops/rl_loop.py
async def run_rl_training(
    backend: PyTorchTrainingBackend,
    data_buffer: DataBuffer,
    rollout_manager: AsyncRolloutManager,
    inference_engines: List[InferenceEngine],
    config: RLTrainingConfig,
):
    for step in range(config.num_steps):
        # 1. Generate rollouts
        batch = await rollout_manager.generate_batch()

        # 2. Compute rewards
        rewards = [compute_reward(s) for s in batch.samples]

        # 3. Train
        fwd_metrics = await backend.forward_backward(rl_batch).result()
        opt_metrics = await backend.optim_step().result()

        # 4. Weight sync
        if step % config.sync_every == 0:
            ckpt_path = await backend.save_checkpoint(step, metrics)
            await sync_weights_to_engines(inference_engines, ckpt_path)
```

**Status**: Structure correct, missing GRPO loss implementation

#### 3. ✅ Data Types
```python
# rollouts/rollouts/training/types.py
@dataclass
class Sample:
    prompt: str | list[dict[str, str]]
    response: str
    tokens: list[int]
    loss_mask: list[float]
    reward: float
    metadata: dict[str, Any]

@dataclass(frozen=True)
class RLTrainingConfig:
    num_steps: int
    sync_every: int = 10
    baseline: float = 0.0
    log_every: int = 10
    checkpoint_every: int = 100
```

**Status**: Good, but missing some Slime config fields

#### 4. ✅ Weight Sync
```python
# rollouts/rollouts/training/weight_sync.py
async def sync_weights_to_engines(
    engines: List[InferenceEngine],
    checkpoint_path: str,
):
    for engine in engines:
        await engine.reload_weights(checkpoint_path)
```

**Status**: Working

#### 5. ✅ Multi-Node Orchestration (NEW!)
```python
# miniray/cluster.py
from miniray import Cluster, NodeConfig, create_nccl_configs

cluster = Cluster(nodes=[
    NodeConfig("node1", num_workers=8),
    NodeConfig("node2", num_workers=8),
])
workers = cluster.start(work_fn="module.train_worker")
```

**Status**: Working

---

### ❌ What We're Missing (Critical Gaps)

#### 1. ❌ GRPO Loss Implementation

**What Slime has:**
```python
# Slime's train.py (conceptual)
def grpo_loss(
    log_probs,          # Policy log probs
    ref_log_probs,      # Reference model log probs
    rewards,            # Rewards from RM
    advantages,         # GRPO advantages
    eps_clip,           # PPO clip epsilon
    kl_coef,            # KL penalty weight
):
    # Group-relative policy optimization
    ratio = torch.exp(log_probs - ref_log_probs)

    # PPO clipping
    clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)

    # Policy loss (maximize advantage)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    # KL penalty
    kl_div = (log_probs - ref_log_probs).mean()

    # Total loss
    loss = policy_loss + kl_coef * kl_div

    return loss
```

**What we have:**
```python
# rollouts/rollouts/training/rl_losses.py
# EMPTY! No GRPO implementation yet
```

**Gap**: Need to implement GRPO loss (1 week)

---

#### 2. ❌ Reward Model Integration

**What Slime has:**
```bash
ROLLOUT_ARGS=(
   --rm-type math                    # Built-in math reward model
   --rm-type deepscaler              # DeepScaler reward model
   # OR custom --rm-path /path/to/model
)
```

**What we have:**
```python
# rollouts/rollouts/training/loops/rl_loop.py
def compute_reward(sample: Sample) -> float:
    """Pure function: Compute reward from sample."""
    # Simple environment-based grading
    if sample.metadata.get("correct", False):
        return 1.0
    return 0.0
```

**Gap**:
- ✅ We have environment grading (correctness checking)
- ❌ We don't have reward model inference
- ❌ We don't have reward model types (math, code, etc.)

**Solution**: Add reward model support (3-5 days)

```python
# NEW: rollouts/rollouts/training/reward_models.py
class RewardModel(Protocol):
    def score(self, prompt: str, response: str) -> float: ...

class MathRewardModel:
    """Grade math solutions for correctness."""
    def score(self, prompt: str, response: str) -> float:
        # Extract answer, check against label
        pass

class HuggingFaceRewardModel:
    """Generic HF reward model."""
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def score(self, prompt: str, response: str) -> float:
        # Run inference
        pass

# Usage in rl_loop.py:
def compute_reward(sample: Sample, rm: RewardModel) -> float:
    return rm.score(sample.prompt, sample.response)
```

---

#### 3. ❌ GRPO-Specific Rollout Generation

**What Slime has:**
```bash
ROLLOUT_ARGS=(
   --n-samples-per-prompt 8              # Generate 8 completions per prompt
   --over-sampling-batch-size 64         # Generate extra, filter to best
   --dynamic-sampling-filter-path slime.rollout.filter_hub...
)
```

**What we have:**
```python
# rollouts/rollouts/training/types.py
@dataclass(frozen=True)
class RolloutConfig:
    batch_size: int
    n_samples_per_prompt: int = 1        # ✅ We have this!
    over_sampling_factor: float = 1.0    # ✅ We have this!
    generate_fn: Optional[Callable] = None
    reward_fn: Optional[Callable] = None
    filter_fn: Optional[Callable] = None  # ✅ We have this!
```

**Status**: ✅ We have the config fields!

**Gap**: Need to use them correctly in rollout generation (2-3 days)

```python
# rollouts/rollouts/training/rollout_gen/async_rollout_manager.py
async def generate_batch(self) -> RolloutBatch:
    # Generate n_samples_per_prompt completions for each prompt
    prompts = self.data_buffer.sample(self.config.batch_size)

    # Expand: each prompt → n_samples_per_prompt samples
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * self.config.n_samples_per_prompt)

    # Generate all
    samples = await self.config.generate_fn(expanded_prompts, self.config)

    # Group by prompt (for GRPO advantage computation)
    grouped_samples = group_by_prompt(samples, self.config.n_samples_per_prompt)

    # Filter (dynamic sampling)
    if self.config.filter_fn:
        grouped_samples = [
            self.config.filter_fn(group)
            for group in grouped_samples
        ]

    return RolloutBatch(...)
```

---

#### 4. ❌ Multi-Node Training Backends

**What Slime has:**
```bash
python3 train.py \
   --actor-num-nodes 1 \              # Can be 2, 4, 8...
   --actor-num-gpus-per-node 4 \
   --train-backend fsdp                # FSDP across nodes
```

**What we have:**
```python
# rollouts/rollouts/training/backends/fsdp.py
class FSDPBackend:
    # Only works single-node!
    pass
```

**Gap**: Need multi-node FSDP backend using MiniRay (1 week)

**Solution** (from WHAT_MINIRAY_UNLOCKS.md):
```python
# NEW: rollouts/rollouts/training/backends/miniray_fsdp.py
from miniray import Cluster, NodeConfig, setup_nccl_env, set_gpu_affinity

class MultiNodeFSDPBackend:
    """FSDP backend using MiniRay for multi-node."""

    def __init__(self, nodes: list[NodeConfig], model_name: str):
        self.cluster = Cluster(nodes=nodes)
        self.workers = self.cluster.start(work_fn="__main__.train_worker")
        self._setup_nccl()

    def _setup_nccl(self):
        from miniray import create_nccl_configs
        configs = create_nccl_configs(
            master_addr=self.cluster.nodes[0].host,
            nodes=[(n.host, n.num_workers) for n in self.cluster.nodes],
        )
        for worker, config in zip(self.workers, configs):
            worker.send({"cmd": "init_nccl", "config": config.__dict__})

    async def forward_backward(self, batch):
        # Broadcast batch to all workers
        for worker in self.workers:
            worker.send({"cmd": "forward_backward", "batch": batch})

        # Collect metrics from rank 0 only
        metrics = self.workers[0].recv(max_size=1024)
        return metrics
```

---

#### 5. ❌ Config Unification

**What Slime has:**
- Single `train.py` with all args
- Bash scripts compose arg groups
- Clear separation of concerns

**What we have:**
- `RLTrainingConfig` (training loop)
- `RolloutConfig` (rollout generation)
- Backend config (FSDP, optimizer)
- No unified config object

**Gap**: Need master config that combines everything (2 days)

```python
# NEW: rollouts/rollouts/training/config.py
@dataclass(frozen=True)
class RLMasterConfig:
    """Master config for RL training (Slime-style)."""

    # Model
    model_name: str
    hf_checkpoint: Path
    ref_checkpoint: Optional[Path] = None  # For KL divergence

    # Training
    num_steps: int
    global_batch_size: int

    # Rollout generation
    rollout_batch_size: int
    n_samples_per_prompt: int
    rollout_max_response_len: int
    rollout_temperature: float

    # GRPO
    advantage_estimator: str = "grpo"
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28

    # Optimizer
    lr: float = 1e-6
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98

    # Reward model
    rm_type: str = "math"  # "math", "code", "hf:<model_name>"

    # Evaluation
    eval_interval: int = 20
    eval_prompt_data: Optional[Path] = None

    # Infrastructure
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    colocate: bool = False  # Training + inference on same GPUs?

    # Logging
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_group: Optional[str] = None
```

---

## Side-by-Side Comparison Table

| Feature | Slime | Our Setup | Status | Work Needed |
|---------|-------|-----------|--------|-------------|
| **Training Loop** | ✅ train.py | ✅ rl_loop.py | ✅ Working | None |
| **GRPO Loss** | ✅ Implemented | ❌ Missing | ❌ Gap | 1 week |
| **Reward Models** | ✅ math, deepscaler, custom | ⚠️ Only env grading | ⚠️ Partial | 3-5 days |
| **Multi-Sample Generation** | ✅ n-samples-per-prompt | ✅ Config exists | ⚠️ Not used | 2-3 days |
| **Dynamic Sampling** | ✅ filter_fn | ✅ Config exists | ⚠️ Not used | 2-3 days |
| **Multi-Node Training** | ✅ FSDP across nodes | ❌ Single-node only | ❌ Gap | 1 week |
| **Multi-Node Orchestration** | ✅ Ray | ✅ MiniRay | ✅ Working | None |
| **Weight Sync** | ✅ D5 sync | ✅ sync_weights_to_engines | ✅ Working | None |
| **Evaluation** | ✅ eval-interval | ❌ Not implemented | ❌ Gap | 3 days |
| **Config System** | ✅ Unified args | ⚠️ Split configs | ⚠️ Messy | 2 days |
| **Logging** | ✅ WandB | ⚠️ JSONLLogger | ⚠️ Basic | 1 day |

---

## Concrete Action Plan: 2-3 Week Timeline

### Week 1: Core RL Components

**Day 1-2: GRPO Loss**
```python
# NEW: rollouts/rollouts/training/rl_losses.py
def grpo_loss(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
) -> Dict[str, torch.Tensor]:
    """GRPO loss with PPO clipping."""
    ratio = torch.exp(policy_log_probs - ref_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)

    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    return {"policy_loss": policy_loss, "ratio": ratio.mean()}
```

**Day 3-4: Reward Models**
```python
# NEW: rollouts/rollouts/training/reward_models.py
class MathRewardModel:
    def score(self, prompt: str, response: str) -> float:
        # Extract answer, check correctness
        pass

class HuggingFaceRewardModel:
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def score(self, prompt: str, response: str) -> float:
        # Run inference
        pass
```

**Day 5: GRPO Rollout Generation**
```python
# UPDATE: rollouts/rollouts/training/rollout_gen/async_rollout_manager.py
async def generate_batch(self) -> RolloutBatch:
    # Expand prompts (n_samples_per_prompt)
    # Group by prompt
    # Filter (dynamic sampling)
    pass
```

### Week 2: Multi-Node Training

**Day 6-8: Multi-Node FSDP Backend**
```python
# NEW: rollouts/rollouts/training/backends/miniray_fsdp.py
class MultiNodeFSDPBackend:
    def __init__(self, nodes: list[NodeConfig], model_name: str):
        self.cluster = Cluster(nodes=nodes)
        self.workers = self.cluster.start(work_fn="__main__.train_worker")
        self._setup_nccl()

    async def forward_backward(self, batch):
        # Broadcast to workers
        # Collect from rank 0
        pass
```

**Day 9-10: Test Multi-Node**
- Test on 2 nodes
- Verify NCCL works
- Benchmark vs single-node

### Week 3: Polish & Integration

**Day 11-12: Config Unification**
```python
# NEW: rollouts/rollouts/training/config.py
@dataclass(frozen=True)
class RLMasterConfig:
    # All Slime args in one place
    pass
```

**Day 13: Evaluation**
```python
# UPDATE: rollouts/rollouts/training/loops/rl_loop.py
if step % config.eval_interval == 0:
    eval_metrics = await evaluate(eval_prompts, backend, config)
```

**Day 14: WandB Integration**
```python
# UPDATE: rollouts/rollouts/training/metrics.py
class WandBLogger(MetricsLogger):
    def log(self, metrics, step):
        wandb.log(metrics, step=step)
```

**Day 15: End-to-End Test**
- Run full RL loop
- Single-node (4 GPUs)
- Multi-node (2x4 GPUs)
- Verify convergence

---

## Minimal Example: What a Config Would Look Like

### Slime's Config (Bash)
```bash
python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct/ \
   --ref-load /root/Qwen2.5-0.5B-Instruct_torch_dist/ \
   --prompt-data gsm8k/train.parquet \
   --rm-type math \
   --num-rollout 3000 \
   --rollout-batch-size 32 \
   --n-samples-per-prompt 8 \
   --rollout-temperature 0.8 \
   --global-batch-size 256 \
   --advantage-estimator grpo \
   --use-kl-loss \
   --kl-loss-coef 0.00 \
   --eps-clip 0.2 \
   --optimizer adam \
   --lr 1e-6 \
   --use-wandb \
   --wandb-project slime-test
```

### Our Config (Python, after Week 3)
```python
# examples/train_rl_gsm8k.py
from rollouts.training.config import RLMasterConfig
from rollouts.training.loops.rl_loop import run_rl_training

config = RLMasterConfig(
    # Model
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    hf_checkpoint=Path("/root/Qwen2.5-0.5B-Instruct"),
    ref_checkpoint=Path("/root/Qwen2.5-0.5B-Instruct_torch_dist"),

    # Training
    num_steps=3000,
    global_batch_size=256,

    # Rollout
    rollout_batch_size=32,
    n_samples_per_prompt=8,
    rollout_max_response_len=1024,
    rollout_temperature=0.8,

    # GRPO
    advantage_estimator="grpo",
    use_kl_loss=True,
    kl_loss_coef=0.0,
    eps_clip=0.2,

    # Optimizer
    lr=1e-6,

    # Reward model
    rm_type="math",

    # Infrastructure
    actor_num_nodes=1,
    actor_num_gpus_per_node=4,
    colocate=True,

    # Logging
    use_wandb=True,
    wandb_project="slime-test",
)

# Run!
metrics = await run_rl_training(config)
```

---

## Summary: What's Missing

### Critical (Must Have):
1. **GRPO loss implementation** (1 week)
2. **Multi-node FSDP backend** (1 week)
3. **Config unification** (2 days)

### Important (Should Have):
4. **Reward model integration** (3-5 days)
5. **GRPO rollout generation** (2-3 days)
6. **Evaluation loop** (3 days)

### Nice to Have:
7. **WandB logging** (1 day)
8. **Dynamic sampling filters** (2 days)

---

## Timeline Summary

- **Week 1**: Core RL (GRPO loss, reward models, rollout gen)
- **Week 2**: Multi-node (MiniRay FSDP backend)
- **Week 3**: Polish (config, eval, logging)

**Total**: ~2-3 weeks to working RL loop

**First milestone** (single-node): ~1 week
**Second milestone** (multi-node): ~2 weeks
**Production-ready**: ~3 weeks

---

## What to Prioritize

**For single-node RL (1 week)**:
1. GRPO loss
2. Math reward model
3. GRPO rollout generation
4. Test on 1 node

**For multi-node RL (2 weeks)**:
5. MultiNodeFSDPBackend
6. Test on 2 nodes
7. Config unification

**For production (3 weeks)**:
8. Evaluation
9. WandB
10. Dynamic sampling

**Start with #1-4 for fastest path to working RL!**
