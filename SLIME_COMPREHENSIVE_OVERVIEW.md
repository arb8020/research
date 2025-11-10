# Slime: Comprehensive RL Training Framework Overview

## Executive Summary

**Slime** is an LLM post-training framework specifically designed for **Reinforcement Learning (RL) scaling**. Created by THUDM and used to train GLM-4.5 and GLM-4.6, slime provides a production-grade system that combines high-performance training infrastructure with flexible data generation capabilities. It bridges Megatron (training) and SGLang (inference) into a unified, efficient RL training pipeline.

**Key Vision**: SGLang-native post-training framework that decouples rollout generation from training, enabling asynchronous, scalable RL workflows.

---

## 1. Overall Architecture & Main Components

### High-Level Design
Slime uses a **modular, decoupled architecture** with three main components:

```
┌─────────────────┐         ┌──────────────┐         ┌──────────────┐
│   Training      │◄────────┤  Data Buffer │────────►│   Rollout    │
│  (Megatron)     │         │  (Manager)   │         │  (SGLang)    │
└─────────────────┘         └──────────────┘         └──────────────┘
      • Forward/                • Prompt            • Generation
        Backward                 Initialization     • Reward Computation
      • Distributed            • Custom Data       • Inference
        Training               • Rollout Methods    • Multi-Engine
      • Checkpoint                                   Support
        Management
```

### Core Components

1. **Training Module (Megatron-based)**
   - Located in: `/slime/backends/megatron_utils/` and `/slime/backends/fsdp_utils/`
   - Supports both Megatron and FSDP backends
   - Handles model initialization, optimization, and gradient computation
   - Integrated with Ray for distributed actor management

2. **Rollout Module (SGLang-based)**
   - Located in: `/slime/backends/sglang_utils/`
   - Manages multiple inference engines
   - Handles batch generation and reward computation
   - Router-based request distribution

3. **Data Buffer & Management**
   - Located in: `/slime/ray/rollout_data_source.py`
   - Bridges data between rollout and training
   - Supports both global datasets and dynamic prompts
   - Handles sample grouping and shuffling

4. **Ray-based Orchestration**
   - Located in: `/slime/ray/`
   - Manages distributed actors for training and rollout
   - Placement group management for GPU allocation
   - Asynchronous data flow coordination

---

## 2. RL Algorithms Implemented

### Core RL Algorithms

#### A. **PPO (Proximal Policy Optimization)**
- **Implementation**: `/slime/backends/megatron_utils/loss.py`, `/slime/utils/ppo_utils.py`
- **Key Features**:
  - Policy gradient loss with clipping (`eps_clip`, `eps_clip_high`)
  - Dual-clip PPO support (asymmetric clipping)
  - Per-token loss computation
  - KL divergence regularization with configurable coefficient
  - Value function critic with value clipping
  - Entropy regularization

- **PPO Loss Computation**:
  ```python
  pg_loss, clipfrac = compute_policy_loss(
      ppo_kl, advantages, eps_clip, eps_clip_high
  )
  loss = pg_loss - entropy_coef * entropy + kl_loss_coef * kl_loss
  ```

#### B. **GRPO (Group Relative Policy Optimization)**
- **Implementation**: `/slime/backends/megatron_utils/loss.py` (line 239-243)
- **Key Features**:
  - Directly uses rewards as advantages (without value estimation)
  - Reduces variance by computing per-group rewards
  - Suitable for tasks with clear pass/fail rewards
  - No value function needed
  
- **Advantage Estimation**:
  ```python
  advantages = returns = get_grpo_returns(rewards, kl)
  ```

#### C. **GSPO (Group Sample Policy Optimization)**
- Similar to GRPO but includes:
  - Per-sample KL divergence computation
  - All-gather across context parallelism ranks for full sequence KL
  - Better handling of distributed computation

#### D. **REINFORCE++ and REINFORCE++ Baseline**
- **Implementation**: `/slime/utils/ppo_utils.py` (lines 134-243)
- Discounted return computation with configurable gamma
- Baseline-based advantage estimation
- Supports masked token-level returns

### KL Divergence Approximations

Located in `/slime/utils/ppo_utils.py`, supports multiple KL approximation types:

1. **k1**: Direct log ratio
2. **k2**: Squared log ratio / 2
3. **k3**: Reverse KL (exp(log_ratio) - 1 - log_ratio)
4. **low_var_kl**: Clamped reverse KL for lower variance

```python
def compute_approx_kl(log_probs, log_probs_base, kl_loss_type):
    log_ratio = log_probs - log_probs_base
    # Different KL approximations based on kl_loss_type
```

### Advantage Estimators

Controlled by `args.advantage_estimator`:

- **PPO**: Traditional advantage with GAE (Generalized Advantage Estimation)
  - Uses `gamma` (discount factor) and `lambd` (GAE lambda)
  - Value-based advantage computation
  
- **GRPO**: Simple group-relative rewards
  
- **GSPO**: Per-sample KL with all-gather
  
- **REINFORCE++**: Discounted token-level returns
  
- **REINFORCE++ Baseline**: Baseline-corrected advantages

---

## 3. Dataset Handling & Data Pipeline

### Data Loading & Management

**Dataset Class** (`/slime/utils/data.py`):
- Supports JSONL and Parquet formats
- Lazy loading with row slicing support (e.g., `path@[0:1000]`)
- Multi-modal support (text + images/videos)
- Tokenization with configurable max length
- Chat template application for instruction tuning

```python
class Dataset:
    def __init__(self, path, tokenizer, max_length, 
                 prompt_key="text", multimodal_keys=None, 
                 label_key=None, apply_chat_template=False):
        # Loads and preprocesses data
```

### Sample Types

**Sample Dataclass** (`/slime/utils/types.py`):
```python
@dataclass
class Sample:
    group_index: Optional[int]        # Prompt group ID
    index: Optional[int]              # Global sample index
    prompt: Union[str, list[dict]]    # Input prompt
    tokens: list[int]                 # Tokenized prompt+response
    response: str                     # Generated response
    response_length: int              # Response token count
    reward: Optional[float|dict]      # Scalar or multi-component reward
    loss_mask: Optional[list[int]]    # Token-level loss masking
    weight_versions: list[str]        # Model weight version tracking
    rollout_log_probs: Optional[list] # Rollout-time log probabilities
    status: Status                    # PENDING, COMPLETED, TRUNCATED, ABORTED
    metadata: dict                    # Custom metadata
    train_metadata: dict              # Training-specific info (loss type, etc.)
    spec_info: SpecInfo              # Speculative decoding stats
```

### Rollout Data Flow

**RolloutDataSource** (`/slime/ray/rollout_data_source.py`):

1. **Initialization**:
   - Loads global dataset if `args.rollout_global_dataset` is True
   - Creates tokenizer for prompt tokenization
   - Initializes epoch and sample tracking

2. **Sample Generation**:
   - `get_samples(num_samples)`: Yields prompt samples with grouping
   - Creates `n_samples_per_prompt` copies per prompt
   - Handles epoch rotation and shuffling

3. **Data Batching**:
   - Groups samples by prompt for importance-weighted rollout
   - Tracks group/sample indices for reconstruction

### Data Packing (FSDP Backend)

**Sequence Packing** (`/slime/backends/fsdp_utils/data_packing.py`):
- Reduces padding overhead for variable-length sequences
- Uses balanced partitioning to distribute work across GPUs
- Maintains cumulative sequence lengths (`cu_seqlens`) for efficient attention computation

```python
def pack_sequences(tokens, loss_masks, rewards, raw_rewards, 
                   response_lengths, advantages, returns,
                   max_tokens_per_gpu=None, num_packs=None):
    # Packs sequences to minimize padding while balancing compute
```

---

## 4. Training Loop Implementation

### Main Training Flow

**Entry Point**: `/Users/chiraagbalu/research/slime/train.py`

```python
def train(args):
    # 1. Allocate GPUs and setup placement groups
    pgs = create_placement_groups(args)
    
    # 2. Initialize rollout manager with SGLang engines
    rollout_manager = create_rollout_manager(args, pgs["rollout"])
    
    # 3. Create training actor and critic models
    actor_model, critic_model = create_training_models(args, pgs)
    
    # 4. Weight synchronization
    actor_model.update_weights()
    
    # 5. Main training loop
    for rollout_id in range(args.num_rollout):
        # Generate rollout data
        rollout_data = ray.get(rollout_manager.generate.remote(rollout_id))
        
        # Train critic (if using critic)
        if args.use_critic:
            critic_model.async_train(rollout_id, rollout_data)
        
        # Train actor policy
        actor_model.async_train(rollout_id, rollout_data)
        
        # Periodically save checkpoints
        if (rollout_id + 1) % args.save_interval == 0:
            actor_model.save_model(rollout_id)
        
        # Synchronize weights to rollout engines
        actor_model.update_weights()
        
        # Evaluation
        if (rollout_id + 1) % args.eval_interval == 0:
            ray.get(rollout_manager.eval.remote(rollout_id))
```

### Rollout Manager (`/slime/ray/rollout.py`)

Coordinates generation and reward computation:

```python
@ray.remote
class RolloutManager:
    def generate(self, rollout_id):
        # 1. Get prompts from data source
        # 2. Run generation via SGLang engines
        # 3. Compute rewards
        # 4. Convert to training format
        # 5. Handle dynamic filtering (if enabled)
        
    def eval(self, rollout_id):
        # Run evaluation on eval dataset
        # Compute pass@k, accuracy metrics
        
    def save(self, rollout_id):
        # Persist rollout data to disk
```

### Training Actor (`/slime/backends/megatron_utils/actor.py`)

**MegatronTrainRayActor** encapsulates:

1. **Model Initialization**:
   - Loads HF checkpoint or Megatron checkpoint
   - Initializes tokenizer and config
   - Wraps with DDP (Distributed Data Parallel)

2. **Forward Pass**:
   - Computes logits for current policy
   - Extracts log probabilities for response tokens
   - Optionally computes value predictions (critic)

3. **Loss Computation**:
   - Dispatches to appropriate loss function (policy/value/SFT)
   - Computes PPO/GRPO/REINFORCE++ losses
   - Handles KL regularization

4. **Backward Pass**:
   - Gradient accumulation with micro-batching
   - All-reduce across data parallel group
   - Optimizer step

### Training State Management

**Checkpoint System** (`/slime/backends/megatron_utils/checkpoint.py`):
- Saves/loads model state, optimizer state, and training step
- Supports resume from checkpoints
- Integrates with Megatron's checkpoint utilities

**Weight Update** (`/slime/backends/megatron_utils/update_weight_utils.py`):
- Broadcasts updated weights from training rank-0 to rollout engines
- Handles distributed state dict conversion
- Supports offload modes for memory efficiency

---

## 5. Model Support & Distributed Training Features

### Supported Models

**Native Support via Megatron**:
- Llama 3 series
- Qwen 3 series (Qwen3, Qwen3Next, Qwen3MoE)
- Qwen 2.5 series
- GLM-4 series (GLM-4.5, GLM-4.6)
- DeepSeek V3 series (V3, V3.1, DeepSeek R1)

**Model Conversion Tools** (`/slime/tools/`):
- `convert_hf_to_torch_dist.py`: HuggingFace → Megatron distributed
- `convert_torch_dist_to_hf.py`: Megatron → HuggingFace format
- `convert_to_hf.py`: Various format conversions
- FP8 quantization support

**Megatron-to-HF Converters** (`/slime/backends/megatron_utils/megatron_to_hf/`):
- Qwen2, Qwen3, Qwen3MoE
- GLM-4, GLM-4MoE
- Llama
- DeepSeekV3
- MIMO

### Distributed Training Backends

#### **Option 1: Megatron Backend** (`/slime/backends/megatron_utils/`)
- **Parallelism Strategies**:
  - Tensor Parallelism (TP): Model sharding across devices
  - Pipeline Parallelism (PP): Vertical model partitioning
  - Data Parallelism (DP): Batch splitting
  - Context Parallelism (CP): Sequence dimension partitioning
  - Sequence Parallelism (SP)

- **Key Capabilities**:
  - Mixed precision training (FP32, FP16, BF16)
  - Gradient accumulation with micro-batching
  - Flash attention integration
  - Memory-efficient attention patterns

#### **Option 2: FSDP Backend** (`/slime/backends/fsdp_utils/`)
- **Features**:
  - Fully Sharded Data Parallel (PyTorch native)
  - Automatic model sharding
  - CPU offload support
  - Simpler setup for smaller models
  - Data packing for efficiency

**Selection**:
```bash
# Megatron backend (default, for large models)
--backend megatron

# FSDP backend (for smaller models or HF-only setups)
--backend fsdp
```

### Distributed Training Arguments

Key configuration options in `/slime/utils/arguments.py`:

```python
# GPU Allocation
--actor-num-nodes 2              # Training nodes
--actor-num-gpus-per-node 8      # GPUs per training node
--critic-num-nodes 1             # Critic nodes (optional)
--rollout-num-gpus 8             # Inference GPUs

# Parallelism
--tensor-model-parallel-size 2   # Tensor parallelism
--pipeline-model-parallel-size 1 # Pipeline parallelism

# Memory Optimization
--offload                        # Offload both train and rollout
--offload-train                  # Offload only training
--offload-rollout                # Offload only inference
--colocate                       # Run training and inference on same GPUs

# Training Configuration
--global-batch-size 256
--micro-batch-size 8
--gradient-accumulation-steps 4
--num-rollout 1000              # Training iterations
```

### Asynchronous & Overlapping Operations

**Offloading Modes**:
- **TMS (Tensor Memory Saver)**: Smart parameter loading/unloading
- **Move**: CPU-GPU movement-based offloading

**Training Loop Optimization**:
```python
# Overlap rollout generation with training
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
if args.offload_rollout:
    ray.get(rollout_manager.offload.remote())  # Unload rollout engines

# Train while next rollout is generating (with async operations)
critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
actor_train_handle = actor_model.async_train(rollout_id, rollout_data_ref)
ray.get([critic_train_handle, actor_train_handle])  # Wait for completion
```

---

## 6. Evaluation & Metrics Capabilities

### Evaluation Framework

**Eval Dataset Management** (`/slime/utils/eval_config.py`):
```python
class EvalDatasetConfig(BaseModel):
    name: str                              # Dataset name
    path: str                              # Data file path
    rm_type: Optional[str]                 # Reward model type
    n_samples_per_eval_prompt: Optional[int]  # Sampling for eval
    temperature: Optional[float]
    max_response_len: Optional[int]
    metadata_overrides: Dict[str, Any]    # Custom overrides
```

Multiple eval datasets can be configured:
```python
args.eval_dataset_configs = [
    EvalDatasetConfig(name="math", path="math_val.jsonl", rm_type="math"),
    EvalDatasetConfig(name="gsm8k", path="gsm8k_val.jsonl"),
]
```

### Reward Models Hub (`/slime/rollout/rm_hub/`)

Pre-built reward models for common tasks:

1. **Math Utils** (`math_utils.py`):
   - Answer normalization (LaTeX handling, fraction formatting)
   - Symbolic expression comparison
   - Handles multiple answer formats

2. **Math DAPO Utils** (`math_dapo_utils.py`):
   - Multi-step solution verification
   - Process reward modeling
   - Trajectory-level rewards

3. **GPQA** (`gpqa.py`):
   - Graduate-level QA evaluation
   - Exact match and semantic similarity

4. **IFBench** (`ifbench.py`):
   - Instruction-following benchmark
   - Multi-aspect evaluation

5. **DeepScaler** (`deepscaler.py`):
   - Composite reward computation
   - Scaling-aware metrics

6. **F1 Score** (`f1.py`):
   - Precision/recall-based evaluation

### Metric Computation

**Metric Utils** (`/slime/utils/metric_utils.py`):

```python
def compute_pass_rate(flat_rewards, group_size):
    """Compute pass@k for code/math generation"""
    # Returns pass@1, pass@2, pass@4, ... for n-sample generations
    
def compute_statistics(values):
    """Compute mean, median, std of reward distribution"""
```

**Training Metrics** (`/slime/utils/train_metric_utils.py`):

- **Performance Metrics**:
  - `perf/log_probs_time`: Forward pass time
  - `perf/log_probs_tflops`: Throughput in TFLOPs
  - `perf/actor_train_tflops`: Training throughput
  - `perf/step_time`: Total iteration time
  - `perf/wait_time_ratio`: Sync overhead

- **Loss Metrics**:
  - `loss`: Total loss
  - `pg_loss`: Policy gradient loss
  - `entropy_loss`: Entropy regularization
  - `value_loss`: Critic loss
  - `pg_clipfrac`: Gradient clipping fraction
  - `ppo_kl`: KL divergence

- **Training Dynamics**:
  - `train_rollout_logprob_abs_diff`: Distribution shift
  - `tis`/`ois`: Importance sampling weights
  - `kl_loss`: KL regularization

### Multi-Task Evaluation

Example from examples/eval_multi_task/:
```python
# Multiple eval datasets with different reward models
eval_configs = [
    ("math_eval.jsonl", "math_utils.grade_answer"),
    ("code_eval.jsonl", "custom_code_rewards"),
    ("gsm8k.jsonl", None),  # External scorer
]

# Each eval dataset independently scored and logged
```

---

## 7. Unique Features & Optimizations

### A. Custom Rollout Functions

**Flexibility in Data Generation** (`args.rollout_function_path`):

Users can implement arbitrary rollout logic:

```python
# Example: Search-R1 with web search
def generate(prompts, model, reward_model, **kwargs):
    results = []
    for prompt in prompts:
        # Multi-turn generation with tool calling
        for turn in range(max_turns):
            response = model.generate(prompt)
            search_results = search_api(response)
            prompt += search_results
            # ... continue turns
        reward = reward_model(final_response)
        results.append({
            "prompt": prompt,
            "response": response,
            "reward": reward
        })
    return results
```

Examples in `/slime/examples/`:
- `search-r1/`: Multi-turn search with tool-calling
- `multi_agent/`: Multi-agent collaboration
- `retool/`: Tool-use and planning
- `true_on_policy/`: On-policy generation without replay

### B. Dynamic Sample Filtering

**Filter Hub** (`/slime/rollout/filter_hub/`):

Filter low-quality samples during rollout:

```python
class DynamicFilterOutput:
    keep: bool                 # Whether to keep sample
    reason: Optional[str]      # Why filtered

# Example: Remove zero-std rewards
def check_reward_nonzero_std(args, samples, **kwargs):
    rewards = [s.get_reward_value(args) for s in samples]
    keep = torch.tensor(rewards).std() > 0.0
    return DynamicFilterOutput(keep=keep)
```

### C. Router Middleware System

**Slime Router** (`/slime/router/router.py`):
- Custom request routing between inference engines
- Middleware-based extensibility
- Load balancing across multiple engines
- Request queueing and priority handling

**Example Middleware** (`/slime/router/middleware_hub/radix_tree_middleware.py`):
- Radix tree-based prompt caching
- Reduces redundant computation
- Accelerates similar request sequences

### D. Speculative Decoding Support

**SGLang Integration**:
- Speculative decoding metrics in Sample:
  ```python
  class SpecInfo:
      spec_accept_token_num: int      # Tokens accepted
      spec_draft_token_num: int       # Tokens drafted
      spec_verify_ct: int             # Verification rounds
      spec_accept_rate: float         # Acceptance rate
  ```
- Automatic tracking during rollout

### E. Context & Sequence Parallelism

**Advanced Parallelism** (`/slime/backends/megatron_utils/cp_utils.py`):

- **Context Parallelism**: Splits long sequences across GPUs
  - Reduces per-GPU memory
  - Maintains correctness through all-gather operations
  
- **Sequence Parallel Attention**: Further reduces memory with distributed attention

- **Loss Computation with CP**: Special handling for distributed loss calculation

```python
def get_logits_and_tokens_offset_with_cp(total_length, response_length):
    # Computes chunk boundaries for context-parallel ranks
    # Returns offsets for slicing response tokens correctly
```

### F. True On-Policy Training

**Example**: `/slime/examples/true_on_policy/`

Trains only on freshly generated rollouts (no replay buffer):
- No off-policy corrections needed
- Simplest RL algorithm
- Requires efficient rollout generation

### G. Fault Tolerance

**Health Monitoring** (`/slime/utils/health_monitor.py`):
- Monitors inference engine health
- Automatic engine restart on failure
- Seamless recovery without training interruption

```python
if args.use_fault_tolerance:
    self._health_monitor = RolloutHealthMonitor(self, args)
```

### H. Memory-Efficient Training

**Data Packing** (FSDP):
- Reduced padding overhead
- Balanced partitioning for compute efficiency

**Torch Memory Saver**:
- Smart activation checkpointing
- Reduces peak memory usage
- Works with both Megatron and FSDP

**CPU Offload**:
- Parameter offloading during forward/backward
- Gradient checkpointing
- Allows training larger models with fewer GPUs

### I. Weight Version Tracking

Each sample records `weight_versions`:
- Tracks which model weights generated the sample
- Enables off-policy correction
- Useful for distributed training with async updates

### J. Comprehensive Logging & Monitoring

**Logging Systems**:
- **Weights & Biases (W&B)**: Full experiment tracking
- **TensorBoard**: Real-time visualization
- **Detailed Performance Profiling**: TFLOPs, latency breakdown
- **Health Monitoring**: Resource usage, bottlenecks

---

## 8. Configuration & Arguments

### Key Arguments Categories

#### Cluster Configuration
```bash
--actor-num-nodes 2
--actor-num-gpus-per-node 8
--rollout-num-gpus 8
--num-gpus-per-node 8
--colocate                    # Share GPUs between train & inference
```

#### Training Hyperparameters
```bash
--num-rollout 1000            # Training iterations
--rollout-batch-size 256      # Prompts per rollout
--n-samples-per-prompt 1      # Generations per prompt
--global-batch-size 256       # Training batch size
--micro-batch-size 8
--lr 1e-6
--entropy-coef 0.01
--gamma 0.99                  # Discount factor
--lambd 0.95                  # GAE lambda
```

#### RL Algorithm
```bash
--advantage-estimator ppo     # ppo | grpo | gspo | reinforce_plus_plus
--kl-coef 0.05                # KL penalty coefficient
--kl-loss-type k3             # k1 | k2 | k3 | low_var_kl
--eps-clip 0.2                # PPO clip range
--value-clip 0.2              # Value function clip range
--normalize-advantages        # Whitened advantage normalization
```

#### Rollout Configuration
```bash
--rollout-function-path pkg.module.function
--eval-function-path pkg.module.eval_function
--prompt-data data/prompts.jsonl
--rollout-max-prompt-len 1024
--rollout-max-response-len 2048
--temperature 0.7
```

#### Data Processing
```bash
--rollout-global-dataset      # Load global dataset vs. streaming
--apply-chat-template         # Use tokenizer chat template
--label-key label             # Ground truth key
--metadata-key metadata
--tool-key tools              # For multi-tool tasks
```

---

## 9. Typical Workflow & Example

### Complete Training Example

```bash
python train.py \
    --hf-checkpoint meta-llama/Llama-3-8b \
    --actor-num-nodes 2 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --num-rollout 1000 \
    --rollout-batch-size 256 \
    --n-samples-per-prompt 1 \
    --advantage-estimator ppo \
    --kl-coef 0.05 \
    --prompt-data prompts.jsonl \
    --rollout-function-path slime.rollout.sft_rollout.generate \
    --custom-rm-path rewards.compute_reward \
    --output-dir ./checkpoints \
    --save-interval 50 \
    --eval-interval 100
```

### Training Data Format

**Prompts** (prompts.jsonl):
```json
{"text": "Q: What is 2+2?\nA:", "metadata": {"difficulty": "easy"}}
{"text": "Implement a binary search function in Python\n", "label": "<code>..."}
```

**Custom Reward Function** (rewards.py):
```python
def compute_reward(prompt, response, label=None, **kwargs):
    if label and response.strip() == label.strip():
        return 1.0
    elif math.isclose(eval(response), eval(label)):
        return 1.0
    else:
        return 0.0
```

---

## 10. Advanced Capabilities

### Fully Asynchronous Training

Example: `/slime/examples/fully_async/`
- Rollout generation and training completely decoupled
- Multiple rollouts in flight simultaneously
- Optimal hardware utilization

### True On-Policy RL

Example: `/slime/examples/true_on_policy/`
- Trains exclusively on fresh rollouts
- No replay buffer or off-policy corrections
- Simplest, most interpretable training

### Multi-Agent Training

Example: `/slime/examples/multi_agent/`
- Coordinate multiple agents
- Agent-to-agent communication
- Collaborative training dynamics

### Reproducibility & Debugging

Example: `/slime/examples/reproducibility/`
- Fixed seed management
- Deterministic sampling
- Debug mode for logging generation details

---

## 11. Comparison with Other Frameworks

| Feature | Slime | OpenRLHF | veRL |
|---------|-------|----------|------|
| **Training Backend** | Megatron/FSDP | Megatron | Megatron |
| **Inference Engine** | SGLang | vLLM | vLLM |
| **Architecture** | Decoupled rollout/train | Coupled | Coupled |
| **Asynchronous** | Full async support | Limited | Limited |
| **Router** | Custom middleware system | No | No |
| **Fault Tolerance** | Built-in | No | No |
| **Speculative Decoding** | Native SGLang support | Limited | Limited |
| **Custom Rollouts** | Flexible callback system | Limited | Limited |
| **Parallelism** | TP/PP/DP/CP/SP | TP/PP/DP | TP/PP/DP |

---

## 12. References & Documentation

- **Main Repository**: https://github.com/THUDM/slime
- **Vision Blog**: https://lmsys.org/blog/2025-07-09-slime/
- **Release Notes**: v0.1.0 release documentation
- **Quick Start**: `/docs/en/get_started/quick_start.md`
- **Usage Guide**: `/docs/en/get_started/usage.md`
- **Developer Guide**: `/docs/en/developer_guide/`

---

## Summary

Slime represents a **production-grade RL training framework** optimized for LLM post-training at scale. Its key strengths are:

1. **High-Performance Architecture**: Megatron + SGLang integration with multiple parallelism strategies
2. **Flexibility**: Custom rollout functions, reward models, and middlewares
3. **Scalability**: Decoupled, asynchronous design enabling full GPU utilization
4. **Robustness**: Built-in fault tolerance and comprehensive monitoring
5. **Modern Algorithms**: PPO, GRPO, GSPO, REINFORCE++ with advanced KL regularization
6. **Production Ready**: Powers GLM-4.5/4.6 and supports major LLM architectures

It's specifically designed for organizations seeking to scale RL training to production, with careful attention to system efficiency, fault tolerance, and algorithmic soundness.
