# SLIME Hyperparameters Comprehensive Reference

This document provides a complete inventory of all hyperparameters used in SLIME for training, extracted from:
- `references/slime/slime/utils/arguments.py` (main SLIME arguments)
- `references/slime/slime/backends/fsdp_utils/arguments.py` (FSDP-specific)
- `references/slime/slime/backends/sglang_utils/arguments.py` (SGLang inference)
- `dev/integration_training/base_config.py` (integration training defaults)

---

## 1. OPTIMIZER PARAMETERS (FSDP Backend)

**Location:** `slime/backends/fsdp_utils/arguments.py`

### Optimizer Selection
- `optimizer`: str = "adam"
  - Options: "adam" (GPU AdamW), "deepspeed_cpu_adam" (CPU-offloaded)

### Learning Rate
- `lr`: float = 2e-5 (0.00002)
  - Base learning rate for training
  - Used for Megatron backend with default: `--lr` = 1e-6 (0.000001)

### Learning Rate Schedule
- `lr_decay_style`: str = "constant"
  - Determines how learning rate decays during training
  - Options likely: "constant", "linear", "cosine", "polynomial"

- `warmup_ratio`: float = 0.03
  - Fraction of total steps to linearly increase learning rate from 0 to `lr`
  - At default warmup_ratio=0.03, for 10,000 total steps: 300 steps of warmup

### Optimizer Hyperparameters (Adam)
- `adam_beta1`: float = 0.9
  - Exponential decay rate for 1st moment estimates (momentum)

- `adam_beta2`: float = 0.95
  - Exponential decay rate for 2nd moment estimates (RMSprop)
  - Default is 0.999 in standard PyTorch, but SLIME uses 0.95

- `adam_eps`: float = 1e-8
  - Small constant for numerical stability in Adam denominator

### Regularization
- `weight_decay`: float = 0.0
  - L2 regularization coefficient
  - Default: 0.0 (no regularization)

---

## 2. GRADIENT CLIPPING & ACCUMULATION

**Location:** `slime/utils/arguments.py` (reset_arg function)

### Gradient Clipping
- `--clip-grad`: float = 1.0
  - Max norm for gradient clipping
  - Applied globally across all parameters
  - Prevents exploding gradients during training

### Gradient Accumulation
- `--micro-batch-size`: int = 1 (default, can be overridden)
  - Number of samples processed per GPU before gradient update
  - Global batch size = micro_batch_size * num_gpus * accumulation_steps

### Dynamic Batch Sizing
- `--use-dynamic-batch-size`: bool = False
  - If True: adjusts micro batch size to fit max_tokens_per_gpu
  - Useful for variable-length sequences

- `--max-tokens-per-gpu`: int = None
  - Max tokens per GPU when using dynamic batching
  - For context parallel (CP): max_tokens_per_gpu ≈ max_response_len / cp_size

---

## 3. TRAINING PARAMETERS

**Location:** `slime/utils/arguments.py`

### Batch Sizes
- `--rollout-batch-size`: int (REQUIRED)
  - Number of prompts in each rollout step
  - Total data returned = rollout_batch_size * n_samples_per_prompt

- `--global-batch-size`: int = None (auto-calculated)
  - Total batch size across all GPUs
  - Formula: rollout_batch_size * n_samples_per_prompt / num_steps_per_rollout

- `--micro-batch-size`: int = 1
  - Per-GPU batch size
  - Overridden by --use-dynamic-batch-size if enabled

- `--n-samples-per-prompt`: int = 1
  - Number of responses for each prompt in generation
  - Total samples = rollout_batch_size * n_samples_per_prompt

### Training Schedule
- `--num-steps-per-rollout`: int = None
  - Number of training steps per rollout iteration
  - If set: global_batch_size = rollout_batch_size * n_samples_per_prompt / num_steps_per_rollout

- `--seed`: int = 1234
  - Random seed for reproducibility
  - Affects model initialization, data shuffling, sampling

### Checkpointing
- `--save-interval`: int = None
  - Steps between checkpoint saves

- `--load`: str = None
  - Path to checkpoint to load for resuming training

- `--save`: str = None
  - Directory to save checkpoints

---

## 4. PRECISION & COMPUTATION

**Location:** `slime/backends/fsdp_utils/arguments.py` & megatron_utils/arguments.py

### Mixed Precision
- `bf16`: bool = True (Megatron default)
  - Use bfloat16 precision for training
  - Set in `set_default_megatron_args()`

### Gradient Checkpointing
- `gradient_checkpointing`: bool = False
  - If True: trade compute for memory by recomputing activations

### Attention Implementation
- `attn_implementation`: str = "flash_attention_2"
  - FSDP backend: "flash_attention_2" for efficient attention

---

## 5. RL-SPECIFIC HYPERPARAMETERS

**Location:** `slime/utils/arguments.py` (add_algo_arguments)

### PPO/RL Training
- `--eps-clip`: float = 0.2
  - PPO clip range (trust region constraint)
  - ratio = log(π_new / π_old)
  - Loss clipped to [1 - eps_clip, 1 + eps_clip]

- `--eps-clip-high`: float = None (defaults to eps_clip)
  - Upper bound for PPO clipping (asymmetric clipping)

- `--eps-clip-c`: float = None
  - Lower bound from Dual-clip PPO (https://arxiv.org/pdf/1912.09729)

- `--value-clip`: float = 0.2
  - Clip range for value function loss

### KL Divergence Penalty
- `--kl-coef`: float = 0.00
  - KL penalty coefficient for reward shaping
  - Applied to reward signal BEFORE advantage calculation
  - reward_shaped = reward - kl_coef * kl_div

- `--use-kl-loss`: bool = False
  - Whether to add KL loss term from GRPO
  - Different from kl_coef (applied to loss, not reward)

- `--kl-loss-coef`: float = 0.0
  - KL penalty coefficient added to final PPO loss
  - Cannot be set simultaneously with kl_coef != 0

- `--kl-loss-type`: str = "k1"
  - KL divergence calculation method
  - Options: "k1", "k2", "k3", "low_var_kl"

### Advantage Estimation
- `--advantage-estimator`: str = "grpo"
  - Options: "grpo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "ppo"
  - GRPO: REINFORCE + baseline + importance weighting
  - PPO: Full PPO algorithm with critic

- `--disable-compute-advantages-and-returns`: bool = False
  - If set: skip advantage/return computation (for SFT or custom loss)

- `--normalize-advantages`: bool = False
  - Normalize advantages to zero mean, unit variance
  - REQUIRED for reinforce_plus_plus variants

### GAE (Generalized Advantage Estimation)
- `--gamma`: float = 1.0
  - Discount factor for future rewards
  - 1.0 = no discounting, 0.99 = slight discounting

- `--lambd`: float = 1.0
  - GAE lambda for exponential weighting of TD-λ errors
  - 1.0 = simple Monte Carlo return (no TD bootstrapping)
  - 0.0 = full TD (only 1-step TD error)

### Entropy & Regularization
- `--entropy-coef`: float = 0.0
  - Entropy bonus coefficient to encourage exploration
  - Loss += entropy_coef * entropy(π)

- `--disable-grpo-std-normalization`: bool = False
  - From Dr.GRPO (https://arxiv.org/pdf/2503.20783)
  - Normalize rewards by standard deviation

- `--disable-rewards-normalization`: bool = False
  - Disable reward normalization

### Loss Function
- `--loss-type`: str = "policy_loss"
  - Options: "policy_loss", "sft_loss", "custom_loss"
  - "policy_loss": PPO-style policy gradient
  - "sft_loss": Supervised fine-tuning with log-likelihood
  - "custom_loss": User-defined loss function

- `--custom-loss-function-path`: str = None
  - Path to custom loss function (required if loss_type="custom_loss")

### Reference Model & Off-Policy Correction
- `--ref-load`: str = None
  - Checkpoint path for reference model
  - Used for KL divergence calculation
  - Falls back to --load if not set

- `--ref-ckpt-step`: int = None
  - Specific checkpoint step for reference model

- `--ref-update-interval`: int = None
  - Steps between updating reference model from actor
  - If None: reference model stays fixed

- `--use-rollout-logprobs`: bool = False
  - Use logprobs from rollout (not actor) for importance sampling
  - If False: use actor logprobs

- `--use-tis`: bool = False
  - Trajectory Importance Sampling for off-policy correction
  - See: https://fengyao.notion.site/off-policy-rl

- `--tis-clip`: float = 2.0
  - Clipping threshold C for importance sampling ratios
  - variance = tis_clip, stability = tis_clip_low

- `--tis-clip-low`: float = 0
  - Lower bound clipping for importance sampling ratios

- `--custom-tis-function-path`: str = None
  - Path to custom TIS (Trajectory IS) function

### Critic Model (PPO only)
- `--num-critic-only-steps`: int = 0
  - Training steps where only critic is updated

- `--critic-load`: str = None
  - Checkpoint for critic model (defaults to --load)

- `--critic-save`: str = None
  - Directory to save critic checkpoints

- `--critic-lr`: float = None
  - Learning rate for critic (defaults to --lr)

- `--critic-lr-warmup-iters`: int = 0
  - Linear warmup iterations for critic learning rate

---

## 6. INFERENCE/ROLLOUT PARAMETERS

**Location:** `slime/utils/arguments.py`

### Generation Hyperparameters
- `--rollout-temperature`: float = 1.0
  - Sampling temperature (0.0 = greedy, higher = more diverse)

- `--rollout-top-p`: float = 1.0
  - Nucleus (top-p) sampling parameter
  - 1.0 = all tokens considered, <1.0 = cutoff at p probability

- `--rollout-top-k`: int = -1
  - Top-k sampling (-1 = disabled, >0 = sample from top k tokens)

- `--rollout-max-response-len`: int = 1024
  - Maximum number of tokens to generate per prompt
  - Also called max_tokens in SGLang

- `--rollout-max-context-len`: int = None
  - Maximum context size for inference engine
  - Should not exceed max_position_embeddings in HF config

- `--rollout-max-prompt-len`: int = None
  - Maximum input prompt length (filters long prompts at init)

- `--rollout-skip-special-tokens`: bool = False
  - Skip special tokens in generated response

### Generation Sampling
- `--n-samples-per-prompt`: int = 1
  - Number of different responses for each prompt

- `--rollout-shuffle`: bool = False
  - Shuffle prompts during rollout

- `--rollout-seed`: int = 42
  - Random seed for rollout (prompt shuffling, sampling)

- `--over-sampling-batch-size`: int = None (defaults to rollout_batch_size)
  - Granularity for sampling operations
  - Used when available samples fall below target

### Dynamic Sampling Filter
- `--dynamic-sampling-filter-path`: str = None
  - Path to filter function for dynamic sampling
  - Selects samples based on reward characteristics (e.g., non-zero std)
  - Example: `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`

### Partial Rollout
- `--partial-rollout`: bool = False
  - If True: recycle unfinished samples back to buffer
  - Useful for very long responses

---

## 7. DATA PARAMETERS

**Location:** `slime/utils/arguments.py`

### Dataset Configuration
- `--prompt-data`: str = None
  - Path to prompt data (JSONL format)
  - Each line must contain --input-key and --label-key

- `--input-key`: str = "input"
  - JSON key for input/prompt field

- `--label-key`: str = None
  - JSON key for label field

- `--metadata-key`: str = "metadata"
  - JSON key for metadata

- `--apply-chat-template`: bool = False
  - Apply tokenizer chat template to input
  - Input should be list of OpenAI-format messages

- `--apply-chat-template-kwargs`: dict = {}
  - Additional kwargs for chat template application

### Training Data Control
- `--rollout-global-dataset`: bool = True
  - Use global dataset for rollout
  - If False: user must manage data

- `--balance-data`: bool = False
  - Balance token counts across data parallel ranks
  - Uses karmarkar_karp algorithm for verl

- `--use-dynamic-batch-size`: bool = False
  - Adjust batch size based on max_tokens_per_gpu

- `--log-probs-max-tokens-per-gpu`: int = None
  - Max tokens for computing log probabilities
  - Should be larger than max_tokens_per_gpu for better performance

---

## 8. EVALUATION PARAMETERS

**Location:** `slime/utils/arguments.py`

### Eval Interval
- `--eval-interval`: int = None
  - Training steps between evaluations
  - Must be set if eval_datasets configured

### Eval Data
- `--eval-prompt-data`: list[str] = None
  - Eval dataset paths (alternating name/path: "aime /path/to/aime.jsonl")

- `--eval-config`: str = None
  - YAML/JSON config file for eval datasets (overrides eval-prompt-data)

### Eval Generation Parameters
- `--eval-temperature`: float = None (uses rollout_temperature)
- `--eval-top-p`: float = None (uses rollout_top_p)
- `--eval-top-k`: int = None (uses rollout_top_k)
- `--eval-max-response-len`: int = None (uses rollout_max_response_len)
- `--n-samples-per-eval-prompt`: int = 1
  - Number of responses per eval prompt

---

## 9. DISTRIBUTED TRAINING PARAMETERS

**Location:** `slime/utils/arguments.py`

### Cluster Configuration
- `--actor-num-nodes`: int = 1
  - Number of nodes for training actor

- `--actor-num-gpus-per-node`: int = 8
  - GPUs per node for training

- `--critic-num-nodes`: int = None (defaults to actor_num_nodes)
  - Number of nodes for critic (PPO only)

- `--critic-num-gpus-per-node`: int = None (defaults to actor_num_gpus_per_node)

- `--rollout-num-gpus`: int = None
  - Total GPUs for inference
  - Ignored if --colocate set (computed from actor GPUs + critic GPUs)

- `--rollout-num-gpus-per-engine`: int = 1
  - GPUs per inference engine (tensor parallelism size for SGLang)

- `--num-gpus-per-node`: int = 8
  - GPUs per node for rollout

### Colocate & Offload
- `--colocate`: bool = False
  - Colocate inference and training on same GPUs
  - Implies --offload-train and --offload-rollout

- `--offload`: bool = False
  - Shorthand for --offload-train + --offload-rollout

- `--offload-train`: bool = None (derived from --colocate)
  - Offload training actor to CPU during inference

- `--offload-train-mode`: str = "tms"
  - Offload approach: "tms" (tensor memory swap) or "move"

- `--offload-rollout`: bool = None (derived from --colocate)
  - Offload inference to CPU during training

### Backend & Communication
- `--train-backend`: str = "megatron"
  - Options: "megatron", "fsdp"
  - Training framework choice

- `--distributed-backend`: str = "nccl"
  - PyTorch distributed backend (NCCL for GPU)

- `--distributed-timeout-minutes`: int = 10
  - Timeout for distributed communication

---

## 10. MEGATRON-SPECIFIC DEFAULTS

**Location:** `slime/backends/megatron_utils/arguments.py`

Set in `set_default_megatron_args()`:

- `use_distributed_optimizer`: bool = True
  - Use zero optimizer for memory efficiency

- `bf16`: bool = True
  - Use bfloat16 precision

- `seq_length`: int = 4096
  - Maximum sequence length

- `max_position_embeddings`: int = seq_length
  - Token position embedding range

- `rope_type`: str = "yarn" if multi_latent_attention else "rope"
  - RoPE variant for rotary position embeddings

- `variable_seq_lengths`: bool = True (always set for SLIME)
  - Support variable sequence lengths

---

## 11. WANDB LOGGING PARAMETERS

**Location:** `slime/utils/arguments.py`

- `--use-wandb`: bool = False
  - Enable Weights & Biases logging

- `--wandb-project`: str = None
  - W&B project name

- `--wandb-mode`: str = None
  - Options: "online", "offline", "disabled"

- `--wandb-group`: str = None
  - W&B group for runs

- `--wandb-run-id`: str = None
  - Specific run ID for resuming runs

- `--wandb-random-suffix`: bool = True
  - Add random suffix to run name for uniqueness

- `--wandb-always-use-train-step`: bool = False
  - Use train step instead of rollout step for metrics

- `--log-passrate`: bool = False
  - Log pass@n of responses

- `--log-reward-category`: str = None
  - Log statistics by reward category

---

## 12. INTEGRATION TRAINING DEFAULTS

**Location:** `dev/integration_training/base_config.py`

### SFT Configuration
```
SFTConfig:
  num_epochs: 1
  batch_size: 4
  target_examples_per_step: 32
  unembedding_lr: 0.004
  embedding_lr: 0.2
  matrix_lr: 0.02
  weight_decay: 0.0
  init_lr_frac: 0.02
  eval_every: 100
  eval_steps: 100
  checkpoint_every: 500
  log_every: 10
```

### RL Configuration
```
RLConfig:
  num_epochs: 1
  examples_per_step: 16
  num_samples: 16
  batch_size: 8
  max_new_tokens: 256
  temperature: 1.0
  top_k: 50
  unembedding_lr: 0.004
  embedding_lr: 0.2
  matrix_lr: 0.02
  weight_decay: 0.0
  init_lr_frac: 0.05
  eval_every: 60
  eval_examples: 400
  save_every: 60
  baseline: 0.0
```

---

## Summary Table: Most Critical Hyperparameters

| Category | Parameter | Default | Range/Type | Notes |
|----------|-----------|---------|-----------|-------|
| **Learning** | lr | 1e-6 (megatron) / 2e-5 (fsdp) | float | Base learning rate |
| | warmup_ratio | 0.03 | 0.0-0.5 | Fraction of training |
| | adam_beta1 | 0.9 | 0.8-0.99 | Momentum decay |
| | adam_beta2 | 0.95 | 0.9-0.999 | Variance decay |
| | adam_eps | 1e-8 | 1e-6 to 1e-10 | Numerical stability |
| **Regularization** | weight_decay | 0.0 | 0.0-0.1 | L2 penalty |
| | clip_grad | 1.0 | 0.1-10.0 | Gradient norm clip |
| **Training** | rollout_batch_size | REQUIRED | int | Prompts per step |
| | global_batch_size | auto | int | Batch size all GPUs |
| | micro_batch_size | 1 | int | Per-GPU batch size |
| **RL** | eps_clip | 0.2 | 0.1-0.5 | PPO clip range |
| | kl_coef | 0.0 | 0.0-1.0 | KL penalty weight |
| | gamma | 1.0 | 0.9-1.0 | Discount factor |
| | lambd | 1.0 | 0.0-1.0 | GAE lambda |
| **Generation** | rollout_max_response_len | 1024 | int | Max tokens to generate |
| | rollout_temperature | 1.0 | 0.0-2.0 | Sampling temperature |
| | rollout_top_p | 1.0 | 0.0-1.0 | Nucleus sampling |

