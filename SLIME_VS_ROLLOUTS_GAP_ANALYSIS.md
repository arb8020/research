# Slime vs Rollouts: Gap Analysis

## Executive Summary

**If you wanted to do an RL run à la Slime, here's what you wouldn't be able to do with rollouts:**

### 🔴 Critical Gaps (Blockers for Production RL)
1. **No PPO implementation** - Only GRPO works, no value function critic
2. **No multi-GPU distributed training** - FSDP/DeepSpeed not integrated
3. **No SGLang decoupled rollout architecture** - Rollouts are in-process, not separate engines
4. **No asynchronous training/rollout overlap** - Must generate then train sequentially
5. **No reward model training** - Only environment-based grading works
6. **No context parallelism** - Can't split long sequences across GPUs

### 🟡 Important Gaps (Limits Scale/Robustness)
7. **No health monitoring/fault tolerance** - No automatic engine restart
8. **No data packing** - Wastes compute on padding
9. **No TIS (off-policy correction)** - Can't reuse old rollouts safely
10. **No router middleware** - Missing prompt caching optimization
11. **No multi-component rewards** - Single scalar reward only
12. **No speculative decoding integration**

### 🟢 Minor Gaps (Nice-to-Haves)
13. **No W&B/TensorBoard detailed profiling** - Basic logging only
14. **No dynamic sample filtering** - Keep all rollouts regardless of quality
15. **No custom rollout functions** - Limited to standard generation

---

## Detailed Comparison

### 1. RL Algorithms

| Algorithm | Slime | Rollouts | Gap |
|-----------|-------|----------|-----|
| **GRPO** | ✅ Full | ✅ Basic | Rollouts has simple version, Slime has per-sample KL, advanced advantage estimation |
| **GSPO** | ✅ Full | ❌ Missing | GRPO with distributed per-sample KL computation |
| **PPO** | ✅ Full | ❌ Stub only | **CRITICAL**: No value function, no multi-epoch training, no trust region |
| **REINFORCE++** | ✅ Full | ❌ Missing | Token-level discounted returns |
| **KL Divergence** | 4 types | Basic | Slime has k1/k2/k3/low_var_kl approximations |
| **TIS (Off-policy)** | ✅ Yes | ❌ No | **IMPORTANT**: Can't reuse old rollouts with weight version tracking |

**Impact**: Rollouts can only run basic GRPO. For research requiring PPO or advanced variants, you're blocked.

---

### 2. Architecture & Scalability

| Feature | Slime | Rollouts | Gap |
|---------|-------|----------|-----|
| **Decoupled Rollout/Training** | ✅ SGLang engines | ❌ In-process | **CRITICAL**: Rollouts can't run separate inference servers, must load model twice |
| **Async Training** | ✅ Yes | ❌ No | **CRITICAL**: Can't overlap generation + training for full GPU utilization |
| **Ray Orchestration** | ✅ Yes | ❌ No | Slime uses Ray for distributed actor management |
| **Multi-GPU (FSDP)** | ✅ Full | ❌ Missing | **CRITICAL**: Can't train models >1 GPU memory |
| **Multi-GPU (DeepSpeed)** | ✅ Via Megatron | ❌ Missing | Same as above |
| **Tensor Parallel** | ✅ Yes | ❌ No | Can't split large models across GPUs |
| **Pipeline Parallel** | ✅ Yes | ❌ No | Can't split layers across nodes |
| **Context Parallel** | ✅ Yes | ❌ No | **IMPORTANT**: Can't handle very long sequences (>32k) |
| **Data Packing** | ✅ Yes | ❌ No | Wastes ~30-50% compute on padding |

**Impact**: Rollouts is single-GPU only. For models >24GB or multi-node training, you're completely blocked.

---

### 3. Rollout Generation

| Feature | Slime | Rollouts | Gap |
|---------|-------|----------|-----|
| **SGLang Integration** | ✅ Native | ⚠️ Via InferenceEngine | Rollouts can call SGLang but doesn't use decoupled architecture |
| **Custom Rollout Functions** | ✅ Yes | ❌ Limited | **IMPORTANT**: Can't implement search, tool-calling, multi-agent in rollouts |
| **Router Middleware** | ✅ Yes | ❌ No | Missing radix-tree prompt caching optimization |
| **Health Monitoring** | ✅ Auto restart | ❌ No | **IMPORTANT**: Engine crashes kill training |
| **Speculative Decoding** | ✅ Native | ❌ No | Missing 2-3x speedup for generation |
| **Dynamic Filtering** | ✅ Yes | ❌ No | Rollouts trains on all samples, even low-quality |
| **Weight Sync** | ✅ Automatic | ✅ Manual | Both support, Slime is more automated |

**Impact**: For complex rollout patterns (e.g., MCTS, multi-turn tool use), Slime is far more flexible.

---

### 4. Data Pipeline

| Feature | Slime | Rollouts | Gap |
|---------|-------|----------|-----|
| **Dataset Loading** | ✅ JSONL/Parquet | ✅ HuggingFace | Both good, different formats |
| **Sample Grouping** | ✅ n_samples_per_prompt | ⚠️ Manual | Slime automates GRPO sample pairing |
| **Multi-modal (Images)** | ✅ Yes | ❌ No | Can't handle vision models |
| **Data Packing** | ✅ FSDP | ❌ No | Rollouts wastes compute on padding |
| **Epoch Handling** | ✅ Automatic | ✅ DataBuffer | Both support deterministic cycling |
| **Off-policy Tracking** | ✅ Weight versions | ❌ No | Can't safely reuse old rollouts |

**Impact**: Rollouts is text-only and less efficient. For vision models or off-policy training, blocked.

---

### 5. Reward & Evaluation

| Feature | Slime | Rollouts | Gap |
|---------|-------|----------|-----|
| **Reward Model Hub** | ✅ Math/GPQA/IF | ❌ No | **CRITICAL**: Rollouts has no built-in reward models |
| **Environment Grading** | ❌ No | ✅ Yes | Rollouts advantage: calculator, search tools |
| **Multi-component Rewards** | ✅ Yes | ❌ No | Slime can combine multiple reward signals |
| **Process Rewards (DAPO)** | ✅ Yes | ❌ No | Can't reward intermediate steps |
| **Pass@k Metrics** | ✅ Automatic | ❌ Manual | Need to implement yourself |
| **Evaluation Loop** | ✅ Integrated | ⚠️ Separate | Rollouts has eval framework but not integrated in training |

**Impact**: For standard benchmarks (math, QA), Slime is ready. Rollouts requires custom reward implementation.

---

### 6. Training Loop & Optimization

| Feature | Slime | Rollouts | Gap |
|---------|-------|----------|-----|
| **SFT Training** | ✅ Full | ⚠️ 90% done | Rollouts needs orchestration glue |
| **RL Training** | ✅ Production | ⚠️ 90% done | Same as above |
| **LR Scheduler** | ✅ Cosine/Linear | ❌ Missing | **IMPORTANT**: Rollouts uses constant LR |
| **Gradient Clipping** | ✅ Yes | ⚠️ Manual | Need to add to backend |
| **Mixed Precision** | ✅ Automatic | ⚠️ Manual | Need to configure PyTorch AMP |
| **Checkpointing** | ✅ Distributed | ✅ Single GPU | Both work for their scope |
| **Resume Training** | ✅ Yes | ✅ Yes | Both support |
| **Fault Tolerance** | ✅ Auto retry | ❌ No | Training crashes on errors |

**Impact**: Rollouts can do basic training but lacks robustness for long runs.

---

### 7. Monitoring & Logging

| Feature | Slime | Rollouts | Gap |
|---------|-------|----------|-----|
| **W&B Integration** | ✅ Full | ✅ Basic | Slime has detailed profiling |
| **TensorBoard** | ✅ Yes | ❌ No | Rollouts only has JSONL + W&B |
| **Performance Metrics** | ✅ TFLOPs, MFU | ❌ No | **IMPORTANT**: Can't measure hardware efficiency |
| **Training Dynamics** | ✅ Clipfrac, KL, entropy | ⚠️ Basic | Rollouts logs loss/reward, missing RL-specific metrics |
| **Health Monitoring** | ✅ Automatic | ❌ No | No alerts for stuck/crashed engines |

**Impact**: Rollouts has basic logging but lacks production monitoring.

---

### 8. Model Support

| Feature | Slime | Rollouts | Gap |
|---------|-------|----------|-----|
| **Llama 3** | ✅ Native (Megatron) | ✅ PyTorch | Both support |
| **Qwen 2.5/3** | ✅ Native | ✅ Via HF | Rollouts less optimized |
| **GLM-4** | ✅ Native | ⚠️ Generic | Need custom integration |
| **DeepSeek V3** | ✅ Native | ⚠️ Generic | Same as above |
| **Distributed Models** | ✅ TP/PP/FSDP | ❌ No | **CRITICAL**: Can't run >1 GPU models |

**Impact**: Rollouts works for standard HF models, Slime optimized for specific architectures.

---

### 9. Memory & Performance Optimization

| Feature | Slime | Rollouts | Gap |
|---------|-------|----------|-----|
| **Flash Attention** | ✅ Automatic | ⚠️ Manual | Need to enable via PyTorch config |
| **Gradient Checkpointing** | ✅ TMS | ❌ No | **IMPORTANT**: Rollouts uses more memory |
| **CPU Offload** | ✅ Automatic | ❌ No | Can't train larger models |
| **Data Packing** | ✅ Yes | ❌ No | Rollouts wastes 30-50% compute |
| **Async Data Loading** | ✅ Yes | ⚠️ Basic | Rollouts has async rollouts but not data pipeline |

**Impact**: Slime is 2-3x more memory/compute efficient.

---

### 10. Configuration & Extensibility

| Feature | Slime | Rollouts | Gap |
|---------|-------|----------|-----|
| **Config System** | ✅ Comprehensive args | ⚠️ Dataclasses | Rollouts less user-friendly (no YAML/CLI) |
| **Custom Rollout Logic** | ✅ Plugin functions | ❌ Limited | **IMPORTANT**: Hard to extend rollout behavior |
| **Custom Rewards** | ✅ Plugin functions | ✅ Protocol-based | Both good, different approaches |
| **Middleware System** | ✅ Yes | ❌ No | Can't add custom routing/caching |
| **Extensibility** | ✅ Production hooks | ✅ Research hooks | Slime more batteries-included |

**Impact**: Slime easier to configure, Rollouts requires code changes.

---

## What You CAN Do with Rollouts Today

Despite gaps, Rollouts is solid for:

✅ **Single-GPU SFT training** (90% done, needs orchestration)
✅ **Single-GPU GRPO training** (90% done, needs orchestration)
✅ **Agent-based evaluation** (calculator, search, tool calling)
✅ **Custom environments** (screenshot, web search)
✅ **HuggingFace dataset integration**
✅ **Basic metrics logging** (JSONL + W&B)
✅ **Weight sync to SGLang/vLLM** (for inference)
✅ **Research prototyping** (clean codebase, easy to modify)

---

## What You CANNOT Do (vs. Slime)

### 🔴 Complete Blockers

1. **PPO training** - No value function critic implementation
2. **Multi-GPU training** - Can't scale beyond 1 GPU memory
3. **Asynchronous rollout/training** - Sequential only, wastes GPU
4. **Context parallelism** - Can't handle very long sequences
5. **Reward model training** - No built-in reward models for benchmarks
6. **Vision model RL** - No multi-modal support

### 🟡 Partial Blockers (Workarounds Exist)

7. **Off-policy training** - No weight version tracking (could track manually)
8. **Production robustness** - No fault tolerance (could add retry logic)
9. **Data packing** - Wastes compute (could implement custom collator)
10. **Advanced metrics** - Missing TFLOPs, clipfrac (could compute manually)

### 🟢 Minor Limitations

11. **Custom rollout patterns** - Limited flexibility (could extend InferenceEngine)
12. **Dynamic filtering** - Trains on all samples (could filter in dataset loader)
13. **Speculative decoding** - Missing speedup (SGLang supports, just not integrated)

---

## Effort to Close Critical Gaps

### Quick Wins (~1-2 days each)
- ✅ Finish SFT/RL loop orchestration (4 hours)
- ✅ Add LR scheduler (2-4 hours)
- ✅ Basic fault tolerance (retry logic) (4-6 hours)
- ✅ Advanced RL metrics (clipfrac, KL, entropy) (2-4 hours)

### Medium Effort (~3-5 days each)
- 🟡 FSDP integration (wrap PyTorchTrainingBackend)
- 🟡 PPO with value function (implement critic + multi-epoch)
- 🟡 Reward model training (add reward model backend)
- 🟡 Data packing (custom collator for variable-length seqs)

### Heavy Lift (~1-2 weeks each)
- 🔴 Decoupled SGLang architecture (major refactor)
- 🔴 Async training/rollout overlap (Ray integration)
- 🔴 Full Megatron integration (TP/PP/context parallel)
- 🔴 Production monitoring (health checks, alerts, dashboards)

---

## Recommendation

**For research/prototyping**: Rollouts is excellent. Clean code, easy to modify, good agent framework.

**For production RL at scale**: Use Slime. It has:
- ✅ Production-tested architecture (powers GLM-4.5/4.6)
- ✅ Full distributed training (FSDP, Megatron, TP/PP)
- ✅ Async training/rollout for efficiency
- ✅ Fault tolerance and monitoring
- ✅ Built-in reward models for benchmarks

**Hybrid approach**: Use Rollouts for agent evaluation + custom environments, then feed data to Slime for training.

**If you invest in Rollouts**: Focus on closing critical gaps (FSDP, PPO, reward models). The architecture is sound, just needs 2-3 weeks of focused work for production readiness.

---

## Gap Summary Table

| Category | Slime | Rollouts | Gap Severity |
|----------|-------|----------|--------------|
| **RL Algorithms** | PPO/GRPO/REINFORCE++/GSPO | GRPO only | 🔴 Critical |
| **Distributed Training** | Full (FSDP/Megatron/TP/PP) | Single GPU only | 🔴 Critical |
| **Architecture** | Decoupled async | Coupled sequential | 🔴 Critical |
| **Reward Models** | Built-in hub | Environment grading only | 🔴 Critical |
| **Memory Efficiency** | Data packing, offload, checkpointing | Standard PyTorch | 🟡 Important |
| **Monitoring** | Full profiling + health checks | Basic logging | 🟡 Important |
| **Fault Tolerance** | Auto retry + restart | None | 🟡 Important |
| **Custom Rollouts** | Plugin system | Limited | 🟢 Minor |
| **Config System** | Comprehensive CLI/YAML | Dataclasses only | 🟢 Minor |

---

## Final Answer to "What would I not be able to do?"

**If you wanted to run RL training like Slime, you would be blocked on:**

1. **Training models >24GB** (no multi-GPU support)
2. **Running PPO** (only GRPO works)
3. **Efficient training** (no async rollout/training overlap → 50% GPU waste)
4. **Long sequence RL** (no context parallelism, limited to ~4k tokens)
5. **Benchmark evaluation** (no built-in reward models for GSM8K, MATH, etc.)
6. **Production robustness** (no fault tolerance, health monitoring)
7. **Off-policy RL** (no weight version tracking for TIS)
8. **Vision model RL** (no multi-modal support)

**But you COULD do:**
- Single-GPU GRPO training (<24GB models)
- Custom environment evaluation (calculator, tools)
- Research prototyping with clean, modifiable codebase
- Agent-based workflows (better than Slime for this!)

**Bottom line**: Rollouts is 90% of the way to basic RL, but Slime has critical production features (distributed training, async architecture, PPO, reward models) that would take 1-2 months to replicate.
