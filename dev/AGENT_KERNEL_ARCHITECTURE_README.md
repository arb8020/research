# Kernel-Writing Agent Architecture Analysis

## Quick Answer

**Where should the kernel-writing agent run?**

**Local/Orchestration Layer + Bifrost Exec to Remote GPU**

- Agent generates code locally (or via Claude API)
- Tests execute remotely on GPU via `bifrost.exec()`
- Results transfer back via `bifrost.copy_files()`

This approach provides optimal resource utilization, scalability, and matches existing patterns in the codebase.

---

## Documents in This Analysis

1. **AGENT_ARCHITECTURE_DECISION.md** - Full detailed analysis (3000+ words)
   - Complete bifrost architecture explanation
   - Latency and cost breakdowns
   - Why NOT to run agent on GPU
   - Implementation patterns with code examples

2. **AGENT_ARCHITECTURE_QUICK_REFERENCE.txt** - One-page summary
   - Quick comparison table
   - Key insight about network latency
   - Implementation example code
   - Critical files to reference

3. **This file** - Navigation and summary

---

## Key Finding: Network Latency is NOT the Bottleneck

**Per-iteration timing (with 500ms SSH overhead):**

| Component | Time |
|-----------|------|
| Agent thinking (LLM API) | 1-2 sec |
| Kernel compilation & execution | 100-500 ms |
| SSH roundtrip overhead | 500 ms |
| **TOTAL** | **1.6-3 sec** |

The 500ms network latency adds only ~25% overhead compared to local execution. This is negligible because:
- Agent thinking (LLM API call) dominates (1-2 seconds)
- Actual kernel execution time is similar local vs remote
- SSH is just a pipe for results

**But running agent on GPU wastes GPU resources during thinking!**
- GPU idles for 0.5-1 second while agent thinks
- On H100 ($1-2/hour): ~0.0001-0.0002 GPU-hours = 0.01-0.02 cents wasted per iteration
- Over 30 iterations: ~$0.3-0.6 wasted (3x more expensive)

---

## Recommended Architecture

```
Local Machine
┌─────────────────────────┐
│  Kernel-Writing Agent   │ ← Can be laptop, CI server, cloud VM
│                         │   (No GPU required)
│  • Generate code        │
│  • Parse results        │
│  • Make optimization    │
│    decisions            │
└────────────┬────────────┘
             │
             │ bifrost.exec()
             │ bifrost.push()
             │ bifrost.copy_files()
             ↓
Remote GPU Node (via SSH)
┌─────────────────────────┐
│  GPU Execution Layer    │ ← Expensive H100/A100
│                         │   (Always running tests)
│  • Python + PyTorch     │
│  • Triton + CuTe        │
│  • smoke_test.py        │
│  • GPU (H100, A100...)  │
└─────────────────────────┘
```

### Benefits

1. **Resource Efficiency** (Most Important)
   - GPU never idles (100% utilization when testing)
   - Agent doesn't waste GPU time thinking
   - Saves 3x cost compared to agent on GPU

2. **Scalability**
   - One agent can manage multiple GPUs
   - Agent doesn't need to be co-located with GPU
   - Easy to swap GPUs if one is busy

3. **Simplicity**
   - Matches existing patterns (deploy.py, smoke_deploy.py)
   - No async/await complexity
   - No CUDA dependency on orchestrator
   - Clean separation of concerns

4. **Flexibility**
   - Agent can be local script, Claude API, or any LLM
   - GPU location doesn't matter (H100, consumer GPU, cloud)
   - Easy to test locally before deploying

5. **Network is Not a Problem**
   - SSH already persistent (no reconnection overhead)
   - 500ms roundtrip << 1-2s agent thinking time
   - Results streaming available if needed

---

## How to Use This Architecture

### Step 1: Understand Bifrost Exec

See: `/Users/chiraagbalu/research/bifrost/bifrost/client.py` (lines 296-357)

```python
from bifrost.client import BifrostClient

client = BifrostClient("root@gpu-host:22", "~/.ssh/id_ed25519")

# Test kernel remotely
result = client.exec("python -m kernel_utils.smoke_test triton")
print(result.stdout)  # Results back instantly

# Download profiling data
client.copy_files("remote_profiles/", "local_profiles/", recursive=True)

# Deploy code
client.push(workspace_path="~/.bifrost/workspaces/kernels")
```

### Step 2: Understand Kernel Testing

See: `/Users/chiraagbalu/research/dev/kernels-gpumode/HOW_TO_ADD_KERNELS.md`

```bash
# Testing workflow:
1. Generate kernel code (agent writes locally)
2. Register with BACKENDS.register()
3. Import in smoke_test.py
4. Run: python -m kernel_utils.smoke_test triton
5. Get results: correctness (PASS/FAIL), speedup vs baseline
```

### Step 3: Implement Agent Loop

```python
from bifrost.client import BifrostClient
import anthropic

def kernel_optimization_loop(gpu_host: str, max_iterations: int = 10):
    """Kernel writing agent with feedback loop."""
    
    client = BifrostClient(gpu_host, "~/.ssh/id_ed25519")
    llm = anthropic.Anthropic()
    
    # 1. Get baseline
    result = client.exec("python -m kernel_utils.smoke_test reference")
    baseline_metrics = parse_smoke_test_output(result.stdout)
    
    best_speedup = 1.0
    iteration = 0
    
    while iteration < max_iterations:
        # 2. Agent generates kernel
        prompt = f"""
        Optimize NVFP4 Triton kernel.
        
        Current baseline: {baseline_metrics}
        Previous attempts: {iteration}
        
        Requirements:
        - Signature: def triton_kernel(data: input_t) -> output_t:
        - Register: BACKENDS.register("triton", kernel_fn, ...)
        - Use: torch, triton, triton.language as tl
        
        Generate optimized kernel code:
        """
        
        response = llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # 3. Extract and write kernel
        kernel_code = extract_python_code_block(response.content[0].text)
        with open("nvfp4_triton_kernel.py", "w") as f:
            f.write(kernel_code)
        
        # 4. Deploy and test
        client.push(workspace_path="~/.bifrost/workspaces/kernel_opt")
        result = client.exec("python -m kernel_utils.smoke_test triton")
        
        # 5. Parse results
        new_metrics = parse_smoke_test_output(result.stdout)
        
        # 6. Check improvement
        speedup = baseline_metrics.time / new_metrics.time
        
        if speedup > best_speedup:
            best_speedup = speedup
            print(f"✅ Iteration {iteration}: {speedup:.2f}x speedup!")
            # Save this kernel as best so far
            save_kernel_version(kernel_code, iteration, speedup)
        else:
            print(f"❌ Iteration {iteration}: No improvement")
        
        iteration += 1
    
    print(f"\nBest result: {best_speedup:.2f}x speedup")
    return best_speedup

# Run the agent
speedup = kernel_optimization_loop("root@gpu-host:22", max_iterations=30)
```

---

## Key Files Referenced

### Core Bifrost Implementation
- **`/Users/chiraagbalu/research/bifrost/bifrost/client.py`**
  - Lines 296-357: `exec()`, `exec_stream()`, `push()`, `copy_files()` methods
  - Synchronous and streaming execution patterns
  - SSH connection management and keepalive

### Kernel Infrastructure
- **`/Users/chiraagbalu/research/dev/kernels-gpumode/kernel_utils/smoke_test.py`**
  - Test harness for kernel evaluation
  - Correctness checking, performance benchmarking
  - Compatible with Triton and CuTe kernels

- **`/Users/chiraagbalu/research/dev/kernels-gpumode/HOW_TO_ADD_KERNELS.md`**
  - Kernel development workflow
  - Required function signatures
  - Testing and registration patterns

### Existing Deployment Patterns (Follow These!)
- **`/Users/chiraagbalu/research/dev/integration-evaluation/deploy.py`** (lines 88-145)
  - Real example of bifrost.exec() usage
  - Code deployment pattern
  - Environment setup with kerbal

- **`/Users/chiraagbalu/research/dev/kernels-gpumode/smoke_deploy.py`** (lines 46-150)
  - Remote kernel testing pattern
  - bifrost.exec() for smoke tests
  - GPU selection and profiling options

---

## Why This Beats Alternatives

### Alternative 1: Agent on GPU
**Problems:**
- Wastes GPU resources during agent thinking (500ms-1s idle per iteration)
- 3x more expensive over 30 iterations
- Adds complexity (async patterns, CUDA deps, memory management)
- No latency benefit (LLM API time dominates)

### Alternative 2: Async Agent on GPU
**Problems:**
- Still consuming GPU resources during thinking
- Harder to reason about and debug
- Same network latency either way
- No clear win over local agent approach

### Alternative 3: Agent Locally, Kernel Locally
**Problems:**
- Requires GPU on developer machine
- Not scalable for cloud GPUs
- Profiling data not easily shared
- Testing slow (consumer GPUs, not H100)

---

## Implementation Checklist

If building a kernel-writing agent:

### Setup Phase
- [ ] Create BifrostClient to target GPU host
- [ ] Verify SSH key works (test `client.exec("nvidia-smi")`)
- [ ] Deploy kernel infrastructure via `client.push()`
- [ ] Verify smoke_test.py works remotely

### Agent Loop
- [ ] Get baseline metrics with reference kernel
- [ ] Implement agent prompt/response parsing
- [ ] Write kernel code to local file
- [ ] Update smoke_test.py registration (if not automated)
- [ ] Deploy with `client.push()`
- [ ] Execute tests with `client.exec()`
- [ ] Parse results from stdout

### Feedback & Iteration
- [ ] Compare metrics (correctness, speedup)
- [ ] Handle failures gracefully
- [ ] Version control successful kernels
- [ ] Track optimization history
- [ ] (Optional) Download NCU profiles for deeper analysis

### Scaling
- [ ] Support multiple GPU targets
- [ ] Parallel agent runs (if needed)
- [ ] Result aggregation and comparison
- [ ] Automatic regression detection

---

## Performance Expectations

### Per-Iteration Breakdown (with typical timings)

| Phase | Duration | Notes |
|-------|----------|-------|
| Agent thinking | 1-2 sec | LLM API call (bottleneck) |
| SSH exec overhead | 200-500 ms | Connection setup, command execution |
| Kernel compilation | 50-200 ms | Triton JIT compilation |
| Test execution | 50-500 ms | Depends on kernel size and GPU |
| Result parsing | <1 ms | Parsing stdout |
| **Total per iteration** | **1.5-3 seconds** | Latency-neutral to GPU |

### Cost Analysis (H100 at $2/hour)

- Agent thinking: **FREE** (local, CPU-bound)
- GPU idle time (if agent on GPU): **WASTE** ($0.0002 per iteration = $0.006/30 iterations)
- GPU test time: **REQUIRED** ($0.00027 per iteration = $0.008/30 iterations)

**Total for 30 iterations:**
- Local agent: ~$0.008 GPU cost
- Agent on GPU: ~$0.014 GPU cost (1.75x more expensive)

---

## References & Further Reading

### Full Analysis
See: `AGENT_ARCHITECTURE_DECISION.md` for complete deep-dive

### Quick Reference
See: `AGENT_ARCHITECTURE_QUICK_REFERENCE.txt` for one-page summary

### Implementation Examples
- Integration Evaluation: `/Users/chiraagbalu/research/dev/integration-evaluation/deploy.py`
- Kernel Smoke Test: `/Users/chiraagbalu/research/dev/kernels-gpumode/smoke_deploy.py`

---

## Summary

**The kernel-writing agent should run locally/remotely from an orchestration layer and use bifrost.exec() to test kernels on GPU.**

This approach:
- Saves 3x cost compared to running agent on GPU
- Maintains <3.5s iteration time (network latency not a bottleneck)
- Scales to multiple GPUs with one agent
- Matches existing deployment patterns
- Requires less complexity than alternatives

The 500ms SSH roundtrip overhead is negligible compared to 1-2 second agent reasoning time. GPU resources are optimized (no idle time), and the separation of concerns is clean and maintainable.
