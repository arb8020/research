# Agent Kernel-Writing Architecture Decision

## Executive Summary

The kernel-writing agent should run **from local/orchestration layer and use bifrost exec** for remote execution. This provides the best combination of:
- Low latency for agent reasoning (local/API-based, not waiting for GPU)
- Clean separation of concerns (agent logic vs GPU execution)
- Scalability (one agent can manage multiple remote GPUs)
- Flexibility (agent can run anywhere, GPU execution is isolated)

---

## Current Architecture Analysis

### 1. Bifrost Execution Model

**What Bifrost Exec Does:**
- Executes arbitrary commands on remote GPU nodes via SSH
- Returns `ExecResult(stdout, stderr, exit_code)` immediately
- Supports environment variables and working directories
- Handles connection management, keepalive, and reconnection
- Used extensively in deploy.py (line 88), smoke_deploy.py (line 140), etc.

**Core Methods:**
```python
# Synchronous execution (waits for completion)
result = bifrost_client.exec(command, env=None, working_dir=None)

# Streaming execution (yields output as it arrives)
for line in bifrost_client.exec_stream(command):
    print(line)

# Detached execution (returns job_id, runs in background)
job = bifrost_client.run_detached(command, session_name="kernel_training")
status = bifrost_client.get_job_status(job.job_id)
```

**Key Properties:**
- Executes in context of remote home or deployed workspace
- Commands execute immediately (no queueing)
- SSH transport maintained with keepalive (30s intervals)
- Connection auto-reconnects on failure (retry logic with exponential backoff)
- File transfer via SFTP integrated (copy_files, upload_files, download_files)

### 2. Current Kernel Development Workflow

**Current Pattern (HOW_TO_ADD_KERNELS.md):**
1. Write kernel file locally (nvfp4_triton_kernel.py)
2. Register with BACKENDS.register() on import
3. Import in kernel_utils/smoke_test.py (lines 33-41)
4. Test locally: `python -m kernel_utils.smoke_test triton`
5. Deploy remotely: `python smoke_deploy.py --ssh root@host:port --backends triton`

**Local Testing Path:**
- Developer writes kernel on their machine
- Runs smoke_test.py locally (if they have GPU)
- Kernel templates provided (nvfp4_triton_kernel.py, nvfp4_cute_kernel.py)

**Remote Testing Path:**
- smoke_deploy.py uses bifrost.exec() to:
  - Deploy code (bifrost_client.push())
  - Setup Python environment
  - Run: `CUDA_VISIBLE_DEVICES={gpu_id} python -m kernel_utils.smoke_test`
- Results returned to stdout

### 3. Existing Agent Integration Patterns

**Integration Evaluation (local.py + deploy.py):**
- Evaluation orchestration runs locally
- Deploys to remote via bifrost
- Runs models on remote GPU (evaluating Prime Intellect environments)
- Streaming results back via tmux log capture

**Pattern:**
```
Local Machine (Agent/Orchestrator)
    ↓ bifrost.push() + bifrost.exec()
Remote GPU
    ↓
Model evaluation with local.py
    ↓ bifrost.copy_files()
Results back to Local
```

### 4. Kernel Writing/Compilation/Testing Workflow

**Multi-Stage Process:**

**Stage 1: Code Generation**
- Agent thinks about kernel optimization
- Generates Python code (Triton or CuTe CUDA)
- Writes to file on local filesystem

**Stage 2: Registration**
- Add BACKENDS.register() call
- Import in smoke_test.py
- (Could be automated)

**Stage 3: Testing**
- Correctness: Compare against reference kernel
- Performance: Benchmark against baseline
- Tests defined in kernel_utils/task.py (SMOKE_TESTS, CORRECTNESS_TESTS, PERFORMANCE_TESTS)

**Stage 4: Profiling** (Optional)
- NVIDIA Nsight Compute (NCU) profiling
- Torch profiler traces
- Used for optimization feedback loop

### 5. Remote vs Local Execution Considerations

**Latency Breakdown:**

**Local Execution:**
- Agent response time: 0-2s (LLM API call)
- File write: <1ms
- Import/registration: <100ms
- Smoke test (if local GPU): 100-500ms
- Total for iteration: 0.5-3s

**Remote Execution via Bifrost:**
- Agent response time: 0-2s (same LLM API call)
- File write (local): <1ms
- bifrost.exec() overhead: ~200-500ms (SSH setup, command exec)
- Remote smoke test: 100-500ms (same test, just on remote GPU)
- Result transfer: <100ms
- Total for iteration: 0.5-3.5s (negligible difference!)

**Network Latency:**
- SSH connection already established (persistent client)
- Exec is O(1) operation once connected
- Actual test execution time dominates (GPU kernels, not network)

---

## Recommended Architecture

### **Approach: Agent + Bifrost Exec Pattern**

```
┌─────────────────────────────────────────┐
│     Kernel-Writing Agent                │
│  (Claude or local orchestrator)          │
│                                          │
│  1. Analyze performance metrics          │
│  2. Generate optimized kernel code       │
│  3. Write to local file                  │
│  4. Register in smoke_test.py            │
└────────────┬────────────────────────────┘
             │
             ├─→ bifrost_client.exec(
             │        "python -m kernel_utils.smoke_test triton"
             │    )
             │
             ├─→ bifrost_client.copy_files(
             │        "remote_ncu_profiles/",
             │        "local_profiles/"
             │    )
             │
└────────────┴─→ bifrost_client.exec(
                  "nvidia-smi" (status check)
              )

┌──────────────────────────────────────────┐
│       Remote GPU Node (via SSH)          │
│                                          │
│  ├─ Python environment + torch/triton    │
│  ├─ Kernel code (synced via bifrost)     │
│  ├─ smoke_test.py infrastructure         │
│  ├─ NVIDIA profiling tools (ncu)         │
│  └─ GPU (H100, A100, RTX, etc.)          │
└──────────────────────────────────────────┘
```

### **Why This Pattern?**

**1. Agent Latency is Non-Critical**
- Agent doesn't need GPU access for reasoning
- Kernel code generation is CPU-bound (LLM + text)
- Testing is GPU-bound (actual kernel execution)
- Network roundtrip (500ms) << kernel compilation + execution (100-500ms)

**2. Clean Separation of Concerns**
- Agent: Responsible for code generation, optimization decisions
- GPU: Responsible for execution and profiling
- Bifrost: Manages SSH/deployment/file transfer
- Each component has single responsibility

**3. Scalability**
- One agent can target multiple GPUs (sequential or parallel)
- Agent doesn't consume GPU resources
- Agents can run on shared/weak machines (laptops, CI servers)
- GPUs stay focused on actual computation

**4. Flexibility & Portability**
- Agent code can be:
  - Local Python script
  - Remote Claude API call
  - Agentic framework (AutoGen, LangChain, etc.)
  - Manual human iteration
- GPU location can be:
  - H100 (expensive, fast iteration)
  - Consumer GPU (slower iteration, cheaper)
  - Cloud GPU (variable latency, managed)
- No coupling between agent and GPU locations

**5. Leverage Existing Patterns**
- bifrost.exec() already used throughout codebase:
  - deploy.py (line 88, 112, 119, 128, 132)
  - smoke_deploy.py (line 140)
  - All orchestration currently uses it
- kernel_utils/smoke_test.py provides test infrastructure
- File transfer (copy_files, upload_files) handles profiling artifacts

### **6. Reduced Complexity**
- No need for agent to have CUDA/triton installed
- No need to manage agent GPU memory
- Agent can run in CI/cloud without GPU access
- Testing is already implemented (smoke_test.py)

---

## Implementation Pattern

### **Local Agent → Remote Testing Loop**

```python
# pseudocode
from bifrost.client import BifrostClient
import anthropic

client = BifrostClient("root@gpu-host:22", "~/.ssh/id_ed25519")
llm = anthropic.Anthropic()  # or other API

# 1. Get performance baseline
result = client.exec("python -m kernel_utils.smoke_test reference")
baseline_perf = parse_perf_output(result.stdout)

# 2. Agent thinks about optimization
optimization_prompt = f"""
Current baseline: {baseline_perf}
Task: Generate optimized Triton kernel

Requirements:
- Must match signature: def triton_kernel(data: input_t) -> output_t:
- Register with: BACKENDS.register("triton", triton_kernel, ...)
- Use torch, triton, triton.language as tl

Baseline metrics:
{baseline_perf}

Generate improved kernel code:
"""

response = llm.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": optimization_prompt}]
)

# 3. Write generated kernel locally
kernel_code = extract_code_from_response(response.content[0].text)
with open("nvfp4_triton_kernel.py", "w") as f:
    f.write(kernel_code)

# 4. Deploy and test remotely
client.push(workspace_path="~/.bifrost/workspaces/kernel_opt")
result = client.exec("python -m kernel_utils.smoke_test triton")
new_perf = parse_perf_output(result.stdout)

# 5. Compare results
if is_improved(new_perf, baseline_perf):
    print("✅ Optimization successful!")
    speedup = baseline_perf.time / new_perf.time
    print(f"   Speedup: {speedup:.2f}x")
else:
    print("❌ No improvement, trying again...")
    # Agent iterates with feedback
```

### **Optional: NCU Profiling for Deep Analysis**

If agent needs detailed profiling feedback:

```python
# Get NCU profile remotely
result = client.exec("python -m kernel_utils.smoke_test triton --ncu")

# Download profiling data
client.copy_files(
    "/root/.bifrost/workspaces/kernel_opt/ncu_reports_remote/",
    "./ncu_profiles/",
    recursive=True
)

# Analyze locally
analysis = analyze_ncu_reports("./ncu_profiles/")
agent_feedback = format_for_agent(analysis)

# Agent uses feedback for next iteration
next_optimization = llm.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": optimization_prompt},
        {"role": "assistant", "content": kernel_code},
        {"role": "user", "content": agent_feedback}
    ]
)
```

---

## Why NOT: Agent on GPU

### **Problems with Running Agent on GPU**

**1. Inefficient Resource Use**
- GPUs are expensive (H100: $1-2/hour)
- Agent reasoning is CPU-bound (LLM, not CUDA)
- Keeping GPU idle during agent thinking = $$/second wasted
- Especially bad if using cloud GPUs (per-minute billing)

**2. Latency Not Improved**
- Agent doesn't benefit from GPU for code generation
- LLM API calls (0-2s) >> local execution time
- SSH roundtrip (500ms) is negligible vs iteration time
- Actual bottleneck is agent quality, not network latency

**3. Operational Complexity**
- Need to install/manage LLM libraries on GPU node
- Need to handle GPU memory for both agent + kernel testing
- Memory contention issues
- SSH terminal complexity for interactive agent

**4. Scalability Issues**
- One agent ≠ multiple GPUs efficiently
- Agent blocks GPU while thinking
- Can't easily swap GPUs if one is busy
- Hard to parallelize (one agent per GPU = expensive)

**5. Portability Problems**
- Agent tied to specific GPU hardware
- Different API endpoints for different cloud providers
- Harder to test locally vs remote
- Couples agent logic to GPU location

---

## Alternative: Async Agent on GPU (Not Recommended)

**Scenario:** Agent runs async on GPU, uses bifrost to spawn child processes

**Problem:** Adds complexity without benefits
- Still consuming GPU resources while thinking
- Async patterns are harder to reason about
- No clear win over local agent + bifrost exec
- Same network latency either way

**Verdict:** Skip this. Use local agent + bifrost exec instead.

---

## Summary Table

| Aspect | Local Agent + Bifrost Exec | Agent on GPU |
|--------|---------------------------|-------------|
| **Agent Latency** | Excellent (no network wait for reasoning) | No improvement (reasoning same speed) |
| **GPU Efficiency** | 100% (only for actual kernel testing) | 0% (idle during thinking) |
| **Cost** | $$$ per hour GPU time, $ per hour agent | $$$ per hour GPU time (always) |
| **Scalability** | One agent → many GPUs | One GPU per agent |
| **Complexity** | Simple (bifrost + Python) | Complex (async, CUDA, memory mgmt) |
| **Portability** | Any machine | Requires CUDA/GPU |
| **Testing** | Easy (test locally first) | Hard (need GPU even for dev) |
| **Existing Patterns** | Matches deploy.py, smoke_deploy.py | No precedent in codebase |

---

## Implementation Checklist

If building kernel-writing agent:

- [ ] Create BifrostClient for target GPU node
- [ ] Agent generates kernel code (local, no GPU needed)
- [ ] bifrost.exec() runs smoke_test.py remotely
- [ ] Parse test output for:
  - Correctness (PASS/FAIL)
  - Performance metrics (speedup vs baseline)
  - Error messages
- [ ] (Optional) bifrost.copy_files() to download NCU profiles
- [ ] Agent iterates based on results
- [ ] Version control generated kernels (git)
- [ ] Store optimization history (metrics, timestamps)

---

## Key Files & References

**Bifrost Client:**
- `/Users/chiraagbalu/research/bifrost/bifrost/client.py` (lines 296-357 for exec methods)

**Kernel Testing Infrastructure:**
- `/Users/chiraagbalu/research/dev/kernels-gpumode/kernel_utils/smoke_test.py`
- `/Users/chiraagbalu/research/dev/kernels-gpumode/HOW_TO_ADD_KERNELS.md`

**Deployment Patterns:**
- `/Users/chiraagbalu/research/dev/integration-evaluation/deploy.py` (lines 88-145)
- `/Users/chiraagbalu/research/dev/kernels-gpumode/smoke_deploy.py` (lines 46-150)

**Evaluation Execution Pattern:**
- `/Users/chiraagbalu/research/dev/integration-evaluation/local.py` (runs on remote GPU)
- Model evaluation uses same pattern (deploy locally, execute remotely)

