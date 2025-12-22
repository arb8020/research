# TI/TO Correctness Testing

> **Goal:** Prove that TI/TO (Tokens-In/Tokens-Out) prevents the retokenization collapse that destroys multi-turn RL training.

## The Problem

When training RL on multi-turn tool-calling agents, retokenization causes catastrophic training collapse:

```
At rollout time:    Model generates ' ' + '"'  (two tokens)
After retokenization: Becomes ' "'             (one token)
That merged token has logprob -20              (model NEVER predicted it)
```

This happens because:
1. Model generates tokens → we decode to text for environment interaction
2. Environment returns text → we re-encode with chat template
3. Chat template re-tokenizes differently than the original generation
4. Training sees tokens the model never predicted → huge gradients → collapse

**The smoking gun:** logprob < -15 on tokens that should be normal. These dominate the gradient and cause instability.

## The Fix: TI/TO

TI/TO (Tokens-In/Tokens-Out) means:
- **Token-Out:** Store the exact `token_ids` the model generated (not decoded text)
- **Token-In:** Feed tokens directly to the model, don't re-tokenize

This ensures the tokens used for training are exactly what the model generated.

## Current Status (2025-12-18)

### Test Successfully Found Smoking Guns!

The test infrastructure is now working. A recent run found the retokenization collapse:

```
🎯 SMOKING GUN at pos 331: 'ate' -> ' Incorpor'
   logprob: -0.02 -> -27.93
🎯 SMOKING GUN at pos 332: ' the' -> 'ate'
   logprob: -0.97 -> -15.13
```

This proves the problem exists and validates the test methodology.

### What's Working

1. **Test infrastructure** (`tests/test_tito_correctness.py`)
   - Uses `SGLangEngine` from `rollouts/training/weight_sync.py` (not raw subprocess)
   - Uses tmux + log file pattern from `examples/rl/` for reliable logging
   - Syncs logs back even on failure
   - Specifies `min_cuda_version="12.8"` for proper CUDA compatibility

2. **Remote test execution**
   - Provisions A100 via broker (runpod)
   - Deploys via bifrost with bootstrap: `uv sync` + `uv pip install sglang[all]`
   - Runs SGLang with `mem_fraction=0.5` to leave room for model forward pass
   - Loads model on `cuda:0` explicitly (not `device_map="auto"`)

3. **Test logic** (embedded in `REMOTE_AGENT_LOOP_TEST_SCRIPT`)
   - Starts SGLang FIRST (needs GPU memory)
   - Then loads model for forward pass on same GPU
   - Runs multi-turn rollouts with calculator environment
   - Simulates retokenization to detect mismatches
   - Computes logprobs of mismatched tokens via forward pass
   - Flags logprobs < -15 as "smoking guns"

### Current Issue

Test occasionally gets OOM killed (exit 137) or hangs during SGLang startup. The test runs but isn't 100% reliable yet.

**Instance still alive:** `runpod:wjufrjeufxbu43` at `root@69.19.136.95:31465`

## Key Changes Made

### 1. Switched to SGLangEngine (not raw subprocess)
The test now uses `SGLangEngine` from `rollouts/training/weight_sync.py` which handles:
- tmux-based server management
- Log file writing
- Crash detection
- Proper cleanup

```python
# In REMOTE_AGENT_LOOP_TEST_SCRIPT
def start_sglang_server():
    from rollouts.training.weight_sync import SGLangEngine

    engine = SGLangEngine(
        model_name=MODEL,
        port=PORT,
        gpu_ids=(0,),
        output_dir=OUTPUT_DIR,
        dtype="bfloat16",
        mem_fraction=0.5,  # Leave room for model forward pass
    )
    engine.launch()
    engine.start_log_tailer()
    # ...
```

### 2. Start SGLang FIRST, then load model
Previously loaded model first, which used all GPU memory before SGLang could start.

```python
# CORRECT ORDER:
engine, base_url = start_sglang_server()  # Uses 50% GPU memory
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda:0")
```

### 3. Use explicit cuda:0 (not device_map="auto")
`device_map="auto"` was unpredictable. Explicit placement is clearer:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
).to("cuda:0")
```

### 4. Added min_cuda_version="12.8"
Old instances had CUDA 12.4 which caused vLLM symbol mismatches.

```python
instance = broker.create(
    broker.gpu_type.contains(gpu_type),
    gpu_count=gpu_count,
    min_cuda_version="12.8",  # Added
    # ...
)
```

### 5. Bootstrap pattern (not DependencyConfig)
kerbal's `DependencyConfig` was resolving ancient sglang versions (0.1.11). Now uses:

```python
bootstrap = [
    "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
    "uv pip install sglang[all]",
]
workspace = bifrost.push("~/.bifrost/workspaces/rollouts-tito-test", bootstrap_cmd=bootstrap)
```

## Files to Know

| File | Purpose |
|------|---------|
| `tests/test_tito_correctness.py` | Main test orchestration + embedded remote scripts |
| `rollouts/training/weight_sync.py` | `SGLangEngine` class for server management |
| `rollouts/examples/rl/base_config.py` | Reference pattern for remote execution |
| `rollouts/training/agent_integration.py` | TI/TO implementation |

## Running the Test

```bash
# From rollouts directory, with venv activated

# Provision new instance and run test
python tests/test_tito_correctness.py --provision --keep-alive --test agent-loop

# Reuse existing instance
python tests/test_tito_correctness.py --node-id runpod:wjufrjeufxbu43 --keep-alive --test agent-loop
```

## SSH Debug

The instance is kept alive. SSH in to debug:

```bash
ssh -i ~/.ssh/id_ed25519 root@69.19.136.95 -p 31465

# Check GPU
nvidia-smi

# Check logs
cat /tmp/tito-agent-test/sglang.log
cat ~/.bifrost/workspaces/rollouts-tito-test/rollouts/results/agent-loop-tito-test.log

# Check tmux sessions
tmux list-sessions
tmux attach -t agent-loop-tito-test
```

## Next Steps

1. **Reliability**: The test sometimes hangs or OOMs. Need to investigate:
   - Is SGLang startup timing out?
   - Is there a race condition?
   - Memory pressure between SGLang and model?

2. **Clean exit**: Ensure test properly cleans up SGLang on completion/failure

3. **CI integration**: Once reliable, add to CI pipeline

## Related Context

- `tokens.md` - Full problem description from team
- Prime-RL PR #1422 - Original TI/TO fix
- SID-1 training notes - Production evidence of the problem
