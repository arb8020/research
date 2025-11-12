# Setup Notes: Prime Integration Evaluation

## What Was Created

Following the pattern from `~/wafer_stuff/clicker`, I created a clean evaluation setup for Prime Intellect integration.

### Files Created

```
dev/integration-evaluation/
├── README.md              # Full documentation
├── SETUP_NOTES.md         # This file
├── local.py               # Main entrypoint (matches clicker/run_eval.py)
├── simple_2048_env.py     # Test environment (Prime verifiers format)
└── config/
    └── prime_2048.py      # Config (matches clicker config pattern)
```

### Design Decisions

#### 1. **Followed Clicker Pattern**

From `~/wafer_stuff/clicker/configs/19_gemini_2_5_pro_smoke_01.py`:
- ✅ Frozen dataclass for config
- ✅ Groups: ModelConfig, DatasetConfig, etc.
- ✅ Conversion methods: `to_endpoint()`, `to_eval_config()`
- ✅ Explicit validation in `__post_init__`

#### 2. **Simplified for Prime Integration**

Removed from clicker pattern:
- ❌ GPUConfig (not needed for eval)
- ❌ DeploymentConfig (no deployment yet)
- ❌ Complex filter logic (simpler for now)

Added for Prime:
- ✅ `reward_fn` integration
- ✅ Prime environment loading
- ✅ Dataset conversion utilities

#### 3. **Entry Point Pattern**

`local.py` matches `clicker/run_eval.py`:
- Loads `.env` for API keys
- Loads config from Python file
- Runs evaluation
- Saves results

But adapted for Prime:
- Uses `rollouts.evaluation.evaluate()` directly
- Creates Prime environment first
- Converts Prime rubric to `RewardFunction`

---

## How to Use

### Basic Usage

```bash
cd ~/research/rollouts/dev/integration-evaluation
python local.py --config config/prime_2048.py
```

### With Different Model

Edit `config/prime_2048.py`:

```python
# Use Gemini (matches clicker Gemini config)
model_name: str = "gemini-2.5-flash"
provider: str = "openai"  # Gemini via OpenAI-compatible API
api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key_env_var: str = "GEMINI_API_KEY"
```

### Create New Config

```bash
cp config/prime_2048.py config/my_eval.py
# Edit my_eval.py
python local.py --config config/my_eval.py
```

---

## What I Learned from Clicker Gemini Evaluation

### Good Patterns to Keep

1. **Config structure is excellent**
   ```python
   @dataclass(frozen=True)
   class ModelConfig:
       model_name: str
       provider: Literal["gemini", "openai", "anthropic", "sglang"]
       temperature: float = 0.0
       # ...
   ```
   - Explicit types
   - Sensible defaults
   - Frozen for immutability

2. **Validation in `__post_init__`**
   ```python
   def __post_init__(self):
       assert 0.0 <= self.temperature <= 2.0
       assert self.max_output_tokens > 0
   ```
   - Catches errors early
   - Clear error messages

3. **Conversion methods**
   ```python
   def to_endpoint(self) -> Endpoint:
       return Endpoint(...)
   ```
   - Clean separation between config and runtime types
   - Easy to test

4. **Thin wrapper pattern**
   ```python
   # run_eval.py
   load_dotenv()
   from rollouts.run_eval import main
   sys.exit(main())
   ```
   - Just handles environment setup
   - Delegates to library code

### Differences for Prime Integration

| Clicker | Prime Integration |
|---------|------------------|
| Uses rollouts.run_eval.main() | Custom evaluation loop |
| ScreenSpot environment | Prime verifiers environment |
| Direct reward in environment | Prime rubric adapter |
| to_trajectory converter | Prime dataset converter |

The key difference: **Clicker has domain-specific environments (ScreenSpot)** while **Prime integration is generic (any verifiers environment)**.

---

## About hud-text-2048

### Problem

The actual `hud-text-2048` environment from Prime Hub isn't installable via pip:
```bash
uv pip install hud-text-2048  # Doesn't work
```

The Prime Hub page (https://app.primeintellect.ai/dashboard/environments/hud/hud-text-2048) doesn't provide installation instructions.

### Solution

Created `simple_2048_env.py` which:
- ✅ Implements the same interface as Prime Hub environments
- ✅ Uses `verifiers.SingleTurnEnv` (official base class)
- ✅ Has custom parser and rubric
- ✅ Works immediately for testing

When actual Prime Hub environments become available:
```python
# Just swap the environment creation
from verifiers import load_environment
prime_env = load_environment("hud-text-2048")
# Rest of code stays the same!
```

---

## Architecture Notes

### Reward Flow

```
1. Create Prime Environment (simple_2048_env.py)
   ↓
2. prime_reward_fn(env) creates RewardFunction
   ↓
3. RewardFunction: Trajectory -> Trajectory
   - Uses Prime parser to extract answer
   - Uses Prime rubric to grade
   - Returns trajectory with rewards populated
   ↓
4. EvalConfig uses reward_fn
   ↓
5. evaluate() runs agent, applies reward_fn
   ↓
6. EvalReport with results
```

### Why This Pattern Works

1. **Separation of concerns**
   - Prime environment = dataset + parser + rubric
   - Reward function = adapter to rollouts format
   - Config = orchestration settings

2. **Composability**
   - Can swap Prime environments easily
   - Can add custom signals to Prime rewards
   - Can use Prime rewards in training later

3. **Testability**
   - Each component testable independently
   - Simple 2048 env for quick tests
   - Real Prime envs for production

---

## Next Steps

### Short Term (Testing)

1. Run with local sglang:
   ```bash
   python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct
   python local.py --config config/prime_2048.py
   ```

2. Try with Gemini (like clicker):
   - Edit config to use Gemini
   - Set GEMINI_API_KEY in .env
   - Run evaluation

3. Check results:
   ```bash
   cat results/integration-evaluation/report.json
   ls results/integration-evaluation/samples/
   ```

### Medium Term (Real Prime Environments)

1. Figure out how to install hud-text-2048
   - Check Prime Intellect docs
   - Ask Prime support
   - Or use their API directly

2. Swap in real environment:
   ```python
   # In local.py
   from verifiers import load_environment
   prime_env = load_environment("hud-text-2048")
   ```

3. Test with multiple Prime Hub environments

### Long Term (Training Integration)

1. Modify `training/loops/rl_loop.py` to accept RewardFunction
2. Use Prime rewards during GRPO training
3. Test full RL pipeline with Prime environments

---

## Comparison: Clicker vs Integration Eval

### File Structure

```
clicker/
├── run_eval.py              # Thin wrapper
├── config.py                # Config dataclasses
└── configs/
    └── 19_gemini_*.py       # Specific eval config

integration-evaluation/
├── local.py                 # Direct evaluation
├── simple_2048_env.py       # Test environment
└── config/
    └── prime_2048.py        # Config instance
```

### Config Pattern

**Clicker** (separate modules):
```python
# config.py has dataclasses
# configs/19_gemini_*.py has instances
config = Config(
    model=ModelConfig(...),
    dataset=DatasetConfig(...),
    ...
)
```

**Integration Eval** (combined):
```python
# config/prime_2048.py has both
@dataclass(frozen=True)
class IntegrationEvalConfig:
    ...

config = IntegrationEvalConfig()
```

Both patterns work! Clicker's is better for complex shared configs. Ours is simpler for one-off evaluations.

---

## Gemini Evaluation Thoughts

From `configs/19_gemini_2_5_pro_smoke_01.py`:

### What I Liked

1. **Clean provider abstraction**
   ```python
   provider="openai"  # Gemini through OpenAI-compatible API
   ```
   Makes sense - use standard OpenAI client with different endpoint

2. **Explicit smoke test**
   ```python
   limit=5,  # Just 5 samples for smoke test
   ```
   Good practice - test before full run

3. **Config documentation**
   ```python
   """Gemini 2.5 Pro - Smoke test (5 samples from vscode_macos).

   Quick validation of Gemini 2.5 Pro before full evaluation.

   Usage:
       python run_eval.py --config ...
   """
   ```
   Docstring at top with usage - very clear!

### What Could Be Better

1. **API key handling**
   - Uses env var `GEMINI_API_KEY`
   - Could document this in .env.example

2. **Temperature 0.0 hardcoded**
   - Makes sense for evaluation
   - Could be config option for exploration

3. **Single-turn assumption**
   ```python
   max_turns=1,  # ScreenSpot is single-turn
   ```
   - Good for ScreenSpot
   - Prime environments might be multi-turn

### Overall Assessment

**Excellent config pattern!** Clean, explicit, well-documented. I adopted the same style for Prime integration.

---

## Summary

✅ Created `dev/integration-evaluation/` following clicker pattern
✅ Supports any Prime verifiers environment
✅ Clean config pattern with validation
✅ Ready to test with local sglang or Gemini
✅ Documented thoroughly

**Next**: Run evaluation and verify results!
