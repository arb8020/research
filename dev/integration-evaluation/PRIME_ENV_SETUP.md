# Prime Environment Setup Guide

## Installation

### Installing Prime Environments

Prime environments are distributed via the Prime Hub. To install an environment:

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Prime CLI** (requires Python 3.11+):
   ```bash
   uv tool install prime
   ```

3. **Install the environment package**:
   ```bash
   prime env install siro/backend-bench
   ```

4. **Check installation info** (see alternate installation methods):
   ```bash
   prime env info siro/backend-bench
   ```

5. **Use the environment**:
   ```python
   from verifiers import load_environment
   env = load_environment("siro/backend-bench", gpu="local")
   ```

### Backend-Bench Specific

The backend-bench environment is available at:
- **Prime Hub**: https://app.primeintellect.ai/dashboard/environments/siro/backend-bench
- **Install command**: `prime env install siro/backend-bench`
- **Package name**: `backend-bench` (on PyPI as `backend-bench>=0.2.0`)

For deployment, the package can be installed via pip (it's available on PyPI):
```bash
pip install backend-bench>=0.2.0
```

This installs both `backend-bench` and its dependency `backendbench`, which provides the `backend_bench` module that verifiers uses.

## Configuration Pattern

Configs can now explicitly specify environment-specific parameters instead of relying on silent defaults.

### Example: Backend-Bench Configuration

In `configs/prime_backend_bench.py`, parameters are now explicit:

```python
@dataclass(frozen=True)
class IntegrationEvalConfig:
    # ... model config ...

    # Backend-bench specific parameters (explicit!)
    # Available options from backend_bench.load_environment signature:
    #   - gpu: 'local' | 'T4' | 'L4' | 'A100' | 'H100' | 'H200' | 'B200'
    #   - suite: 'smoke' | 'opinfo' | 'torchbench' | 'facto'
    #   - ops: list[str] | None (specific ops to test, or None for all)
    #   - num_turns: int (feedback loop iterations)
    #   - feedback_loop: 'until_correct' | 'until_max_turns' | 'none'
    backend_bench_gpu: str = "local"  # CHANGE TO 'A100' for Modal cloud GPU!
    backend_bench_suite: str = "torchbench"
    backend_bench_ops: List[str] | None = None
    backend_bench_num_turns: int = 1
    backend_bench_feedback_loop: str = "until_max_turns"
```

### How It Works

`local.py` automatically detects these parameters and passes them:

```python
# Extract environment-specific parameters from config
env_kwargs = {}
if hasattr(config, 'backend_bench_gpu'):
    env_kwargs['gpu'] = config.backend_bench_gpu
    logger.info(f"   GPU: {config.backend_bench_gpu}")
if hasattr(config, 'backend_bench_suite'):
    env_kwargs['suite'] = config.backend_bench_suite
    logger.info(f"   Suite: {config.backend_bench_suite}")
# ... etc ...

prime_env = load_environment(config.env_name, **env_kwargs)
```

## Print Interception

Backend-bench uses `print()` instead of logging. We intercept these prints:

```python
from shared import intercept_prints

class BackendBenchEnvironment:
    def __init__(self, prime_env, initial_state):
        self.logger = logging.getLogger(f"{__name__}.BackendBenchEnvironment")
        # ...

    async def on_assistant_message(self, message, state):
        # Intercept prints during env_response
        with intercept_prints(self.logger):
            updated_messages, updated_state = await trio_asyncio.aio_as_trio(
                self.prime_env.env_response
            )(self.messages, self.state)
```

All `print()` calls inside the context manager become `logger.info()` messages.

## Usage

### Running with Explicit Config

```bash
cd ~/research/dev/integration-evaluation
python local.py configs/prime_backend_bench.py
```

Output shows explicit parameters:
```
🎮 Loading Prime environment: siro/backend-bench
   GPU: local
   Suite: torchbench
   Explicit parameters: ['gpu', 'suite', 'ops', 'num_turns', 'feedback_loop']
```

### Changing GPU

Edit `configs/prime_backend_bench.py`:

```python
backend_bench_gpu: str = "A100"  # Instead of "local"
```

### Finding Available Parameters

For any Prime environment, check the `load_environment()` signature:

```python
# For backend-bench:
python3 -c "import backend_bench; import inspect; print(inspect.signature(backend_bench.load_environment))"
```

Output:
```
(suite: Literal['smoke', 'opinfo', 'torchbench', 'facto'] = 'torchbench',
 ops: list[str] | None = None,
 gpu: str = 'local',
 num_turns: int = 1,
 feedback_loop: Literal['until_correct', 'until_max_turns', 'none'] = 'until_max_turns')
```

## Pattern for Other Environments

To add explicit parameters for another environment (e.g., `siro/math-python`):

1. **Find available parameters:**
   ```bash
   python3 -c "import math_python; import inspect; print(inspect.signature(math_python.load_environment))"
   ```

2. **Add to config dataclass:**
   ```python
   @dataclass(frozen=True)
   class IntegrationEvalConfig:
       # Math-python specific parameters
       math_python_difficulty: str = "easy"
       math_python_max_attempts: int = 3
   ```

3. **Update local.py** (if needed, current implementation is generic):
   ```python
   # Already handles arbitrary parameters with pattern:
   # if hasattr(config, 'envname_paramname'):
   #     env_kwargs['paramname'] = config.envname_paramname
   ```

## Benefits

| Before | After |
|--------|-------|
| ❌ Silent defaults (`gpu='local'`) | ✅ Explicit in config (`backend_bench_gpu: str = "local"`) |
| ❌ Hidden parameters | ✅ Documented in dataclass |
| ❌ Print() output lost | ✅ Captured in JSONL logs |
| ❌ Can't see what's configurable | ✅ `inspect.signature()` shows all params |
| ❌ Need to read source | ✅ All options in config comments |

## Files Modified

1. **configs/prime_backend_bench.py**
   - Added explicit backend-bench parameters to dataclass
   - Added print interception to BackendBenchEnvironment
   - Documented all available options in comments

2. **local.py**
   - Auto-detects environment-specific parameters from config
   - Passes them to load_environment()
   - Logs which parameters are explicit

3. **shared/shared/print_interceptor.py** (new)
   - Intercepts print() and converts to logging
   - Context manager for safe cleanup
   - Supports tee mode (print + log)
