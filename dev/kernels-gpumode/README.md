# Prime Intellect Integration Evaluation

This directory contains the setup for evaluating models with Prime Intellect verifiers environments.

## Overview

**Phase 1: Evaluation** - Use Prime environments and rubrics to evaluate models with the rollouts framework.

Pattern inspired by `~/wafer_stuff/clicker` evaluation setup.

## Directory Structure

```
dev/integration-evaluation/
├── README.md                 # This file
├── local.py                  # Main entrypoint (run evaluations)
├── simple_2048_env.py        # Simple 2048 environment for testing
└── config/
    └── prime_2048.py         # Example config for 2048 evaluation
```

## Quick Start

### 1. Ensure sglang is running (or use another provider)

```bash
# Option A: Local sglang
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000

# Option B: Edit config to use OpenAI/Anthropic/Gemini
```

### 2. Run evaluation

```bash
cd ~/research/rollouts/dev/integration-evaluation
python local.py --config config/prime_2048.py
```

### 3. Check results

```bash
ls results/integration-evaluation/
# report.json - Summary metrics
# samples/ - Individual sample results
# trajectories/ - Full trajectories
```

## How It Works

### Architecture

```
Prime Environment (verifiers)
    ↓
simple_2048_env.py creates SingleTurnEnv
    ↓
prime_reward_fn() creates RewardFunction
    ↓
EvalConfig configures evaluation
    ↓
evaluate() runs agent and computes rewards
    ↓
EvalReport with results
```

### Code Flow

1. **`local.py`** - Main entrypoint
   - Loads config from `config/prime_2048.py`
   - Creates Prime environment
   - Creates reward function from Prime rubric
   - Runs evaluation with rollouts framework

2. **`simple_2048_env.py`** - Test environment
   - Simple 2048 game for testing
   - Uses `verifiers.SingleTurnEnv`
   - Custom parser and rubric

3. **`config/prime_2048.py`** - Configuration
   - Model settings (provider, endpoint, etc.)
   - Environment settings (which Prime env to use)
   - Evaluation settings (max_concurrent, output_dir, etc.)

## Configuration

### Creating a New Config

```python
# config/my_eval.py

from pathlib import Path
from dataclasses import dataclass
from rollouts.dtypes import Endpoint, EvalConfig

@dataclass(frozen=True)
class IntegrationEvalConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    provider: str = "sglang"
    api_base: str = "http://localhost:30000/v1"

    # Environment
    env_name: str = "my_env"
    num_samples: int = 50

    # Evaluation
    eval_name: str = "my_eval"
    max_concurrent: int = 4
    output_dir: Path = Path("results/my_eval")

    def to_endpoint(self) -> Endpoint:
        return Endpoint(
            provider=self.provider,
            model=self.model_name,
            api_base=self.api_base,
        )

    def to_eval_config(self, reward_fn) -> EvalConfig:
        return EvalConfig(
            reward_fn=reward_fn,
            max_concurrent=self.max_concurrent,
            output_dir=self.output_dir,
            eval_name=self.eval_name,
        )

config = IntegrationEvalConfig()
```

## Using Real Prime Hub Environments

To use actual environments from https://app.primeintellect.ai/dashboard/environments:

### Option 1: If package is available

```bash
# Install from Prime Hub
uv pip install hud-text-2048  # Example

# Then load in your code
from verifiers import load_environment
prime_env = load_environment("hud-text-2048")
```

### Option 2: Create custom environment

```python
# Create your own environment matching Prime's interface
from verifiers import SingleTurnEnv, Rubric, Parser
from datasets import Dataset

# Load your dataset
dataset = Dataset.from_dict({
    "question": [...],
    "answer": [...],
})

# Create environment
prime_env = SingleTurnEnv(
    dataset=dataset,
    rubric=YourCustomRubric(),
    parser=YourCustomParser(),
)

# Use with rollouts
reward_fn = prime_reward_fn(prime_env)
```

## Comparison with Clicker (Gemini) Evaluation

### Similarities

1. **Config pattern**: Both use frozen dataclasses for configuration
2. **Entry point**: Both have a `local.py`/`run_eval.py` wrapper
3. **Explicit settings**: All parameters explicit (Tiger Style)
4. **Output structure**: Both save to `results/` directory

### Differences

| Clicker | Integration Eval |
|---------|------------------|
| Uses `run_eval.py` from rollouts | Custom `local.py` for Prime |
| ScreenSpot environment | Prime verifiers environments |
| Direct reward computation | Uses Prime rubrics |
| Config has `to_trajectory` loader | Uses Prime dataset converter |

### What I Learned from Clicker

1. ✅ **Config groups are good**: ModelConfig, DatasetConfig, FilterConfig, OutputConfig
2. ✅ **Explicit validation**: `__post_init__` with assertions
3. ✅ **Conversion methods**: `to_endpoint()`, `to_eval_config()` are clean
4. ✅ **Frozen dataclasses**: Immutability prevents bugs
5. ✅ **Thin wrapper pattern**: `run_eval.py` just loads .env and delegates

## Next Steps

### Phase 2: Training Integration

Once evaluation works, integrate Prime rewards into RL training:

```python
# In training/loops/rl_loop.py
def compute_reward(sample: Sample, reward_fn: RewardFunction) -> float:
    # Extract trajectory from sample
    trajectory = sample.metadata["trajectory"]

    # Apply Prime reward function
    scored_traj = reward_fn(trajectory)

    return scored_traj.rewards
```

### Using Other Models

Edit config to use different providers:

```python
# OpenAI
model_name: str = "gpt-4"
provider: str = "openai"
api_base: str = "https://api.openai.com/v1"
api_key_env_var: str = "OPENAI_API_KEY"

# Anthropic
model_name: str = "claude-3-5-sonnet-20241022"
provider: str = "anthropic"
api_base: str = "https://api.anthropic.com"
api_key_env_var: str = "ANTHROPIC_API_KEY"

# Gemini (via OpenAI-compatible endpoint)
model_name: str = "gemini-2.5-flash"
provider: str = "openai"
api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key_env_var: str = "GEMINI_API_KEY"
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'hud_text_2048'`

The Prime Hub environment package isn't installed. Either:
1. Use `simple_2048_env.py` (our test environment)
2. Install the package if available: `uv pip install hud-text-2048`
3. Create your own environment matching the Prime interface

### `Connection refused` to sglang

Start sglang server:
```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --port 30000
```

Or use a different provider (OpenAI, Anthropic, etc.) in the config.

### Results showing 0.0 rewards

Check:
1. Model is generating valid responses
2. Parser is extracting answers correctly
3. Rubric is grading correctly

Inspect sample results:
```python
import json
result = json.load(open("results/integration-evaluation/samples/sample_0000.json"))
print(result["trajectory"]["metadata"])
```

## Example Output

```
📝 Loading config from: config/prime_2048.py
🎯 Configuration loaded
   Model: Qwen/Qwen2.5-7B-Instruct
   Environment: simple_2048
   Samples: 10
   Max concurrent: 4

🎮 Creating Prime environment: simple_2048
   Dataset size: 10
   Rubric: Simple2048Rubric
   Parser: Simple2048Parser

🏆 Creating reward function from Prime rubric

📊 Converting dataset to rollouts format
   Converted 10 samples

🚀 Starting evaluation
   Output dir: results/integration-evaluation
==================================================

🎯 Starting evaluation: prime_2048_eval
📊 Samples to evaluate: 10
🔧 Max concurrent: 4
==================================================
📝 Evaluating sample_0000
   reward=1.000
📝 Evaluating sample_0001
   reward=1.000
...

==================================================
📊 Evaluation Summary: prime_2048_eval
==================================================
Samples evaluated: 10
mean_reward: 0.850
min_reward: 0.000
max_reward: 1.000
std_reward: 0.234

✅ Results saved to: results/integration-evaluation
```

## Summary

This directory demonstrates **Phase 1** of Prime Intellect integration:
- ✅ Load Prime environments
- ✅ Use Prime rubrics for scoring
- ✅ Evaluate models with rollouts framework
- ✅ Save detailed results

**Next**: Phase 2 will integrate Prime rewards into RL training loops.
