# Prime Intellect Integration

This document explains Phase 1 of Prime Intellect integration: **Evaluation with verifiers environments**.

## Overview

We've integrated Prime Intellect's `verifiers` library with the rollouts evaluation framework. This allows you to:

1. ✅ Use Prime environments for evaluation
2. ✅ Leverage Prime rubrics (reward functions) for scoring
3. ✅ Compose Prime rewards with custom signals
4. ✅ Keep your existing rollouts infrastructure unchanged

**Phase 2 (Training)** will come later - using Prime rewards in RL training loops.

---

## Architecture

### **Key Components**

1. **`rollouts/integrations/prime.py`** - Adapter between verifiers and rollouts
2. **`rollouts/evaluation.py`** - Clean evaluation framework with `EvalConfig`
3. **`rollouts/dtypes.py`** - Added `RewardFunction` type and `EvalConfig`

### **How It Works**

```
Prime Environment (verifiers)
    ↓
prime_reward_fn() creates adapter
    ↓
RewardFunction: Trajectory -> Trajectory
    ↓
EvalConfig uses reward_fn
    ↓
evaluate() runs agent and computes rewards
```

---

## Installation

```bash
# Already installed in your environment
uv pip install verifiers
```

---

## Usage Examples

### **Example 1: Simple Prime Evaluation**

```python
from verifiers import SingleTurnEnv, Rubric, Parser
from datasets import Dataset

from rollouts.evaluation import evaluate, EvalConfig
from rollouts.dtypes import Endpoint
from rollouts.integrations.prime import (
    prime_reward_fn,
    convert_verifiers_dataset_to_rollouts,
    prepare_messages_from_verifiers,
)
from rollouts.environments.calculator import CalculatorEnvironment

# 1. Create Prime environment
prime_dataset = Dataset.from_dict({
    "question": ["What is 2 + 2?", "What is 10 * 5?"],
    "answer": ["4", "50"],
})

prime_env = SingleTurnEnv(
    dataset=prime_dataset,
    rubric=Rubric(),  # Prime's rubric
    parser=Parser(),
)

# 2. Create reward function from Prime
reward_fn = prime_reward_fn(prime_env)

# 3. Setup evaluation config
config = EvalConfig(
    reward_fn=reward_fn,
    max_turns=10,
    max_concurrent=4,
    eval_name="prime_eval",
)

# 4. Run evaluation
endpoint = Endpoint(provider="sglang", model="Qwen/Qwen2.5-7B-Instruct", ...)
rollouts_dataset = convert_verifiers_dataset_to_rollouts(prime_env)

report = await evaluate(
    dataset=iter(rollouts_dataset),
    prepare_messages=prepare_messages_from_verifiers,
    environment_factory=lambda: CalculatorEnvironment(),
    endpoint=endpoint,
    config=config,
    dataset_path="prime_dataset",
)

print(f"Mean reward: {report.summary_metrics['mean_reward']}")
```

### **Example 2: Composite Rewards (Prime + Custom)**

```python
from dataclasses import replace
from rollouts.integrations.prime import create_composite_prime_reward

# Define custom signal
def efficiency_signal(trajectory):
    num_turns = len([m for m in trajectory.messages if m.role == "assistant"])
    score = max(0.0, 1.0 - (num_turns - 3) * 0.1)
    return replace(trajectory, rewards=score)

# Compose Prime rubric with efficiency
reward_fn = create_composite_prime_reward(
    verifiers_env=prime_env,
    additional_signals={
        "efficiency": (efficiency_signal, 0.2),  # 20% weight
    }
)

config = EvalConfig(reward_fn=reward_fn, ...)
```

### **Example 3: Manual Composition (Your Philosophy)**

```python
# Get Prime reward function
prime_fn = prime_reward_fn(prime_env)

# User writes their own composition logic
def my_custom_reward(trajectory):
    # Get Prime score
    prime_traj = prime_fn(trajectory)
    prime_score = prime_traj.rewards

    # Get efficiency
    eff_score = efficiency_signal(trajectory).rewards

    # Custom logic
    if prime_score == 0.0:
        total = 0.0  # Don't reward efficiency if wrong
    else:
        total = prime_score * 1.0 + eff_score * 0.3

    # Store custom metadata
    metadata = {
        **prime_traj.metadata,
        "my_breakdown": {"correct": prime_score, "eff": eff_score}
    }

    return replace(trajectory, rewards=total, metadata=metadata)

config = EvalConfig(reward_fn=my_custom_reward, ...)
```

---

## API Reference

### **`prime_reward_fn(verifiers_env, rubric=None, parser=None, ground_truth_key="answer")`**

Creates a `RewardFunction` from a Prime environment.

**Args:**
- `verifiers_env`: Prime `Environment` instance
- `rubric`: Optional rubric override
- `parser`: Optional parser override
- `ground_truth_key`: Key in sample_data for ground truth

**Returns:**
- `RewardFunction` that scores trajectories using Prime's rubric

**Metadata Added:**
- `prime_parsed_answer`: Answer extracted by parser
- `prime_ground_truth`: Ground truth from dataset
- `prime_raw_response`: Raw model response
- `prime_parse_error`: Error if parsing failed
- `prime_grade_error`: Error if grading failed

### **`create_composite_prime_reward(verifiers_env, additional_signals=None, ...)`**

Creates a composite reward combining Prime with additional signals.

**Args:**
- `verifiers_env`: Prime environment
- `additional_signals`: Dict of `{name: (reward_fn, weight)}`

**Returns:**
- Composite `RewardFunction`

**Metadata Added:**
- `reward_breakdown`: Dict with all component scores

### **`convert_verifiers_dataset_to_rollouts(verifiers_env, use_eval_dataset=False)`**

Converts Prime dataset to rollouts format.

**Returns:**
- List of sample dicts with `question` and `answer` keys

### **`prepare_messages_from_verifiers(sample_data)`**

Prepares initial messages from Prime sample format.

**Returns:**
- List of `Message` objects for agent initialization

---

## Integration Points

### **What Works Now (Phase 1 - Evaluation)**

✅ Load Prime environments
✅ Use Prime rubrics for scoring
✅ Compose Prime rewards with custom signals
✅ Run parallel evaluation with rollouts framework
✅ Full evaluation reports with Prime metadata

### **What's Next (Phase 2 - Training)**

🔲 Use Prime rewards in RL training loop
🔲 Integrate `training/loops/rl_loop.py` with Prime rewards
🔲 Dataset loaders for Prime Hub environments
🔲 Weight sync with Prime RL infrastructure

---

## Files Created

```
rollouts/
├── dtypes.py                              # Added RewardFunction, EvalConfig
├── evaluation.py                          # NEW: Clean evaluation framework
├── integrations/
│   ├── __init__.py                        # NEW
│   └── prime.py                           # NEW: Prime adapter
└── examples/
    ├── evaluation_example.py              # NEW: Basic eval examples
    └── prime_integration_example.py       # NEW: Prime integration examples
```

---

## Design Decisions

### **1. Reward Function Type: `Trajectory -> Trajectory`**

We chose this over `Trajectory -> float` because:
- Trajectory already has a `rewards` field - function populates it
- User controls metadata (breakdown, intermediate scores)
- Pure functional transform - immutable, composable
- Consistent with data model

### **2. No `RewardComponent` / `CompositeReward` Classes**

Users compose rewards manually in plain Python:
- Simpler API - no extra abstractions
- More flexible - any composition logic
- Follows "users can write Python" philosophy

### **3. EvalConfig Pattern**

Matches `RunConfig` style:
- `@dataclass(frozen=True)` - immutable
- Sensible defaults
- Single config parameter instead of 14 params

### **4. Ground Truth via Metadata**

Sample data injected into `trajectory.metadata["sample_data"]`:
- Reward functions can access ground truth
- No need for `(Trajectory, Dict) -> float` signature
- Cleaner interface

---

## Testing

Run the example:

```bash
python examples/prime_integration_example.py
```

This will:
1. Run simple Prime evaluation
2. Run composite evaluation (Prime + efficiency)
3. Run manual composition example

---

## Next Steps

**For Evaluation:**
- Try with actual Prime Hub environments (install via `uv run vf-install <env>`)
- Use more complex rubrics (JudgeRubric, custom rubrics)
- Test with multi-turn environments

**For Training (Phase 2):**
- Modify `training/loops/rl_loop.py` to accept `RewardFunction`
- Create dataset loaders for Prime Hub
- Test GRPO with Prime rewards
- Implement weight sync if using Prime RL infrastructure

---

## Example Output

```
🎯 Starting evaluation: prime_calculator_eval
📊 Samples to evaluate: 3
🔧 Max concurrent: 2
==================================================
📝 Evaluating sample_0000
   reward=1.000
📝 Evaluating sample_0001
   reward=1.000
📝 Evaluating sample_0002
   reward=0.000

==================================================
📊 Evaluation Summary: prime_calculator_eval
==================================================
Samples evaluated: 3
mean_reward: 0.667
min_reward: 0.000
max_reward: 1.000
```

---

## Troubleshooting

**Q: `ModuleNotFoundError: No module named 'wordle'`**
A: Prime Hub environments are separate packages. Install with:
```bash
uv run vf-install wordle --from-repo
```

**Q: How do I use custom Prime rubrics?**
A: Pass your rubric to `prime_reward_fn()`:
```python
from verifiers import JudgeRubric
custom_rubric = JudgeRubric(judge_model="gpt-4", ...)
reward_fn = prime_reward_fn(prime_env, rubric=custom_rubric)
```

**Q: Can I use this without Prime environments?**
A: Yes! You can create a `RewardFunction` manually:
```python
def my_reward(trajectory):
    score = check_correctness(trajectory)
    return replace(trajectory, rewards=score)

config = EvalConfig(reward_fn=my_reward, ...)
```

---

## Summary

**Phase 1 Complete!** ✅

You can now:
- Evaluate models against Prime environments
- Use Prime rubrics for scoring
- Compose Prime rewards with custom signals
- Keep your existing rollouts infrastructure

**Next**: Phase 2 will integrate Prime rewards into your RL training loop.
