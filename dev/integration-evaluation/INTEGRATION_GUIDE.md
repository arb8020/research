# Prime → Rollouts Integration Guide

## Overview

This document explains how Prime Intellect environments are integrated with the rollouts evaluation framework, the type mismatches, and our adaptation strategy.

## Prime's Evaluation Flow

### 1. **Prime Environment Structure**
```python
env = load_environment('wiki-search')
# Returns: ToolEnv with:
#   - dataset: List of dicts with {question, answer, prompt, ...}
#   - rubric: RubricGroup with [ToolRubric, JudgeRubric]
#   - parser: Parser to extract answers from responses
#   - max_turns: 10 (multi-turn support)
```

### 2. **Prime's Scoring Mechanism**
```python
# Prime uses async scoring:
score: RolloutScore = await rubric.score_rollout(
    prompt=[{role: "system", content: ...}, {role: "user", content: ...}],
    completion=[{role: "assistant", content: ...}, ...],
    answer="ground truth",
    state={},  # Tool execution state
    info=sample_data
)

# Returns:
# RolloutScore(
#     reward: float,  # 0.0 to 1.0
#     metrics: dict[str, float]  # Breakdown by rubric
# )
```

### 3. **RubricGroup Components (wiki-search)**
- **ToolRubric**: Counts/validates tool usage
- **JudgeRubric**: LLM-based answer comparison (uses GPT-4 to judge)

## Rollouts Evaluation Flow

### 1. **Rollouts Structure**
```python
# Trajectory representation
Trajectory(
    messages=[
        Message(role="system", content="..."),
        Message(role="user", content="..."),
        Message(role="assistant", content="..."),
    ],
    metadata={"sample_data": {...}},  # Original dataset sample
    rewards=0.0  # To be computed
)
```

### 2. **Reward Function Signature**
```python
# Rollouts uses SYNC reward functions
def reward_fn(trajectory: Trajectory) -> Trajectory:
    # Extract answer, compare to ground truth
    # Return trajectory with rewards populated
    return replace(trajectory, rewards=score)
```

## The Integration Gap

### Type Mismatches

| Aspect | Prime | Rollouts | Issue |
|--------|-------|----------|-------|
| **Async** | `async def score_rollout()` | `def reward_fn()` | Prime uses asyncio, rollouts uses trio |
| **Message Format** | OpenAI ChatCompletion dicts | `Message(role, content)` dataclass | Need conversion |
| **Input** | `(prompt, completion, answer, state)` | `Trajectory` | Different interfaces |
| **Output** | `RolloutScore(reward, metrics)` | `Trajectory` with rewards | Different return types |
| **Execution** | Prime handles generation | Rollouts handles generation | Prime's evaluate() not used |

### Data Flow Comparison

**Prime's Full Flow:**
```
Dataset → env.evaluate() → generate rollouts → score_rollout() → RolloutScore
```

**Rollouts' Flow:**
```
Dataset → prepare_messages() → run_agent() → Trajectory → reward_fn() → Trajectory with rewards
```

**Our Hybrid:**
```
Prime Dataset → prepare_messages() → run_agent() → Trajectory → prime_reward_fn() → ???
                                                                      ↑
                                                    Need to call Prime's scorer here
```

## Solution Options

### Option 1: Simple String Matching (Current)
```python
def prime_reward_fn(verifiers_env):
    def reward(trajectory: Trajectory) -> Trajectory:
        # Extract answer using Prime's parser
        parsed = parser.parse(last_assistant_message)
        ground_truth = trajectory.metadata["sample_data"]["answer"]

        # Simple comparison (no async, works with trio)
        if parsed.lower() == ground_truth.lower():
            score = 1.0
        elif parsed in ground_truth or ground_truth in parsed:
            score = 0.5
        else:
            score = 0.0

        return replace(trajectory, rewards=score)
    return reward
```

**Pros:** Simple, sync, works with trio
**Cons:** Doesn't use Prime's JudgeRubric or ToolRubric

### Option 2: Make Reward Function Async (Recommended)
```python
# Change RewardFunction type to support async
RewardFunction = Callable[[Trajectory], Awaitable[Trajectory]] | Callable[[Trajectory], Trajectory]

async def prime_reward_fn(verifiers_env):
    async def reward(trajectory: Trajectory) -> Trajectory:
        # Convert trajectory to Prime format
        prompt, completion = convert_trajectory_to_prime_format(trajectory)
        ground_truth = trajectory.metadata["sample_data"]["answer"]

        # Call Prime's scorer with trio-asyncio
        import trio_asyncio
        async with trio_asyncio.open_loop():
            score_result = await rubric.score_rollout(
                prompt=prompt,
                completion=completion,
                answer=ground_truth,
                state={},
                info=trajectory.metadata["sample_data"]
            )

        return replace(trajectory, rewards=score_result.reward)
    return reward
```

**Pros:** Uses Prime's full scoring (ToolRubric + JudgeRubric)
**Cons:** Requires trio-asyncio, makes evaluation.py more complex

### Option 3: Sync Wrapper with trio-asyncio
```python
def prime_reward_fn(verifiers_env):
    def reward(trajectory: Trajectory) -> Trajectory:
        import trio_asyncio

        async def async_score():
            # Convert and score
            prompt, completion = convert_messages(trajectory)
            return await rubric.score_rollout(...)

        # Run asyncio code from trio context
        score_result = trio_asyncio.aio_as_trio(async_score)()
        return replace(trajectory, rewards=score_result.reward)
    return reward
```

**Pros:** Keeps reward_fn sync, uses Prime's scoring
**Cons:** trio-asyncio adds complexity

## Recommended Approach

For now, use **Option 1** (simple string matching) because:
1. It works immediately with trio
2. No async complexity
3. Good enough for initial integration testing
4. Can upgrade to Option 2/3 later if needed

Later, implement **Option 2** to get full Prime scoring fidelity.

## Key Adaptation Points

1. **Dataset Format**: Prime samples have `{question, answer, prompt}` - use `prepare_messages()` to convert
2. **Parser**: Use Prime's parser to extract answers from model responses
3. **Scoring**: Start simple (string match), upgrade to full Prime scoring later
4. **Tools**: wiki-search has tools but we're not exposing them yet (WikiSearchEnvironment returns empty tools)

## Testing the Integration

```python
# Test simple scoring
from verifiers import load_environment
from rollouts.integrations.prime import prime_reward_fn

env = load_environment('wiki-search')
reward_fn = prime_reward_fn(env)

# Should work synchronously with trio
trajectory = Trajectory(
    messages=[...],
    metadata={"sample_data": {"answer": "ground truth"}}
)
scored = reward_fn(trajectory)
print(scored.rewards)  # 0.0 to 1.0
```
