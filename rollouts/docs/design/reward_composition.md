# Reward Composition Design

> Pure functions, explicit state, composable rewards.

## Current State

`EvalConfig` takes a single `RewardFunction`:

```python
RewardFunction = Callable[[Trajectory], Trajectory] | Callable[[Trajectory], Awaitable[Trajectory]]

config = EvalConfig(
    reward_fn=my_reward,
    ...
)
```

This works but doesn't support:
- Weighted combination of multiple rewards
- LLM-as-judge with caching
- Advantage computation for RL
- Structured reward breakdowns

## Design Principles

From `code_style/FAVORITES.md`:
- **Functions over classes** when there's no meaningful state
- **Frozen dataclasses for config** - immutable, serializable
- **Explicit state threading** - cache in, cache out
- **Single assignment** - name each transformation

## Proposed Design

### Core Types

```python
from dataclasses import dataclass
from typing import Callable, Awaitable, Mapping
from rollouts.dtypes import Trajectory

# Reward functions take trajectory, return float
RewardFn = Callable[[Trajectory], float] | Callable[[Trajectory], Awaitable[float]]

@dataclass(frozen=True)
class JudgeConfig:
    """LLM judge configuration - immutable."""
    endpoint: Endpoint
    rubric: str

@dataclass(frozen=True)
class WeightedRewardConfig:
    """Weighted reward combination - immutable."""
    weights: Mapping[str, float]  # name -> weight

    def __post_init__(self):
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"
```

### Pure Functions

#### Cache Management

```python
def cache_key(trajectory: Trajectory) -> str:
    """Generate cache key from trajectory content. Pure function."""
    content = json.dumps([
        {"role": m.role, "content": m.content}
        for m in trajectory.messages
    ], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:32]

def load_judge_cache(cache_path: Path) -> dict[str, float]:
    """Load cache from disk. Pure I/O function."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}

def save_judge_cache(cache: dict[str, float], cache_path: Path) -> None:
    """Save cache to disk. Pure I/O function."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
```

#### LLM Judge

```python
async def score_with_llm_judge(
    trajectory: Trajectory,
    config: JudgeConfig,
    cache: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Score trajectory with LLM judge.

    Pure function: explicit cache in, updated cache out.
    Returns: (score, updated_cache)
    """
    key = cache_key(trajectory)

    if key in cache:
        return cache[key], cache

    # Build prompt and call judge
    score = await call_llm_judge(config, trajectory)

    # Return score and updated cache (immutable update)
    new_cache = {**cache, key: score}
    return score, new_cache
```

#### Simple Reward Functions

```python
def length_penalty_reward(trajectory: Trajectory) -> float:
    """Reward penalizing long responses. Pure function."""
    total_tokens = sum(c.usage.completion_tokens for c in trajectory.completions)

    if total_tokens <= 500:
        return 1.0
    elif total_tokens >= 2000:
        return 0.0
    else:
        return 1.0 - (total_tokens - 500) / 1500

def format_compliance_reward(trajectory: Trajectory) -> float:
    """Reward for following expected format. Pure function."""
    # Check for code blocks, step markers, etc.
    ...
```

#### Weighted Composition

```python
async def compute_weighted_reward(
    trajectory: Trajectory,
    reward_fns: dict[str, RewardFn],
    config: WeightedRewardConfig,
) -> tuple[float, dict[str, float]]:
    """Compute weighted reward from multiple functions.

    Returns: (total_reward, component_scores)
    """
    assert set(reward_fns.keys()) == set(config.weights.keys())

    scores: dict[str, float] = {}
    for name, fn in reward_fns.items():
        if inspect.iscoroutinefunction(fn):
            scores[name] = await fn(trajectory)
        else:
            scores[name] = fn(trajectory)

    total = sum(scores[name] * config.weights[name] for name in scores)
    return total, scores
```

#### Advantage Computation

```python
def compute_advantages(
    rewards: list[float],
    baseline: float | None = None,
    normalize: bool = True,
) -> list[float]:
    """Compute advantages from rewards. Pure function."""
    if not rewards:
        return []

    if baseline is None:
        baseline = statistics.mean(rewards)

    advantages = [r - baseline for r in rewards]

    if normalize and len(advantages) > 1:
        std = statistics.stdev(advantages)
        if std > 1e-8:
            advantages = [a / std for a in advantages]

    return advantages
```

#### Batch Scoring

```python
async def score_batch(
    trajectories: list[Trajectory],
    judge_config: JudgeConfig,
    weights_config: WeightedRewardConfig,
    cache: dict[str, float],
    baseline: float | None = None,
) -> tuple[list[Trajectory], dict[str, float]]:
    """Score batch and compute advantages.

    Returns: (scored_trajectories_with_advantages, updated_cache)
    """
    scored: list[Trajectory] = []

    for traj in trajectories:
        # Get judge score (with cache threading)
        judge_score, cache = await score_with_llm_judge(traj, judge_config, cache)

        # Build reward functions
        reward_fns = {
            "llm_judge": lambda _: judge_score,
            "length_penalty": length_penalty_reward,
            "format_compliance": format_compliance_reward,
        }

        # Compute weighted total
        total, components = await compute_weighted_reward(traj, reward_fns, weights_config)

        # Update trajectory
        scored.append(replace(
            traj,
            rewards=total,
            metadata={**traj.metadata, "reward_components": components},
        ))

    # Compute advantages
    rewards = [t.rewards for t in scored]
    advantages = compute_advantages(rewards, baseline, normalize=True)

    # Attach advantages
    result = [replace(t, advantages=adv) for t, adv in zip(scored, advantages)]

    return result, cache
```

## Example Usage

```python
async def main():
    # Configuration (frozen)
    judge_config = JudgeConfig(
        endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-20250514", ...),
        rubric="Score correctness, clarity, completeness (0-1).",
    )

    weights_config = WeightedRewardConfig(
        weights={
            "llm_judge": 0.6,
            "length_penalty": 0.2,
            "format_compliance": 0.2,
        }
    )

    # Load cache (explicit I/O at boundary)
    cache_path = Path(".cache/llm_judge/cache.json")
    cache = load_judge_cache(cache_path)

    # Score batch
    scored, cache = await score_batch(
        trajectories,
        judge_config,
        weights_config,
        cache,
    )

    # Save cache (explicit I/O at boundary)
    save_judge_cache(cache, cache_path)

    return scored
```

## Integration with EvalConfig

```python
# Wrap score_batch for use as RewardFunction
def make_reward_fn(
    judge_config: JudgeConfig,
    weights_config: WeightedRewardConfig,
    cache_path: Path,
) -> RewardFunction:
    """Create a reward function that uses the composition framework."""
    cache = load_judge_cache(cache_path)

    async def reward_fn(trajectory: Trajectory) -> Trajectory:
        nonlocal cache

        # Score single trajectory
        scored, cache = await score_batch(
            [trajectory],
            judge_config,
            weights_config,
            cache,
        )

        # Save cache after each call (or batch)
        save_judge_cache(cache, cache_path)

        return scored[0]

    return reward_fn

# Usage
config = EvalConfig(
    reward_fn=make_reward_fn(judge_config, weights_config, cache_path),
    ...
)
```

## File Structure

```
rollouts/rollouts/rewards/
├── __init__.py           # Re-exports
├── types.py              # RewardFn, JudgeConfig, WeightedRewardConfig
├── cache.py              # cache_key, load_judge_cache, save_judge_cache
├── judge.py              # score_with_llm_judge, call_llm_judge
├── functions.py          # length_penalty_reward, format_compliance_reward, ...
├── composition.py        # compute_weighted_reward
├── advantages.py         # compute_advantages
└── batch.py              # score_batch, score_trajectory
```

## Why Not a Class?

From the discussion with Claude:

> "The cache threading pattern is particularly clean:
> `scored_traj, cache = await score_trajectory(traj, ..., cache)`
> cache is explicitly updated, no hidden mutation"

A `Rubric` class would hide the cache inside instance state. The functional approach:
- Makes cache flow explicit (no hidden state)
- Makes testing trivial (pure functions)
- Composes with `dict[str, RewardFn]` instead of inheritance
- Keeps configs immutable and validated

## Tests

```python
def test_weighted_reward_sums_correctly():
    config = WeightedRewardConfig(weights={"a": 0.5, "b": 0.5})
    fns = {"a": lambda _: 1.0, "b": lambda _: 0.0}
    total, _ = compute_weighted_reward(trajectory, fns, config)
    assert total == 0.5

def test_weights_must_sum_to_one():
    with pytest.raises(AssertionError):
        WeightedRewardConfig(weights={"a": 0.3, "b": 0.3})

def test_judge_cache_hit():
    cache = {"abc123": 0.8}
    score, new_cache = await score_with_llm_judge(traj, config, cache)
    assert score == 0.8
    assert new_cache is cache  # No mutation

def test_advantages_normalized():
    rewards = [1.0, 2.0, 3.0]
    advs = compute_advantages(rewards, normalize=True)
    assert abs(sum(advs)) < 1e-6  # Mean-centered
    assert abs(statistics.stdev(advs) - 1.0) < 1e-6  # Unit variance
```

## Migration

1. Create `rollouts/rollouts/rewards/` with the modules above
2. Add tests in `rollouts/tests/test_rewards.py`
3. Update `EvalConfig` docs to show composition pattern
4. Deprecate nothing - this is additive
