# Evaluation Abstractions

**DRI:** Chiraag
**Claude:** [this conversation]

## Context
Make it easy/clean to run evaluations with composable scoring, sliceable results, and compatibility with RL training loops.

## Non-Goals
- Distributed evaluation (single-machine only for now)
- Custom aggregation functions (use default mean/std/min/max)
- Evaluation registry/hub (just local files)

## Solution
**Input:** Dataset rows, score function, environment factory
**Output:** Scored trajectories with metrics breakdown, sliceable by metadata

## Usage

### Perspective 1: Researcher Running Experiments

```python
# experiments/evals/math_500/baseline_01_01.py
"""Baseline eval on MATH-500. 100 samples, single-turn."""

from dataclasses import replace
from evals import EvalConfig, Sample, Score, Metric, evaluate
from evals.loaders import load_jsonl

# ── Config ──────────────────────────────────────────────────────────
from configs.evals.math_base import config as base_config

config = replace(
    base_config,
    max_samples=100,
    max_concurrent=10,
    experiment_name="math_500_baseline",
)

# ── Dataset ─────────────────────────────────────────────────────────
def load_dataset() -> list[Sample]:
    return [
        Sample(
            id=str(i),
            input=row,
            ground_truth=row["answer"],
            metadata={"difficulty": row["level"]},
        )
        for i, row in enumerate(load_jsonl("data/math_500.jsonl"))
    ]

# ── Scorer ──────────────────────────────────────────────────────────
def score_fn(trajectory: Trajectory, sample: Sample) -> Score:
    """Extract answer, check correctness."""
    response = trajectory.messages[-1].content
    extracted = extract_boxed_answer(response)
    correct = normalized_equal(extracted, sample.ground_truth)

    return Score(metrics=(
        # Reward (weight > 0)
        Metric("correct", 1.0 if correct else 0.0, weight=1.0),
        # Tracking only (weight = 0)
        Metric("num_turns", len(trajectory.messages), weight=0),
    ))

# ── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--gpu-id", type=str)
    args = parser.parse_args()

    if args.remote or args.gpu_id:
        run_remote(__file__, gpu_id=args.gpu_id)
    else:
        import trio

        report = trio.run(
            evaluate,
            load_dataset(),
            config,
            score_fn,
        )

        report.save(f"outputs/evals/{config.experiment_name}")

        print(f"Score: {report.summary['mean']:.2%}")
        assert report.summary["mean"] > 0.5, "Baseline should be >50%"
```

### Perspective 2: Researcher Slicing/Dicing Results

```python
# scripts/analyze_math_evals.py
"""Slice and dice evaluation results post-hoc."""

from evals import EvalReport
from evals.analysis import group_by, summarize
from pathlib import Path

# ── Load Results ────────────────────────────────────────────────────
runs = [EvalReport.load(p) for p in Path("outputs/evals").glob("math_500_*")]

# ── Slice by Difficulty ─────────────────────────────────────────────
for run in runs:
    print(f"\n=== {run.config['experiment_name']} ===")
    print(f"Overall: {run.summary['mean']:.2%}")

    by_difficulty = group_by(run.results, key=lambda r: r.metadata["difficulty"])
    for level, results in sorted(by_difficulty.items()):
        stats = summarize(results)
        print(f"  Level {level}: {stats['mean']:.2%} (n={stats['n']})")

# ── Inspect Failures ────────────────────────────────────────────────
latest = runs[-1]
failures = [r for r in latest.results if r.score.reward < 0.5]
print(f"\n=== Failed Samples ({len(failures)}) ===")
for r in failures[:5]:
    print(f"  {r.sample_id}: reward={r.score.reward:.2f}")
    for m in r.score.metrics:
        print(f"    {m.name}: {m.value}" + (f" (w={m.weight})" if m.weight > 0 else ""))
```

### Perspective 3: RL Training Loop Integration

```python
# rollouts/training/loops/rl_loop.py
"""RL training loop using eval types for scoring.

Compatible with Miles DataBuffer pattern.
Pure function orchestrates stateful objects.
"""

from rollouts.training.datasets import DataBuffer
from rollouts.training.backends import PyTorchTrainingBackend
from evals import Sample, Score, Metric, Trajectory

async def run_rl_training(
    backend: PyTorchTrainingBackend,      # Stateful: owns model/optimizer
    data_buffer: DataBuffer,              # Stateful: owns iteration
    score_fn: Callable[[Trajectory, Sample], Score],  # Pure
    config: RLTrainingConfig,             # Frozen
) -> list[dict]:
    """Pure function orchestrating stateful objects."""
    metrics_history = []

    for step in range(config.num_steps):
        # Get prompts from buffer (stateful)
        prompt_batch = data_buffer.get_prompts(config.batch_size)

        # Generate rollouts
        samples = [
            Sample(id=f"step_{step}_{i}", input={"prompt": p}, ground_truth=None)
            for i, p in enumerate(prompt_batch)
        ]

        trajectories = await generate_rollouts(
            samples=samples,
            endpoint=config.endpoint,
            environment_factory=config.environment_factory,
        )

        # Score rollouts (pure)
        scores = [score_fn(traj, sample) for traj, sample in zip(trajectories, samples)]

        # Compute advantages from reward signal
        rewards = [s.reward for s in scores]
        advantages = compute_grpo_advantages(rewards)

        # Log all metrics (rewards + tracking-only)
        for score in scores:
            for m in score.metrics:
                log_metric(f"score/{m.name}", m.value, step=step)

        # Update weights (stateful)
        loss = backend.train_step(trajectories, advantages)

        metrics_history.append({
            "step": step,
            "loss": loss,
            "mean_reward": sum(rewards) / len(rewards),
        })

    return metrics_history
```

---

## Details

### Core Types

```python
@dataclass(frozen=True)
class Metric:
    """A measured dimension. Weight=0 means track-only, weight>0 means contributes to reward."""
    name: str
    value: float
    weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Score:
    """Collection of metrics, some of which are rewards (weight > 0)."""
    metrics: tuple[Metric, ...]

    @property
    def reward(self) -> float:
        """Weighted average of metrics with weight > 0."""
        weighted = [(m.value, m.weight) for m in self.metrics if m.weight > 0]
        if not weighted:
            return 0.0
        total_weight = sum(w for _, w in weighted)
        return sum(v * w for v, w in weighted) / total_weight

@dataclass(frozen=True)
class Sample:
    """Single evaluation unit."""
    id: str
    input: dict[str, Any]
    ground_truth: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class EvalResult:
    """Single sample evaluation output."""
    sample_id: str
    sample: Sample
    trajectory: Trajectory
    score: Score
    metadata: dict[str, Any] = field(default_factory=dict)  # execution metadata

@dataclass(frozen=True)
class EvalReport:
    """Aggregated evaluation summary."""
    results: tuple[EvalResult, ...]
    summary: dict[str, float]
    config: dict[str, Any]

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> 'EvalReport': ...
```

### Function Signatures

```python
# Transform dataset row -> initial messages
PrepareMessages = Callable[[Sample], list[Message]]

# Score a completed trajectory
ScoreFn = Callable[[Trajectory, Sample], Score]

# Core evaluation function
async def evaluate(
    samples: Iterable[Sample],
    config: EvalConfig,
    score_fn: ScoreFn,
    prepare_messages: PrepareMessages | None = None,
    environment_factory: Callable[[Sample], Environment] | None = None,
) -> EvalReport: ...
```

### Flow

1. Load dataset -> `list[Sample]`
2. For each sample (concurrent up to `max_concurrent`):
   a. `messages = prepare_messages(sample)`
   b. `env = environment_factory(sample)` if provided
   c. `trajectory = await run_agent(messages, env, endpoint)`
   d. `score = score_fn(trajectory, sample)`
   e. Yield `EvalResult`
3. Aggregate results -> `EvalReport`
4. Save to disk (config + results for reproducibility)

### Environment Serialization Contract

Environments must include `env_kind` and `version` in serialized state for safe restore:

```python
@runtime_checkable
class Environment(Protocol):
    async def serialize(self) -> dict:
        """Serialize environment state.

        Must include:
          - env_kind: str (e.g., "calculator", "code_exec")
          - version: str (e.g., "1.0.0")
          - ...rest of state
        """
        ...

    @staticmethod
    async def deserialize(data: dict) -> 'Environment':
        """Deserialize environment from dict.

        Should validate env_kind and version before restoring.
        """
        ...
```

Implementation pattern:

```python
class CalculatorEnvironment:
    ENV_KIND = "calculator"
    VERSION = "1.0.0"

    async def serialize(self) -> dict:
        return {
            "env_kind": self.ENV_KIND,
            "version": self.VERSION,
            "history": self._history,
        }

    @staticmethod
    async def deserialize(data: dict) -> 'CalculatorEnvironment':
        assert data["env_kind"] == CalculatorEnvironment.ENV_KIND
        assert data["version"].startswith("1.")  # compatible versions
        env = CalculatorEnvironment()
        env._history = data["history"]
        return env
```

This prevents silent corruption when restoring snapshots into wrong environment types or incompatible versions.

### Open Questions

- [ ] Should `Metric.value` be constrained to 0-1 or allow arbitrary floats?
- [ ] Do we need async score functions for external verifiers (e.g., Prime)?
- [ ] How to handle environment cleanup on failure?

### Files

**Read:**
- `rollouts/dtypes.py` - existing Trajectory, EvalConfig types
- `rollouts/training/datasets/data_buffer.py` - DataBuffer pattern

**Modify:**
- `rollouts/dtypes.py` - add Metric, Score, Sample, EvalResult, EvalReport
- `rollouts/evaluation.py` - implement evaluate() function
- `rollouts/evals/__init__.py` - new module exports

## References

- [OpenAI simple-evals](https://github.com/openai/simple-evals) - lightweight eval framework
- [Prime Verifiers](https://github.com/PrimeIntellect-ai/verifiers) - environment + rubric patterns
- [Miles RL](https://github.com/radixark/miles) - DataBuffer, async rollout generation
- [Synth AI Workflows](https://docs.usesynth.ai/cookbooks/workflows/overview) - judge-based evaluation
- [K2 Vendor Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) - tool-calling metrics (F1, schema accuracy)
- [Ludic](https://github.com/hallerite/ludic) - Snapshot pattern for env checkpointing with env_kind/version
