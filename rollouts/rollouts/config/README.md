# Rollouts Configuration Module

Standardized configuration protocols and base implementations for evaluation projects.

## Design Philosophy

**Tiger Style + Casey Muratori Principles:**

1. **Protocols over Inheritance** - Define interfaces without coupling
2. **Composition over Inheritance** - Compose configs, don't inherit
3. **Granularity** - Multiple entry points (use, customize, extend, ignore)
4. **Redundancy** - Multiple ways to achieve the same goal
5. **Explicit over Implicit** - All defaults visible, no magic
6. **Immutable** - Frozen dataclasses, no hidden state

## Quick Start

### Key Insight: Environment Injection

**BaseEvaluationConfig contains BaseEnvironmentConfig!**

This creates clean separation:
- **EnvironmentConfig** = Domain-specific task setup (messages, tools, domain settings)
- **EvaluationConfig** = Generic execution settings (turns, samples, concurrency) + environment

```python
from rollouts.config import BaseModelConfig, BaseEnvironmentConfig, BaseEvaluationConfig
from dataclasses import dataclass, field

# Define your environment
@dataclass(frozen=True)
class MyEnvironmentConfig(BaseEnvironmentConfig):
    env_name: str = "my-task"

    def prepare_messages(self, sample_data):
        return [Message(role="user", content=sample_data["prompt"])]

# Compose into main config
@dataclass(frozen=True)
class MyConfig:
    model: BaseModelConfig = field(default_factory=BaseModelConfig)

    # Evaluation contains environment!
    evaluation: BaseEvaluationConfig = field(
        default_factory=lambda: BaseEvaluationConfig(
            environment=MyEnvironmentConfig(),  # ← Inject!
            max_turns=5,
            num_samples=100,
        )
    )

    # Optional: convenience accessor
    @property
    def environment(self):
        return self.evaluation.environment

config = MyConfig()
```

### Extend for Domain-Specific Logic

```python
from rollouts.config import BaseEnvironmentConfig
from rollouts.dtypes import Message

@dataclass(frozen=True)
class KernelEnvConfig(BaseEnvironmentConfig):
    env_name: str = "kernel-agent"
    ssh_target: str = "ubuntu@host:22"
    gpu_id: int = 0

    def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
        """Custom message preparation for kernel tasks."""
        return [
            Message(role="system", content="You are a kernel developer"),
            Message(role="user", content=sample_data["prompt"]),
        ]
```

### Satisfy Protocols Without Base Configs

```python
from rollouts.config import HasModelConfig
from rollouts.dtypes import Endpoint

@dataclass(frozen=True)
class MyModelConfig:
    """Custom model config - no inheritance needed."""
    model: str = "gpt-4"
    temp: float = 0.7

    def to_endpoint(self) -> Endpoint:
        return Endpoint(model=self.model, provider="openai", temperature=self.temp)

# Automatically satisfies HasModelConfig
assert isinstance(MyModelConfig(), HasModelConfig)
```

## Available Protocols

### HasModelConfig

```python
@runtime_checkable
class HasModelConfig(Protocol):
    def to_endpoint(self) -> Endpoint:
        """Convert to rollouts Endpoint for LLM API calls."""
        ...
```

**Satisfied by:** Any config with `to_endpoint()` method.

### HasEnvironmentConfig

```python
@runtime_checkable
class HasEnvironmentConfig(Protocol):
    env_name: str

    def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
        """Prepare initial messages from dataset sample."""
        ...
```

**Satisfied by:** Any config with `env_name` and `prepare_messages()`.

### HasEvaluationConfig

```python
@runtime_checkable
class HasEvaluationConfig(Protocol):
    def to_eval_config(self, score_fn: Callable) -> EvalConfig:
        """Convert to rollouts EvalConfig."""
        ...
```

**Satisfied by:** Any config with `to_eval_config()` method.

### HasOutputConfig

```python
@runtime_checkable
class HasOutputConfig(Protocol):
    output_dir: Path
    experiment_name: str
```

**Satisfied by:** Any config with `output_dir` and `experiment_name`.

## Available Base Configs

### BaseModelConfig

Standard model/endpoint configuration.

```python
@dataclass(frozen=True)
class BaseModelConfig:
    model_name: str = "gpt-4o-mini"
    provider: str = "openai"
    api_base: str = "https://api.openai.com/v1"
    api_key_env_var: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 3
    timeout: float = 120.0

    def to_endpoint(self) -> Endpoint: ...
```

### BaseEnvironmentConfig

Standard environment configuration.

```python
@dataclass(frozen=True)
class BaseEnvironmentConfig:
    env_name: str
    max_steps: int = 100
    timeout_seconds: float = 300.0

    def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
        """Override this in your subclass."""
        ...
```

### BaseEvaluationConfig

Standard evaluation configuration.

```python
@dataclass(frozen=True)
class BaseEvaluationConfig:
    eval_name: str = "evaluation"
    max_turns: int = 3
    num_samples: int = 100
    max_concurrent: int = 4
    output_dir: Path = Path("results")
    verbose: bool = True
    show_progress: bool = True

    def to_eval_config(self, reward_fn: Callable) -> EvalConfig: ...
```

### BaseOutputConfig

Standard output/logging configuration.

```python
@dataclass(frozen=True)
class BaseOutputConfig:
    output_dir: Path = Path("results")
    experiment_name: str = "experiment"
    verbose: bool = True
    log_level: str = "INFO"

    def save_config(self, config: Any, path: Path) -> None: ...
    @staticmethod
    def load_config(config_class: type, path: Path) -> Any: ...
```

## Utilities

### load_config_from_file

```python
from rollouts.config import load_config_from_file

# Load config from Python file
config = load_config_from_file("configs/01_baseline.py")

# Config file should export 'config' variable:
# config = MyConfig(param=value)
```

### validate_config_protocol

```python
from rollouts.config.loader import validate_config_protocol
from rollouts.config import HasModelConfig

errors = validate_config_protocol(my_config, HasModelConfig)
if errors:
    for err in errors:
        print(f"Config error: {err}")
    sys.exit(1)
```

## Complete Example

```python
# configs/01_baseline.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

from rollouts.config import (
    BaseModelConfig,
    BaseEnvironmentConfig,
    BaseEvaluationConfig,
)
from rollouts.dtypes import Message


@dataclass(frozen=True)
class MyEnvironmentConfig(BaseEnvironmentConfig):
    """Custom environment for my task."""
    env_name: str = "my-task"
    dataset_path: Path = Path("data/samples.json")

    def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
        return [
            Message(role="system", content="You are an expert assistant"),
            Message(role="user", content=sample_data["prompt"]),
        ]


@dataclass(frozen=True)
class MyConfig:
    """Main configuration - composes base configs."""

    model: BaseModelConfig = field(
        default_factory=lambda: BaseModelConfig(
            model_name="gpt-4",
            temperature=0.5,
        )
    )

    environment: MyEnvironmentConfig = field(
        default_factory=MyEnvironmentConfig
    )

    evaluation: BaseEvaluationConfig = field(
        default_factory=lambda: BaseEvaluationConfig(
            eval_name="my_eval",
            max_turns=5,
            num_samples=100,
        )
    )

    experiment_name: str = "my_experiment"


# Export config instance
config = MyConfig()
```

```python
# entrypoint.py
import sys
from rollouts.config import load_config_from_file

# Load config
config = load_config_from_file(sys.argv[1])

# Use config
endpoint = config.model.to_endpoint()
messages = config.environment.prepare_messages(sample_data)
eval_config = config.evaluation.to_eval_config(reward_fn)

# Run evaluation
from rollouts import evaluate
results = await evaluate(dataset, prepare_messages, endpoint, eval_config)
```

## Migration from Old Configs

See `kernels-gpumode-agent/CONFIG_MIGRATION.md` for detailed migration guide.

**Summary:**
1. Split monolithic config into composable pieces
2. Use BaseModelConfig for model settings
3. Extend BaseEnvironmentConfig for environment
4. Use BaseEvaluationConfig for eval settings
5. Compose in main config class

## Design Decisions

### Why Protocols?

- **No coupling** - Protocols don't force inheritance
- **Duck typing** - Any object with the right methods works
- **Runtime checkable** - `isinstance(obj, Protocol)` works
- **Easy to remove** - Just stop importing them

### Why Base Configs?

- **Reference implementations** - Show the pattern
- **Reduce duplication** - Common fields in one place
- **Easy to customize** - Override what you need
- **Not required** - Can implement protocols directly

### Why Frozen Dataclasses?

- **Immutable** - Configs are specifications, not mutable state
- **Hashable** - Can use as dict keys
- **Thread-safe** - No accidental mutation
- **Clear intent** - Configs don't change at runtime

### Why Composition?

- **Separation of concerns** - Model, environment, eval separate
- **Easier to test** - Test components independently
- **More flexible** - Mix and match components
- **No coupling** - Components don't know about each other

## Code Style Compliance

✅ **Casey Muratori - Semantic Compression**
- Extracted patterns after seeing 6+ projects
- Only created abstractions from real code

✅ **Casey Muratori - API Design**
- Granularity: Multiple entry points
- Redundancy: Multiple ways to achieve goals
- No coupling: Protocols not inheritance
- Flow control: You call config

✅ **Tiger Style Safety**
- All defaults explicit
- Immutable configs
- No exceptions for control flow
- Type-safe protocols
