@samsja19
when testing a new training codebase if I don't see a debug config that allow me to do a short run on limited hardware then I start to question the ability of the dev to maintain the repo

---

@cHHillee
..., every group of researchers has their own opinion on how configs should be done.
(@giffmana: what config style did you converge to?)
Pythonic + hierarchical (not without contention!)

@_xjdr
gin meets toml is where we landed (surprisingly)

@cHHillee
Serialization is quite valuable.

@giffmana
Guys, trust me, this gin-style is nice on mini or solo codebase, but becomes a complete disaster as the codebase and/or team grows, and is almost impossible to remove later on. I've been through this twice.


---

@ezyang
People who run lots of small training jobs for your day job, what is one thing about experiment management / hygiene that you wish you knew when you started out?

@main_horse
always build the abstraction/interface/tool/database. do not ad-hoc, do not make one-off scripts / bespoke storage, do not do things only you will remember

@OnurBerkTore
- always use config, do not use args.
- spend time to learn wandb or similar, automatically saves code/configs/models.
- name the config with <#>_<active_change><#+1> (example: 22_LR_001_23.yaml, 22_LR_0001_24.yml)
- storage is cheap, copy your data rather than in-place modify.

@giffmana
It's worth it to make sure collecting and joining/slicing all their metrics and hparams post hoc is very low friction.


---

@eugeneyan
after leading a few projects, i've found that once you've set up the evals + experiment harness and make it easy to tweak config and prompts with 1-click run + eval, teams enjoy running experiments and hill climbing those numbers, and progress comes quickly.

but setting up that eval + experiment harness is a challenge for each new project. and few can and want to deal with that ambiguous work. and even after it's set up, few want to look at generated responses to human eval on the qualitative aspects, add new evals to quantify the qualitative, or just look at the raw data really. many just prefer to lean on the numbers.

counterintuitively, having evals and numbers help us get 80% of the way there faster, but maybe they're a crutch that it makes it harder to polish the final 20%?

---

## Alternative Approaches

### Functional Hierarchical Configs

[@ueaj's Optimal Configuration Code](https://publish.obsidian.md/ueaj/Machine+Learning/Pretraining/Optimal+Configuration+Code) - A functional approach using a `@config` decorator that allows composing and overriding configurations through function composition. Particularly good for ML research where you're frequently swapping architectures and components.

Key idea: Instead of data structures, use functions with defaults that can be partially overridden:
```python
@config
def llama(model_d, embed=create_embed, norm=create_norm):
    return LlamaModel(embed=embed(...), norm=norm(...))

# Override just the args
llama_v2 = llama.override(embed=override(dtype=jnp.bfloat16))

# Override the function entirely
llama_v3 = llama.override(embed=create_fp8_embed.override(init="kaiming"))
```

**Pros:** Max flexibility, programmatic value tying, great for research
**Cons:** Less explicit, harder to serialize, more complex mental model

---

## Our Working Config Style (2025-10-14)

Based on the above discussions and our code style principles (Casey Muratori's API design, Tiger Style safety, keep it simple), we're using:

**Pythonic + Hierarchical + Serializable**

### Structure
```python
# config.py - define the schema
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

@dataclass
class DataConfig:
    num_shards: int = 5
    data_dir: Path = Path("data/shards")
    processed_dir: Path = Path("data/processed")

@dataclass
class EmbeddingConfig:
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str | None = None

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    epochs: int = 10

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def save(self, path):
        """Save this exact config for reproducibility"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        """Load a saved config"""
        with open(path) as f:
            data = json.load(f)
        # Automatically instantiate nested dataclasses
        kwargs = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name in data:
                kwargs[field_name] = field_type(**data[field_name])
        return cls(**kwargs)
```

### Usage Pattern

**Experiments as Python files** (version controlled):
```python
# experiments/01_baseline.py
from config import Config

config = Config()  # all defaults

# experiments/02_high_lr_03.py
from config import Config

config = Config()
config.training.learning_rate = 1e-3

# experiments/03_big_model_04.py
from config import Config

config = Config()
config.embedding.model = "all-mpnet-base-v2"
config.embedding.batch_size = 32
```

**In training/experiment scripts**:
```python
# train.py
import sys
import importlib.util
from pathlib import Path

# Load experiment config
spec = importlib.util.spec_from_file_location("exp_config", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = module.config

# Run experiment
run_id = f"run_{timestamp}"
output_dir = Path(f"outputs/{run_id}")
output_dir.mkdir(parents=True, exist_ok=True)

# Save exact config used
config.save(output_dir / "config.json")

# Do the work
train(config)

# Later: reproduce exact run
# config = Config.load("outputs/run_123/config.json")
```

### Why this approach?

✅ **Pythonic**: Configs are Python code, can compute values, import, etc
✅ **Hierarchical**: Nested dataclasses organize related settings
✅ **Serializable**: Save exact config as JSON alongside outputs
✅ **Type-safe**: IDE autocomplete, catch typos at edit time
✅ **Versionable**: Commit experiment `.py` files, track lineage in names
✅ **No magic**: Just dataclasses and JSON, no frameworks
✅ **Casey-approved**: No coupling, transparent data, easy to override
✅ **Not Gin**: No string-based configuration, easy to remove later

### Naming convention
Following @OnurBerkTore: `<#>_<active_change>_<#+1>.py`
- `01_baseline.py` - experiment 1, baseline
- `02_high_lr_03.py` - from exp 2, trying high LR, becomes exp 3
- `03_big_model_04.py` - from exp 3, trying bigger model, becomes exp 4

This tracks experiment lineage in the filename itself.

---

## Our Implementation (2025-11-20)

We implemented the above pattern for our GPU kernel evaluation configs with some refinements:

### Naming Convention

Pattern: `<name>_<id>_<parent>.py`

Where:
- **`name`**: Descriptive name (e.g., `agent_multiturn`, `cutedsl_gpumode`, `agent_with_tools`)
- **`id`**: Sequence number scoped to benchmark directory (01, 02, 03...)
- **`parent`**: ID of the parent config this was derived from

Examples:
- `agent_multiturn_01_01.py` - Base config (parent is self)
- `agent_with_tools_02_01.py` - Derived from config 01
- `cutedsl_gpumode_03_01.py` - Derived from config 01

**Benefits:**
- ✅ Clean Python imports: `from configs.benchmarks.gpumode.agent_multiturn_01_01 import config`
- ✅ Name-first ordering for easy scanning
- ✅ Traceable lineage through parent ID

### Config Inheritance Pattern

Base config (`agent_multiturn_01_01.py`):
```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Config:
    model: BaseModelConfig = field(default_factory=lambda: BaseModelConfig(...))
    target: TargetConfig = field(default_factory=lambda: TargetConfig(...))
    environment_class: type = GPUModeEnvironment
    environment_config: dict = field(default_factory=lambda: {...})
    evaluation: BaseEvaluationConfig = field(default_factory=lambda: BaseEvaluationConfig(...))
    experiment_name: str = "custom_experiment"

config = Config()

def prepare_messages(sample_data):
    # Prepare prompts
    return [...]

reward_fn = kernel_test_reward_fn
```

Derived config (`agent_with_tools_02_01.py`):
```python
from dataclasses import replace
from configs.benchmarks.gpumode.agent_multiturn_01_01 import (
    config as parent_config,
    reward_fn,  # Reuse from parent
)

# Override only what changed
config = replace(
    parent_config,
    environment_class=KernelTestEnvironmentWithTools,
    target=replace(parent_config.target, gpu_ranks=[6]),
    evaluation=replace(
        parent_config.evaluation,
        max_turns=10,
        num_samples=1,
    ),
    model=replace(parent_config.model, model_name="claude-sonnet-4-5-20250929"),
    experiment_name="tool_based_kernel_dev",
)

# Override prepare_messages if needed
def prepare_messages(sample_data):
    # Different prompt for tool-based workflow
    return [...]
```

**Key advantages:**
- Explicit inheritance via imports (not magic)
- Only override what changes (DRY principle)
- Easy to see diff from parent
- Type-safe with dataclasses
- Still serializable for reproducibility
- IDs scoped per benchmark directory

### Directory Structure

Each experiment family (task + model combination) gets its own directory. IDs are scoped per directory - new base config means new directory, numbering restarts at 01.

```
experiments/
├── sft/
│   ├── qwen_0.5b/                  # family: SFT on Qwen 0.5B
│   │   ├── baseline_01_01.py       # base config (parent is self)
│   │   ├── lora_02_01.py           # derived from 01
│   │   └── lora_bf16_03_02.py      # derived from 02
│   └── llama_3b/                   # new family, IDs restart at 01
│       ├── baseline_01_01.py       # new base
│       └── long_ctx_02_01.py       # derived from 01
├── rl/
│   ├── calculator/
│   │   ├── grpo_01_01.py
│   │   └── grpo_high_lr_02_01.py
│   └── code_exec/
│       └── ppo_01_01.py
```

**Rule:** New base config = new directory. Don't end up with `sft_qwen_47_38.py` - keep IDs small and scoped.
