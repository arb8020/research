# Agent Presets

Agent presets bundle `(model, environment, system_prompt)` configurations for quick iteration on agent behavior.

## Design Philosophy

Follows `docs/code_style/experiment_config.md`:
- **Pythonic** - Configs are Python code, can compute values
- **Hierarchical** - Compose and extend presets
- **Serializable** - Save exact config for reproducibility
- **Type-safe** - IDE autocomplete, catch typos at edit time
- **Versionable** - Track lineage with `<name>_<id>_<parent>.py` naming

## Quick Start

### Using Presets

```bash
# List available presets
rollouts --list-presets

# Use a preset (picks model + optimized prompt)
rollouts --preset opus_4
rollouts --preset sonnet_4
rollouts --preset gpt_5_2
rollouts --preset gpt_5_1_codex

# Override specific values if needed
rollouts --preset sonnet_4 --env git

# Use preset from custom file
rollouts --preset ~/my-presets/custom_01_01.py
```

### Available Presets

Presets are optimized for specific models, with system prompts tailored to each model's characteristics:

- **`opus_4_01_01`** - Claude Opus 4.5
  - Anthropic's most capable model
  - 200K context, extended thinking
  - Best for: Complex reasoning, architecture, difficult problems
  
- **`sonnet_4_02_02`** - Claude Sonnet 4.5
  - Balanced capability and speed
  - 200K context, extended thinking
  - Best for: General purpose, good default choice
  
- **`gpt_5_2_03_03`** - GPT-5.2
  - OpenAI's latest model
  - Strong general capabilities
  - Best for: Diverse tasks, latest features
  
- **`gpt_5_1_codex_04_04`** - GPT-5.1 Codex
  - OpenAI's code-optimized model
  - 400K context, 128K output, reasoning
  - Best for: Large codebases, complex coding tasks

## Creating Presets

### Base Preset (New Family)

```python
# rollouts/agent_presets/my_preset_01_01.py
"""My custom preset - description here.

Base preset (parent: 01, self: 01)

Good for:
- Use case 1
- Use case 2
"""

from rollouts.agent_presets.base_preset import AgentPresetConfig

config = AgentPresetConfig(
    name="my_preset",
    model="anthropic/claude-3-5-haiku-20241022",
    env="coding",
    thinking=True,
    system_prompt="""Your custom system prompt here...""",
)
```

### Derived Preset (Inherit + Override)

```python
# rollouts/agent_presets/my_variant_02_01.py
"""Variant of my_preset with different model.

Derived from my_preset_01_01 (parent: 01, self: 02)
Changes: Uses Claude Sonnet for better reasoning.
"""

from dataclasses import replace
from rollouts.agent_presets.my_preset_01_01 import config as parent_config

# Inherit from parent, only override what changed
config = replace(
    parent_config,
    name="my_variant",
    model="anthropic/claude-sonnet-4-5-20250929",
)
```

## Naming Convention

Pattern: `<name>_<id>_<parent>.py`

- **`name`**: Descriptive name (e.g., `fast_coder`, `careful_coder`)
- **`id`**: Sequence number (01, 02, 03...)
- **`parent`**: ID of parent config this was derived from

Examples:
- `fast_coder_01_01.py` - Base config (parent is self)
- `careful_coder_02_01.py` - Derived from config 01
- `git_explorer_03_03.py` - New family, base config (parent is self)

**Benefits:**
- ✅ Name-first ordering for easy scanning
- ✅ Traceable lineage through parent ID
- ✅ Fuzzy matching: `--preset fast_coder` finds `fast_coder_01_01.py`

## Config Structure

### AgentPresetConfig

```python
@dataclass(frozen=True)
class AgentPresetConfig:
    # Required
    name: str                  # Preset identifier
    model: str                 # "provider/model" format
    env: str                   # "none", "calculator", "coding", "git"
    system_prompt: str         # Full system prompt text
    
    # Optional
    thinking: bool = True      # Enable extended thinking
    temperature: float | None = None
    working_dir: Path | None = None
```

### Methods

- **`save(path)`** - Save preset as JSON for reproducibility
- **`load(path)`** - Load saved preset
- **`to_cli_args()`** - Convert to dict for CLI consumption

## Advanced Usage

### Custom Preset Directory

```python
from rollouts.agent_presets import load_preset

# Load from custom directory
preset = load_preset("my_preset_01_01", preset_dir=Path("~/my-presets"))
```

### Programmatic Usage

```python
from rollouts.agent_presets import load_preset

preset = load_preset("fast_coder")

# Get CLI args
cli_args = preset.to_cli_args()

# Save for reproducibility
preset.save(Path("experiments/run_001/preset.json"))

# Load later
from rollouts.agent_presets.base_preset import AgentPresetConfig
loaded = AgentPresetConfig.load(Path("experiments/run_001/preset.json"))
```

### Preset in Experiments

```python
# experiments/my_experiment/config.py
from dataclasses import dataclass, field
from rollouts.agent_presets import load_preset

@dataclass(frozen=True)
class ExperimentConfig:
    agent_preset: str = "fast_coder_01_01"
    num_samples: int = 100
    
    @property
    def agent_config(self):
        """Load agent preset lazily."""
        return load_preset(self.agent_preset)

config = ExperimentConfig()

# Use in run script
endpoint = config.agent_config.to_cli_args()
```

## CLI Precedence

When using `--preset`, individual CLI args override preset values:

```bash
# Preset sets model=haiku, but CLI overrides to opus
rollouts --preset fast_coder --model anthropic/claude-opus-4

# Result: Uses opus model, everything else from fast_coder preset
```

**Precedence (least to most specific):**
1. Built-in defaults in code
2. Preset config (if `--preset` specified)
3. Individual CLI args (`--model`, `--env`, `--system-prompt`)

## Integration with experiment_config.md

Agent presets follow the same patterns as your experiment configs:

| Pattern | Experiment Configs | Agent Presets |
|---------|-------------------|---------------|
| Structure | `configs/<family>/<name>_<id>_<parent>.py` | `agent_presets/<name>_<id>_<parent>.py` |
| Base config | `base_config.py` defines schema | `base_preset.py` defines schema |
| Inheritance | `dataclasses.replace(parent, ...)` | `dataclasses.replace(parent, ...)` |
| Loading | `load_config_from_file()` | `load_preset()` |
| Serialization | `config.save()` → JSON | `preset.save()` → JSON |

**Key difference:** Agent presets are user-facing (CLI tool), while experiment configs are for training runs.

## Examples

Built-in model presets:
- [`opus_4_01_01.py`](opus_4_01_01.py) - Claude Opus 4.5 (highest capability)
- [`sonnet_4_02_02.py`](sonnet_4_02_02.py) - Claude Sonnet 4.5 (balanced)
- [`gpt_5_2_03_03.py`](gpt_5_2_03_03.py) - GPT-5.2 (latest OpenAI)
- [`gpt_5_1_codex_04_04.py`](gpt_5_1_codex_04_04.py) - GPT-5.1 Codex (code-optimized)

## Troubleshooting

### Import Error

```python
# ✗ Bad: Relative imports don't work
from .base_preset import AgentPresetConfig

# ✓ Good: Use absolute imports
from rollouts.agent_presets.base_preset import AgentPresetConfig
```

### Preset Not Found

```bash
$ rollouts --preset my_preset
Error: Preset 'my_preset' not found in rollouts/agent_presets/
Available presets: fast_coder_01_01, careful_coder_02_01, git_explorer_03_03
```

- Check preset file exists in `rollouts/agent_presets/`
- Try exact name: `--preset my_preset_01_01` instead of `--preset my_preset`
- Use `--list-presets` to see available options

### Custom File Not Loading

```bash
# ✗ Bad: Relative path without expansion
rollouts --preset ~/my-presets/custom.py

# ✓ Good: Absolute path or explicit file
rollouts --preset /Users/me/my-presets/custom.py
```

Note: Tilde expansion (`~`) is supported in `load_preset()` but may need explicit handling in shell.
