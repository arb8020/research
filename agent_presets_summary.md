# Agent Presets Implementation Summary

## What We Built

Added **agent presets** to rollouts - a way to bundle `(model, environment, system_prompt)` configurations following your existing experiment config patterns.

## File Structure

```
rollouts/rollouts/
├── agent_presets/
│   ├── __init__.py              # Package exports
│   ├── README.md                # Full documentation
│   ├── base_preset.py           # Schema (AgentPresetConfig dataclass)
│   ├── loader.py                # load_preset(), list_presets()
│   ├── fast_coder_01_01.py      # Example: Claude Haiku, fast iteration
│   ├── careful_coder_02_01.py   # Example: Derived from fast_coder, uses Sonnet
│   └── git_explorer_03_03.py    # Example: New family, git environment
└── cli.py                       # Updated with --preset, --list-presets
```

## Usage

### CLI

```bash
# List available presets
rollouts --list-presets

# Use a preset
rollouts --preset fast_coder

# Override specific parts
rollouts --preset fast_coder --model anthropic/claude-opus-4

# Use custom preset file
rollouts --preset ~/my-presets/custom_01_01.py
```

### Creating New Presets

**Base preset:**
```python
# rollouts/agent_presets/my_preset_01_01.py
from rollouts.agent_presets.base_preset import AgentPresetConfig

config = AgentPresetConfig(
    name="my_preset",
    model="anthropic/claude-3-5-haiku-20241022",
    env="coding",
    system_prompt="""Your prompt here...""",
)
```

**Derived preset:**
```python
# rollouts/agent_presets/my_variant_02_01.py
from dataclasses import replace
from rollouts.agent_presets.my_preset_01_01 import config as parent_config

config = replace(
    parent_config,
    name="my_variant",
    model="anthropic/claude-sonnet-4-5-20250929",
)
```

## Design Principles

Following `docs/code_style/experiment_config.md`:

✅ **Pythonic** - Configs are Python code (not YAML)
✅ **Hierarchical** - Compose via `dataclasses.replace()`
✅ **Serializable** - Save as JSON for reproducibility
✅ **Type-safe** - Frozen dataclasses, IDE autocomplete
✅ **Versionable** - Git-friendly, track lineage
✅ **Explicit** - All parameters visible, no magic
✅ **Consistent** - Same pattern as SFT/RL experiment configs

## Naming Convention

Pattern: `<name>_<id>_<parent>.py`

- `fast_coder_01_01.py` - Base (parent: 01, self: 01)
- `careful_coder_02_01.py` - Derived from 01
- `git_explorer_03_03.py` - New family, base (parent: 03, self: 03)

Benefits:
- Name-first ordering (not `01_fast_coder.py`)
- Fuzzy matching: `--preset fast_coder` finds `fast_coder_01_01.py`
- Traceable lineage through parent ID

## Key Features

1. **Preset Loading**
   - By name: `load_preset("fast_coder")` → fuzzy matches `fast_coder_01_01.py`
   - By exact name: `load_preset("fast_coder_01_01")`
   - By file path: `load_preset("~/my-presets/custom.py")`

2. **CLI Integration**
   - `--preset <name>` loads preset
   - Individual args override preset values
   - `--list-presets` shows available presets
   - Backward compatible (existing CLI args still work)

3. **Inheritance**
   - Import parent config: `from rollouts.agent_presets.parent import config as parent_config`
   - Override with `dataclasses.replace(parent_config, ...)`
   - Same pattern as your experiment configs

4. **Serialization**
   - `preset.save(path)` → JSON
   - `AgentPresetConfig.load(path)` → preset
   - Track exact config used in experiments

## Example Presets

### fast_coder_01_01
- **Model**: Claude 3.5 Haiku (fast)
- **Env**: coding
- **Use**: Rapid prototyping, small changes

### careful_coder_02_01
- **Model**: Claude Sonnet 4.5 (careful)
- **Env**: coding
- **Use**: Complex refactoring, critical bugs
- **Derived from**: fast_coder_01_01

### git_explorer_03_03
- **Model**: Claude Sonnet 4.5
- **Env**: git (worktree with auto-commit)
- **Use**: Experimental changes, high-risk mods

## Testing

Tests pass ✓

```bash
$ cd rollouts && python test_presets.py
============================================================
Agent Preset Tests
============================================================

Testing list_presets()...
Found 3 presets:
  - careful_coder_02_01
  - fast_coder_01_01
  - git_explorer_03_03
✓ list_presets() works

Testing load_preset('fast_coder_01_01')...
✓ Loaded: fast_coder (anthropic/claude-3-5-haiku-20241022)

Testing load_preset('fast_coder') (fuzzy match)...
✓ Loaded: fast_coder (anthropic/claude-3-5-haiku-20241022)

Testing load_preset('careful_coder_02_01')...
✓ Loaded: careful_coder (anthropic/claude-sonnet-4-5-20250929)

Testing preset.to_cli_args()...
✓ Generated CLI args: ['model', 'env', 'system_prompt', 'thinking']

============================================================
✓ All tests passed!
============================================================
```

## Why This Approach?

**Why Python files instead of YAML?**
- Matches your existing experiment config style
- Can compute values, import, use variables
- Type-safe with IDE support
- Easy to extend with `dataclasses.replace()`
- Git-friendly diffs
- No YAML parsing/validation needed

**Why agent_presets/ not just presets/?**
- Clear what it's for (agent configurations)
- Avoids ambiguity (presets for what?)
- Room for other preset types later

**Why frozen dataclasses?**
- Immutable → configs are specifications, not state
- Hashable → can use as dict keys
- Thread-safe
- Follows your existing patterns (Tiger Style)

## Next Steps

1. **Add more presets** - Create presets for common workflows
2. **Project-specific presets** - Users can keep presets in their repos
3. **Documentation** - Add to main rollouts docs
4. **Preset validator** - Check preset schema at load time

## Related Docs

- `rollouts/agent_presets/README.md` - Full preset documentation
- `docs/code_style/experiment_config.md` - Experiment config philosophy
- `rollouts/config/README.md` - Config protocols and base classes
