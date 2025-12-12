🔐 Using OAuth authentication (Claude Pro/Max)
# Goal: Improve Agent Presets for Rollouts

## Context

You're working on **rollouts**, a TUI for interacting with LLM agents. We just implemented an agent preset system that bundles `(model, environment, system_prompt)` configurations.

### Design Decisions Made

1. **Python files over YAML** - Matches existing experiment config pattern (`docs/code_style/experiment_config.md`)
2. **Naming convention**: `<name>_<id>_<parent>.py` (e.g., `fast_coder_01_01.py`)
3. **Inheritance via `dataclasses.replace()`** - Same pattern as SFT/RL experiment configs
4. **Directory**: `rollouts/agent_presets/` (not just `presets/` - be explicit)

### Current System Prompt Philosophy

From the session discussion, good system prompts should:
- Frame collaboration between different capabilities (Claude vs user)
- Set expectations: "Friction is signal" (ask vs guess)
- Acknowledge different modes (exploratory vs execution)
- Be practical: "I don't know enough yet beats plausible code"
- Include concrete tool guidance

### Available Environments

- **none**: No tools, just chat
- **calculator**: Math tools (add, subtract, multiply, divide)
- **coding**: File tools (read, write, edit, bash)
- **git**: Coding tools + automatic git tracking (every change = commit)

### Current Presets

1. **fast_coder_01_01** - Claude 3.5 Haiku, coding env, quick iteration
2. **careful_coder_02_01** - Claude Sonnet 4.5, coding env, complex work (derived from fast_coder)
3. **git_explorer_03_03** - Claude Sonnet 4.5, git env, experimental changes

## Relevant Files

**Implementation (complete, tested ✓):**
- `rollouts/rollouts/agent_presets/base_preset.py` - Schema/dataclass
- `rollouts/rollouts/agent_presets/loader.py` - Loading logic
- `rollouts/rollouts/agent_presets/fast_coder_01_01.py` - Example preset
- `rollouts/rollouts/agent_presets/careful_coder_02_01.py` - Derived preset
- `rollouts/rollouts/agent_presets/git_explorer_03_03.py` - Git env preset
- `rollouts/rollouts/agent_presets/README.md` - Full documentation
- `rollouts/rollouts/cli.py` - CLI integration (--preset, --list-presets)

**Reference:**
- `rollouts/rollouts/environments/coding.py` - Coding environment implementation, original system prompt
- `docs/code_style/experiment_config.md` - Config philosophy (Pythonic + hierarchical + serializable)

**Testing:**
- `rollouts/test_presets.py` - Tests for preset loading (all passing)

## Task

**Goal**: Create better/more presets

Ideas to explore:
1. **More specific presets** - E.g., debugging-focused, testing-focused, documentation-focused
2. **Different model tiers** - E.g., opus variants, different providers (OpenAI, etc.)
3. **Specialized prompts** - Refine system prompts for specific use cases
4. **Environment variations** - Different coding environment configs (e.g., specific working_dir settings)
5. **Calculator/git variants** - Currently only have coding presets

**How to create new presets:**
```python
# Base preset: rollouts/agent_presets/my_preset_04_04.py
from rollouts.agent_presets.base_preset import AgentPresetConfig

config = AgentPresetConfig(
    name="my_preset",
    model="anthropic/claude-3-5-haiku-20241022",
    env="coding",
    system_prompt="""Your prompt here...""",
)
```

**Test with:**
```bash
cd rollouts
python -m rollouts.cli --list-presets
python -m rollouts.cli --preset my_preset
```
