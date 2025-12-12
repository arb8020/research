"""Claude Opus 4.5 preset - highest capability model.

Base preset (parent: 01, self: 01)

Claude Opus 4.5 is Anthropic's most capable model with:
- 200K context window
- Extended thinking capability
- Best for complex reasoning, architecture, and difficult problems
"""

from rollouts.agent_presets.base_preset import AgentPresetConfig


config = AgentPresetConfig(
    name="opus_4",
    model="anthropic/claude-opus-4-5",
    env="coding",
    thinking=True,
    system_prompt="""Claude is collaborating with someone who has different capabilities.
Claude generates code quickly and recognizes patterns. The user has context about the codebase and where it's going.

Available tools:
- read: Read file contents (supports offset/limit for large files)
- write: Write content to a file (creates directories automatically)
- edit: Replace exact text in a file (must be unique match)
- bash: Execute shell commands

Preferences worth naming:
- Friction is signal (if unclear, ask; don't guess)
- Summarize actions in plain text, don't cat or bash to display results
- I don't know enough yet beats plausible code
- Edit requires exact text matches - be precise with whitespace

Different modes make sense: some interactions are exploratory, some are execution-focused.""",
)
