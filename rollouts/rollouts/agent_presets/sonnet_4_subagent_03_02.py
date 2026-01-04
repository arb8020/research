"""Claude Sonnet 4.5 with subagent spawning guidance.

Derived from sonnet_4_02_02 (parent: 02, self: 03)

Changes:
- Adds guidance on spawning sub-agents for focused tasks
- Documents pattern for keeping main context clean

Inspired by pi-coding-agent's philosophy:
- Sub-agents via bash for full observability
- Context gathering in separate sessions, not mid-task
- Create artifacts first, then use in fresh sessions
"""

from dataclasses import replace

from ..agent_presets.sonnet_4_02_02 import config as parent_config

SUBAGENT_GUIDANCE = """

## Spawning sub-agents

For tasks that benefit from isolated context, spawn yourself as a sub-agent via bash.
This keeps your main context clean and gives full observability of what the sub-agent does.

```bash
# Basic pattern - output returns to you
rollouts -p "focused task description" --env coding --no-session

# For longer tasks, capture to a file you can read back
rollouts -p "analyze the auth module" --env coding --no-session > /tmp/analysis.md && cat /tmp/analysis.md
```

Good uses for sub-agents:
- **Code review**: `rollouts -p "review the changes in git diff HEAD~3" --env coding --no-session`
- **Research/exploration**: `rollouts -p "explain the architecture of src/auth/" --env coding --no-session`
- **Generate artifacts**: spawn to create a summary/analysis doc, then read it back

When NOT to use sub-agents:
- For simple tasks you can do directly
- When you need to make changes (sub-agent changes won't persist to user's expectations)
- Mid-task context gathering is often a sign you should have explored first

Pattern for large tasks:
1. First session: explore and create an artifact (PLAN.md, CONTEXT.md)
2. Handoff or start fresh session with that artifact
3. Execute with clean context

For read-only exploration (safer for sub-agents doing research):
```bash
rollouts -p "analyze the codebase structure" --env coding --tools readonly --no-session
```

Available --tools presets: full (default), readonly (just read), no-write (read/edit/bash)
"""

config = replace(
    parent_config,
    name="sonnet_4_subagent",
    system_prompt=parent_config.system_prompt + SUBAGENT_GUIDANCE,
)
