# Features from pi-mono to Consider for rollouts

This document catalogs features from [pi-mono](https://github.com/badlogic/pi-mono) (specifically the `pi-coding-agent` package) that rollouts doesn't have, with implementation notes.

---

## Tier 1: High Impact

### 1. AGENTS.md / CLAUDE.md Discovery

**What it does:** Auto-loads project context files in hierarchical order:
1. Global: `~/.pi/agent/AGENTS.md`
2. Parent directories: Walking up from cwd
3. Current directory: `./AGENTS.md`

All files are concatenated and injected into the system prompt.

**Use cases:**
- Project conventions and coding style
- Common commands (`npm run build`, `pytest`, etc.)
- Architecture documentation
- Team-specific instructions

**Example AGENTS.md:**
```markdown
# Project: rollouts

## Commands
- `uv pip install -e .` - Install
- `pytest tests/` - Run tests
- `ruff check .` - Lint

## Code Style
- Use type hints everywhere
- Prefer dataclasses over dicts
- No `Any` types unless necessary

## Architecture
- `rollouts/tui/` - Terminal UI (Textual)
- `rollouts/providers/` - LLM provider implementations
- `rollouts/environments/` - Tool environments (coding, git, etc.)
```

**Implementation notes:**
- Simple file discovery + concatenation
- Could use `.rollouts/AGENTS.md` or adopt Claude Code's `CLAUDE.md` for compatibility
- Add `--no-agents-md` flag to disable
- Show in `/session` info which files were loaded

---

### 2. Mid-Session Model Switching

**What it does:** `/model` command opens fuzzy-search picker to switch models without losing conversation history.

**Pi's implementation:**
- `Ctrl+P` cycles through models (scoped by `--models` flag)
- `/model` opens full picker with fuzzy search
- Thinking level can be set per-model: `--models sonnet:high,haiku:low`

**Use cases:**
- Start with cheap/fast model for exploration
- Switch to capable model when you find the real problem
- Use different models for different tasks in same session

**Implementation notes:**
- Session already stores messages provider-agnostically
- Need to handle: different context windows, different tool schemas
- Could add `/model` slash command + `Ctrl+P` keybind
- Store model switches in session for replay?

---

### 3. Hooks System

**What it does:** TypeScript modules that subscribe to lifecycle events and can intercept/modify behavior.

**Events:**
```
session (start, shutdown, before_branch, branch, before_switch, switch, before_compact, compact)
agent_start / agent_end
turn_start / turn_end
tool_call (can block or modify)
tool_result (can modify output)
```

**Hook API:**
```typescript
export default function (pi: HookAPI) {
  pi.on("tool_call", async (event, ctx) => {
    if (event.toolName === "bash" && /sudo/.test(event.input.command)) {
      const ok = await ctx.ui.confirm("Allow sudo?", event.input.command);
      if (!ok) return { block: true, reason: "Blocked by user" };
    }
    return undefined;
  });
}
```

**Example hooks from pi-mono:**

| Hook | Description |
|------|-------------|
| `permission-gate.ts` | Confirm before `rm -rf`, `sudo`, `chmod 777` |
| `git-checkpoint.ts` | Stash before each turn, restore on `/branch` |
| `protected-paths.ts` | Block writes to `.env`, `node_modules/`, etc. |
| `file-trigger.ts` | Watch file, inject contents as message (external triggers) |
| `custom-compaction.ts` | Use different model (Gemini Flash) for summarization |
| `auto-commit-on-exit.ts` | Auto-commit changes when session ends |
| `dirty-repo-guard.ts` | Warn if repo has uncommitted changes |

**Key capability - `pi.send()`:** Hooks can inject messages into the conversation:
```typescript
fs.watch("/tmp/trigger.txt", () => {
  const content = fs.readFileSync("/tmp/trigger.txt", "utf-8");
  pi.send(content);  // Wakes up agent with this message
});
```

**Implementation notes:**
- Python equivalent: decorators or simple function registration
- Location: `~/.rollouts/hooks/*.py` and `.rollouts/hooks/*.py`
- Events map well to existing code structure
- `tool_call` interception is most valuable (permission gates)
- `send()` for external triggers is powerful but complex

---

### 4. Custom Slash Commands

**What it does:** Markdown files that become slash commands with argument substitution.

**Location:** `~/.pi/agent/commands/*.md` and `.pi/commands/*.md`

**Format:**
```markdown
---
description: Review staged git changes
---
Review the staged changes (`git diff --cached`). Focus on:
- Bugs and logic errors
- Security issues
- Error handling gaps
```

Filename (without `.md`) becomes command name. Description shown in autocomplete.

**Arguments:**
```markdown
---
description: Create a component
---
Create a React component named $1 with features: $@
```
- `$1`, `$2`, etc. = positional args
- `$@` = all args joined

**Namespacing:** Subdirectories create prefixes: `.pi/commands/frontend/component.md` → `/component (project:frontend)`

**Implementation notes:**
- Very simple to implement
- Complements presets (presets = model+env+prompt, commands = just prompts)
- Could use `.rollouts/commands/*.md`
- Autocomplete in TUI shows description

---

## Tier 2: Medium Impact

### 5. Skills (On-Demand Capability Packages)

**What it does:** Self-contained capability packages loaded when the agent decides they're relevant.

**Spec:** Implements [Agent Skills standard](https://agentskills.io/specification)

**Structure:**
```
my-skill/
├── SKILL.md              # Required: frontmatter + instructions
├── scripts/              # Helper scripts
│   └── search.js
└── references/           # Detailed docs loaded on-demand
    └── api-reference.md
```

**SKILL.md format:**
```markdown
---
name: brave-search
description: Web search via Brave Search API. Use for documentation, facts, or web content.
---

# Brave Search

## Setup
\`\`\`bash
cd /path/to/skill && npm install
\`\`\`

## Usage
\`\`\`bash
./search.js "query"           # Basic search
./search.js "query" --content # Include page content
\`\`\`
```

**How it works:**
1. Agent sees skill descriptions in system prompt (just name + description)
2. When task matches, agent requests to load the skill
3. Full SKILL.md content is injected
4. Agent uses the scripts/workflows defined

**Use cases:**
- Web search (Brave, Serper)
- Browser automation (Playwright, CDP)
- Google Calendar/Gmail/Drive
- PDF/DOCX processing
- Speech-to-text

**Implementation notes:**
- Locations: `~/.rollouts/skills/*/SKILL.md` and `.rollouts/skills/*/SKILL.md`
- Just need: discovery, description extraction, on-demand loading
- Could add `--skills <patterns>` to filter
- Compatible with Claude Code's skill locations

---

### 6. Custom Tools

**What it does:** User-defined tools the LLM can call directly, with custom TUI rendering.

**Location:** `~/.pi/agent/tools/*/index.ts` and `.pi/tools/*/index.ts`

**Example:**
```typescript
const factory: CustomToolFactory = (pi) => ({
  name: "todo",
  label: "Todo",
  description: "Manage a todo list. Actions: list, add, toggle, clear",
  parameters: Type.Object({
    action: StringEnum(["list", "add", "toggle", "clear"]),
    text: Type.Optional(Type.String()),
  }),

  async execute(toolCallId, params) {
    // ... implementation
    return {
      content: [{ type: "text", text: "Added todo #1" }],
      details: { todos: [...] },  // For custom rendering
    };
  },

  // Optional: reconstruct state from session history
  onSession(event) { /* scan past tool results */ },

  // Optional: custom TUI rendering
  renderCall(args, theme) { /* return Component */ },
  renderResult(result, options, theme) { /* return Component */ },
});
```

**Tool API:**
```typescript
interface ToolAPI {
  cwd: string;
  exec(command: string, args: string[]): Promise<ExecResult>;
  ui: {
    select(title: string, options: string[]): Promise<string | null>;
    confirm(title: string, message: string): Promise<boolean>;
    input(title: string, placeholder?: string): Promise<string | null>;
    notify(message: string, type?: "info" | "warning" | "error"): void;
  };
  hasUI: boolean;
}
```

**Example tools from pi-mono:**
- `todo/` - Stateful todo list with session reconstruction
- `question/` - Ask user multiple-choice questions
- `subagent/` - Spawn isolated agent processes (parallel, chain, single)

**Implementation notes:**
- Python version: `.rollouts/tools/*/tool.py` with a `Tool` class
- Need to define a clean Python API for tools
- State reconstruction from session is the tricky part
- Custom rendering via Rich or Textual components

---

### 7. Subagent Tool

**What it does:** Spawns isolated `pi` processes for delegated tasks.

**Modes:**
```python
# Single task
{"agent": "reviewer", "task": "Review the auth module"}

# Parallel tasks
{"tasks": [
  {"agent": "reviewer", "task": "Review auth"},
  {"agent": "reviewer", "task": "Review database"},
]}

# Chain (output flows to next)
{"chain": [
  {"agent": "planner", "task": "Plan the refactor"},
  {"agent": "coder", "task": "Implement: {previous}"},
  {"agent": "reviewer", "task": "Review: {previous}"},
]}
```

**Agent definitions:** `~/.pi/agent/agents/*.md` files with frontmatter:
```markdown
---
name: reviewer
model: claude-sonnet-4-20250514
tools: [read, grep, find]
---

You are a code reviewer. Focus on:
- Security issues
- Performance problems
- Code clarity
```

**Implementation notes:**
- Could be a custom tool or built-in
- Spawns `rollouts` subprocesses with `--stream-json`
- Isolated context windows per subtask
- Agent definitions similar to presets but lighter

---

### 8. Bash Mode (`!command`)

**What it does:** Prefix with `!` to run commands directly, output added to context.

```
!ls -la
!git status
!cat package.json | jq '.dependencies'
```

Output streams in real-time, truncates at 2000 lines / 50KB.

Becomes part of your next prompt:
```
Ran `ls -la`
```
<output>
```
```

**Implementation notes:**
- Very simple TUI addition
- Detect `!` prefix, run via subprocess
- Format output and prepend to next user message
- Add escape handling, streaming display

---

### 9. `@file` Fuzzy Search

**What it does:** Type `@` to fuzzy-search project files. Respects `.gitignore`.

Selected file content is attached to the message.

**Implementation notes:**
- Need fuzzy finder (like `fzf` or Python equivalent)
- Walk project files, filter by `.gitignore`
- On selection, read and attach content
- Could use `prompt_toolkit` completions

---

## Tier 3: Nice to Have

### 10. Message Queuing

**What it does:** Submit messages while agent is working. They queue and process based on mode:
- `all` - Process all queued messages
- `one-at-a-time` - Process one, show queue

Press Escape to abort and restore queued messages to editor.

**Implementation notes:**
- TUI change: non-blocking input during agent turn
- Queue data structure for pending messages
- Escape handler to cancel + restore

---

### 11. Context Compaction with Custom Instructions

**What it does:** `/compact [instructions]` - manual compaction with optional focus.

```
/compact Focus on the API changes we discussed
/compact Summarize the debugging session, keep the solution
```

**Implementation notes:**
- rollouts has `--slice` with `summarize` but no custom instructions
- Add optional goal param to summarization prompt
- Could extend `/compact` or add to `--slice` syntax

---

### 12. Inline Image Rendering

**What it does:** Renders images inline on supported terminals (Kitty, iTerm2, WezTerm, Ghostty).

Uses Kitty graphics protocol or iTerm2 inline images.

**Implementation notes:**
- Python libraries exist: `term-image`, `pixcat`
- Detect terminal capability
- Render in tool output display
- Fallback to text placeholder

---

### 13. More OAuth Providers

**Pi supports:**
| Provider | Models | Cost |
|----------|--------|------|
| GitHub Copilot | GPT-4o, Claude, Gemini | Subscription |
| Google Gemini CLI | Gemini 2.0/2.5 | Free |
| Google Antigravity | Gemini 3, Claude, GPT-OSS | Free |

**Implementation notes:**
- Each provider has its own OAuth flow
- GitHub Copilot: device flow → token
- Google: standard OAuth2
- Need to store tokens securely

---

### 14. Thinking Level Cycling

**What it does:** `Shift+Tab` cycles thinking levels without menu.

Levels: `off` → `minimal` → `low` → `medium` → `high` → `xhigh`

**Implementation notes:**
- Simple keybind in TUI
- Update config, show indicator
- Already have `--thinking` flag

---

## Implementation Priority Recommendation

### Phase 1: Quick Wins (1-2 days each)
1. **AGENTS.md discovery** - Simple file concatenation
2. **Custom slash commands** - Markdown templates
3. **Bash mode (`!cmd`)** - TUI prefix detection

### Phase 2: Medium Effort (3-5 days each)
4. **Mid-session model switching** - `/model` command
5. **Skills** - Discovery + on-demand loading
6. **`@file` fuzzy search** - File picker integration

### Phase 3: Larger Features (1-2 weeks each)
7. **Hooks system** - Event architecture + Python API
8. **Custom tools** - Tool definition API + discovery
9. **Subagent tool** - Process spawning + result aggregation

---

## Compatibility Notes

### Shared Standards
- **AGENTS.md / CLAUDE.md** - Claude Code uses same pattern
- **Skills** - [agentskills.io](https://agentskills.io) is a published spec
- **Session format** - Both use JSONL, could align schemas

### What rollouts has that pi-mono doesn't
- `--slice` with summarize/compact/inject operations
- `--handoff` for goal-directed context extraction
- `--stream-json` for scripting
- Git environment with isolated history
- Presets system (model + env + prompt bundles)
- Exit survey / feedback collection
