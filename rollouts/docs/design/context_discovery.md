# Context File Discovery & System Prompt Visibility

> Auto-discover and inject project context files (AGENTS.md, ROLLOUTS.md, CLAUDE.md) into system prompts, with explicit opt-out. Show system prompt in TUI for full transparency.

## Current State

Context discovery is **already implemented** in `rollouts/prompt.py`:

```python
# prompt.py
PROJECT_CONTEXT_FILES = ["ROLLOUTS.md", "AGENTS.md", "CLAUDE.md"]

def load_project_context(cwd: Path) -> list[tuple[Path, str]]:
    """Load project context files walking up from cwd."""
    # Walks from cwd to root, finds first matching file per directory
    # Returns in root-first order (more specific context comes last)
    ...

def build_system_prompt(..., include_project_context: bool = True):
    if include_project_context:
        context_files = load_project_context(working_dir)
        # Injects as "# Project Context" section
```

The CLI uses this when building prompts:

```python
# cli.py:1252
system_prompt = build_system_prompt(
    env_name=config.env,
    tools=config.environment.get_tools(),
    cwd=config.working_dir,
    env_system_prompt=env_system_prompt,
)
```

## What's Missing

### 1. No CLI Flag to Disable Discovery

Currently no way to opt-out:
```bash
rollouts --env coding  # Always discovers context files
```

Per user request, everything should be explicit with opt-out available.

### 2. No Global Context File

The code has a TODO:
```python
# TODO: Add global context from ~/.rollouts/AGENTS.md (or CLAUDE.md)
# See badlogic/pi-mono's loadProjectContextFiles for reference:
# 1. Load global context from agentDir first (~/.rollouts/)
# 2. Then walk up from cwd to root
```

pi-mono loads `~/.pi/agent/AGENTS.md` first, then walks up from cwd.

### 3. No Visibility Into What Was Loaded

User has no way to see which context files were discovered and injected.

## Proposed Changes

### 1. Add `--no-discover` Flag

```python
# cli.py - add to argument parser
parser.add_argument(
    "--no-discover",
    action="store_true",
    default=False,
    help="Don't auto-discover AGENTS.md/ROLLOUTS.md context files",
)

# CLIConfig
@dataclass
class CLIConfig:
    ...
    no_discover: bool = False
```

Wire into prompt building:
```python
system_prompt = build_system_prompt(
    env_name=config.env,
    tools=config.environment.get_tools(),
    cwd=config.working_dir,
    env_system_prompt=env_system_prompt,
    include_project_context=not config.no_discover,  # NEW
)
```

### 2. Add Global Context File Support

Update `load_project_context`:

```python
def load_project_context(cwd: Path, include_global: bool = True) -> list[tuple[Path, str]]:
    """Load project context files from global + cwd ancestry.
    
    Discovery order:
    1. Global: ~/.rollouts/AGENTS.md (or CLAUDE.md)
    2. Walk from root to cwd, one file per directory
    
    Returns in that order (global first, most specific last).
    """
    context_files: list[tuple[Path, str]] = []
    seen_paths: set[Path] = set()
    
    # 1. Global context
    if include_global:
        global_dir = Path.home() / ".rollouts"
        for name in PROJECT_CONTEXT_FILES:
            global_file = global_dir / name
            if global_file.exists():
                try:
                    content = global_file.read_text()
                    context_files.append((global_file, content))
                    seen_paths.add(global_file)
                    break  # Only one global file
                except (OSError, PermissionError):
                    pass
    
    # 2. Walk up from cwd (existing logic)
    ...
```

### 3. Add `--show-context` Flag (Optional)

Print discovered files and exit:

```bash
$ rollouts --show-context
Context files discovered:
  ~/.rollouts/AGENTS.md (global)
  ~/projects/AGENTS.md
  ~/projects/myapp/AGENTS.md (cwd)
```

Implementation:
```python
parser.add_argument(
    "--show-context",
    action="store_true",
    help="Show discovered context files and exit",
)

# In main():
if config.show_context:
    from rollouts.prompt import load_project_context
    files = load_project_context(config.working_dir)
    if files:
        print("Context files discovered:")
        for path, _ in files:
            label = "(global)" if ".rollouts" in str(path) else ""
            if path.parent == config.working_dir:
                label = "(cwd)"
            print(f"  {path} {label}")
    else:
        print("No context files found")
    sys.exit(0)
```

## File Format

Context files are plain Markdown, injected verbatim:

```markdown
# Project: myapp

## Commands
- `uv pip install -e .` - Install
- `pytest tests/` - Run tests

## Architecture
- `src/api/` - FastAPI endpoints
- `src/core/` - Business logic

## Conventions
- Use type hints everywhere
- Prefer dataclasses over dicts
```

## Implementation Plan

### Phase 1: Add `--no-discover` flag
- [ ] Add argument to parser
- [ ] Add to CLIConfig  
- [ ] Wire into `build_system_prompt` call
- [ ] Update help text

### Phase 2: Add global context support
- [ ] Update `load_project_context` to check `~/.rollouts/`
- [ ] Ensure proper ordering (global first)
- [ ] Test with existing project files

### Phase 3: Add `--show-context` flag (optional)
- [ ] Add argument to parser
- [ ] Implement display logic
- [ ] Exit after display

## CLI Summary

```
--no-discover       Don't auto-discover AGENTS.md/ROLLOUTS.md context files
--show-context      Show discovered context files and exit
```

## Compatibility

| File | Compatibility |
|------|---------------|
| `ROLLOUTS.md` | rollouts-native |
| `AGENTS.md` | pi-mono, Mistral Vibe |
| `CLAUDE.md` | Claude Code |

All three are checked in priority order. First found per directory wins.

---

## Part 2: System Prompt Visibility

### Philosophy

From pi-mono author: "I want to inspect every aspect of my interactions with the model."

Currently the system prompt is invisible to users. They can't see:
- What base prompt is being used
- Which AGENTS.md files were loaded
- What tools are available
- Runtime context (cwd, datetime)

### Proposal: Render System Prompt as First Message

Display the system prompt at the top of the chat, just like user/assistant messages:

```
┌─ System ─────────────────────────────────────────
│ You are a coding assistant with access to file and shell tools.
│ 
│ Available tools:
│ - read: Read file contents...
│ - write: Write content to a file...
│ 
│ # Project Context
│ 
│ ## ~/projects/myapp/AGENTS.md
│ 
│ # Project: myapp
│ - FastAPI app, see docs/
│ 
│ Current time: 2025-12-25 17:45:00
│ Working directory: /Users/chiraag/projects/myapp
└──────────────────────────────────────────────────

> what files are in this project?
```

### Expand/Collapse Mechanism

System prompts can be long (1-2k tokens). Need expand/collapse.

#### Current State

`ToolExecution` component has:
```python
self._expanded = False

def set_expanded(self, expanded: bool) -> None:
    self._expanded = expanded
    self._rebuild_display()
```

But **nothing calls `set_expanded()`** - no keybind, no toggle mechanism.

#### Proposed: Global Toggle with Ctrl+O

Following pi-mono's pattern:

| Key | Action |
|-----|--------|
| `Ctrl+O` | Toggle all expandable content (tool outputs, system prompt) |

Implementation:

```python
# In AgentRenderer or TUI
class AgentRenderer:
    def __init__(self, ...):
        self.global_expanded: bool = False
        self.expandable_components: list[Component] = []  # Track all expandable items
    
    def toggle_expansion(self) -> None:
        """Toggle global expansion state."""
        self.global_expanded = not self.global_expanded
        for component in self.expandable_components:
            if hasattr(component, 'set_expanded'):
                component.set_expanded(self.global_expanded)
        self.tui.request_render()
```

Wire keybind in TUI input handler:
```python
# In TUI._handle_key()
if key == '\x0f':  # Ctrl+O
    if hasattr(self, 'renderer'):
        self.renderer.toggle_expansion()
    return
```

#### Collapsed vs Expanded Display

**System prompt collapsed:**
```
┌─ System (1,247 tokens) ──────────────────────────
│ coding env • 4 tools • 2 context files • Ctrl+O to expand
└──────────────────────────────────────────────────
```

**System prompt expanded:**
```
┌─ System (1,247 tokens) ──────────────────────────
│ You are a coding assistant with access to...
│ [full content]
└──────────────────────────────────────────────────
```

**Tool output collapsed:**
```
☺ bash $ ls -la → 15 lines
```

**Tool output expanded:**
```
☺ bash $ ls -la
  total 24
  drwxr-xr-x  5 user staff  160 Dec 25 10:00 .
  ...
```

### New Component: SystemMessage

```python
class SystemMessage(Container):
    """Renders system prompt with expand/collapse support."""
    
    def __init__(self, content: str, theme: Theme):
        self._content = content
        self._expanded = False
        self._theme = theme
        self._token_count = self._estimate_tokens(content)
        self._rebuild_display()
    
    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._rebuild_display()
    
    def _rebuild_display(self) -> None:
        self.clear()
        
        if self._expanded:
            # Show full content
            text = self._content
        else:
            # Show summary
            text = self._build_summary()
        
        self.add_child(Text(
            text,
            gutter_prefix="⚙ ",
            custom_bg_fn=self._theme.system_message_bg_fn,
        ))
    
    def _build_summary(self) -> str:
        # Parse content to extract key info
        # "coding env • 4 tools • 2 context files • Ctrl+O to expand"
        ...
```

### Implementation Plan

#### Phase 1: Global toggle infrastructure
- [ ] Add `global_expanded` state to AgentRenderer
- [ ] Track expandable components in a list
- [ ] Add `toggle_expansion()` method
- [ ] Wire `Ctrl+O` keybind in TUI

#### Phase 2: System prompt rendering  
- [ ] Create `SystemMessage` component
- [ ] Add collapsed summary format
- [ ] Add to AgentRenderer at session start
- [ ] Register as expandable component

#### Phase 3: Tool output toggle
- [ ] Ensure ToolExecution components register as expandable
- [ ] Verify `set_expanded` triggers re-render

### Status Line Update

Show current expansion state:
```
coding • claude-opus-4 • [collapsed] • session_abc123
```

Or:
```
coding • claude-opus-4 • [expanded] • session_abc123
```
