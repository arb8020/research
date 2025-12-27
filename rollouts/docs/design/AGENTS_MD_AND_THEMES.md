# AGENTS.md Discovery & Display Themes

**Status:** Ready to implement  
**Priority:** High (useful daily)

---

## Part 1: AGENTS.md Discovery

### Overview

Auto-discover and inject project context files into the system prompt. Follows conventions from Claude Code (`CLAUDE.md`) and pi-mono (`AGENTS.md`).

### File Names (in priority order)

```
AGENTS.md      # pi-mono / agentskills.io standard
CLAUDE.md      # Claude Code compatibility
```

### Discovery Order

1. **Global**: `~/.rollouts/AGENTS.md`
2. **Ancestors**: Walk from cwd up to root, collect matching files
3. **Project root**: `.rollouts/AGENTS.md` (if exists)
4. **Current dir**: `./AGENTS.md` or `./CLAUDE.md`

Files are concatenated in this order (global first, most specific last).

### Example

```
~/.rollouts/AGENTS.md           # "Always use type hints"
~/projects/AGENTS.md            # "Python 3.11+, use uv"
~/projects/myapp/AGENTS.md      # "FastAPI app, see docs/"
~/projects/myapp/src/AGENTS.md  # "Core business logic here"
```

If cwd is `~/projects/myapp/src/`, all four files are concatenated.

### Injection

Append to system prompt after the base prompt but before any user content:

```python
system_prompt = base_system_prompt + "\n\n" + agents_md_content
```

### Implementation

```python
# rollouts/context.py

from pathlib import Path

CONTEXT_FILENAMES = ["AGENTS.md", "CLAUDE.md"]

def discover_context_files(cwd: Path) -> list[Path]:
    """Discover context files from global → ancestors → cwd."""
    files = []
    
    # 1. Global
    global_file = Path.home() / ".rollouts" / "AGENTS.md"
    if global_file.exists():
        files.append(global_file)
    
    # 2. Walk up from cwd to root
    current = cwd.resolve()
    ancestor_files = []
    while current != current.parent:
        for name in CONTEXT_FILENAMES:
            candidate = current / name
            if candidate.exists():
                ancestor_files.append(candidate)
                break  # Only one per directory
        current = current.parent
    
    # Add ancestors in root→cwd order
    files.extend(reversed(ancestor_files))
    
    # 3. Project-local (.rollouts/AGENTS.md)
    project_file = cwd / ".rollouts" / "AGENTS.md"
    if project_file.exists() and project_file not in files:
        files.append(project_file)
    
    return files


def load_context_files(cwd: Path) -> str:
    """Load and concatenate all context files."""
    files = discover_context_files(cwd)
    if not files:
        return ""
    
    sections = []
    for f in files:
        content = f.read_text().strip()
        if content:
            # Add source comment for debugging
            sections.append(f"<!-- From: {f} -->\n{content}")
    
    return "\n\n".join(sections)
```

### CLI Integration

```python
# In cli.py, when building system prompt:

from rollouts.context import load_context_files

context_content = load_context_files(working_dir)
if context_content:
    system_prompt = f"{base_prompt}\n\n{context_content}"
```

### Flags

```
--no-context       Don't load AGENTS.md/CLAUDE.md files
--show-context     Print discovered context files and exit
```

---

## Part 2: Display Themes

### Current State

The TUI has three themes: `DARK_THEME`, `MINIMAL_THEME`, `ROUNDED_THEME`. They control:
- Colors (accent, muted, borders, backgrounds)
- Gutter prefixes (☻, ☺, ☹)
- Padding (message_padding_y, tool_padding_y)
- Diff colors

Tool output is formatted by environment-specific formatters (`format_bash`, `format_read`, etc.) which receive an `expanded` bool.

### Problem

1. **No way to toggle expansion per-tool while running** - `expanded` is always False during streaming
2. **Compact mode not compact enough** - still shows multi-line tool calls
3. **No keybind to toggle** - unlike pi-mono's `Ctrl+O`

### Proposed: Compact Theme + Expansion Toggle

#### Compact Display Mode

Add a `compact` display mode that shows minimal tool output:

```
Before (current):
  ☺ bash(command='ls -la')
    ⎿ Command completed
      total 24
      drwxr-xr-x  5 user staff  160 Dec 25 10:00 .
      drwxr-xr-x 10 user staff  320 Dec 25 09:00 ..
      -rw-r--r--  1 user staff 1234 Dec 25 10:00 file.txt
      ... (2 more lines)

After (compact):
  ☺ bash $ ls -la → 5 lines
```

When expanded (Ctrl+O):
```
  ☺ bash $ ls -la
    total 24
    drwxr-xr-x  5 user staff  160 Dec 25 10:00 .
    drwxr-xr-x 10 user staff  320 Dec 25 09:00 ..
    -rw-r--r--  1 user staff 1234 Dec 25 10:00 file.txt
    -rw-r--r--  1 user staff  567 Dec 25 09:30 other.txt
```

#### Per-Tool Expansion

Track expansion state per tool_call_id:

```python
class AgentRenderer:
    def __init__(self, ...):
        ...
        self.expanded_tools: set[str] = set()  # tool_call_ids that are expanded
        self.global_expanded: bool = False     # Ctrl+O toggles all
    
    def toggle_tool_expansion(self, tool_call_id: str) -> None:
        if tool_call_id in self.expanded_tools:
            self.expanded_tools.discard(tool_call_id)
        else:
            self.expanded_tools.add(tool_call_id)
        self._rebuild_tool(tool_call_id)
    
    def toggle_global_expansion(self) -> None:
        self.global_expanded = not self.global_expanded
        for tool_id in self.pending_tools:
            self.pending_tools[tool_id].set_expanded(self.global_expanded)
```

#### Compact Formatters

```python
def format_bash_compact(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme: Theme | None = None
) -> str:
    """Compact bash format: bash $ command → N lines"""
    command = args.get("command", "")
    # Truncate command for display
    cmd_display = command[:50] + "..." if len(command) > 50 else command
    cmd_display = cmd_display.replace("\n", " ")
    
    text = f"bash $ {cmd_display}"
    
    if result:
        output = _get_text_output(result).strip()
        if output:
            lines = output.split("\n")
            is_error = result.get("isError", False)
            
            if expanded:
                # Full output
                text += "\n"
                for line in lines:
                    text += f"  {line}\n"
            else:
                # Just summary
                status = "✗" if is_error else "→"
                text += f" {status} {len(lines)} lines"
    
    return text


def format_read_compact(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme: Theme | None = None
) -> str:
    """Compact read format: 📄 path (N lines)"""
    path = _shorten_path(args.get("file_path") or args.get("path") or "")
    
    if result:
        output = _get_text_output(result)
        lines = output.split("\n")
        text = f"📄 {path} ({len(lines)} lines)"
        
        if expanded:
            text += "\n"
            for i, line in enumerate(lines[:50]):
                text += f"  {i+1:4}  {line}\n"
            if len(lines) > 50:
                text += f"  ... ({len(lines) - 50} more lines)\n"
    else:
        text = f"📄 {path}"
    
    return text


def format_edit_compact(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme: Theme | None = None
) -> str:
    """Compact edit format: ✏️ path (+N/-M lines)"""
    path = _shorten_path(args.get("file_path") or args.get("path") or "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    
    old_lines = len(old_text.split("\n"))
    new_lines = len(new_text.split("\n"))
    diff = new_lines - old_lines
    
    if diff > 0:
        diff_str = f"+{diff}"
    elif diff < 0:
        diff_str = str(diff)
    else:
        diff_str = "±0"
    
    text = f"✏️ {path} ({diff_str} lines)"
    
    if expanded and result:
        # Show full diff
        text += "\n" + _format_diff(old_text, new_text, theme)
    
    return text


def format_write_compact(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme: Theme | None = None
) -> str:
    """Compact write format: 📝 path (N lines)"""
    path = _shorten_path(args.get("file_path") or args.get("path") or "")
    content = args.get("content", "")
    lines = len(content.split("\n"))
    
    text = f"📝 {path} ({lines} lines)"
    
    if expanded:
        text += "\n"
        for i, line in enumerate(content.split("\n")[:30]):
            text += f"  {i+1:4}  {line}\n"
        if lines > 30:
            text += f"  ... ({lines - 30} more lines)\n"
    
    return text
```

#### Theme Configuration

```python
@dataclass
class Theme:
    ...
    
    # Display mode
    compact_tools: bool = False  # Use compact tool formatters
    
    # Compact format symbols
    compact_bash_prefix: str = "$ "
    compact_read_prefix: str = "📄 "
    compact_edit_prefix: str = "✏️ "
    compact_write_prefix: str = "📝 "


COMPACT_THEME = Theme(
    compact_tools=True,
    # Minimal visual noise
    tool_padding_y=0,
    message_padding_y=0,
    # Single-char gutters
    assistant_gutter="",
    tool_success_gutter="✓ ",
    tool_error_gutter="✗ ",
)
```

#### Keybinds

| Key | Action |
|-----|--------|
| `Ctrl+O` | Toggle all tool outputs expanded/collapsed |
| `Ctrl+T` | Toggle thinking blocks visible/hidden (if we add this) |

### Implementation Plan

#### Phase 1: AGENTS.md Discovery
- [ ] Add `rollouts/context.py` with discovery logic
- [ ] Wire into CLI system prompt building
- [ ] Add `--no-context` and `--show-context` flags
- [ ] Add to session info display (`/session` or status line)

#### Phase 2: Compact Formatters
- [ ] Add compact formatter variants to `coding.py`
- [ ] Add `compact_tools` flag to Theme
- [ ] Create `COMPACT_THEME` preset
- [ ] Wire formatter selection based on theme

#### Phase 3: Expansion Toggle
- [ ] Track `expanded_tools` set in AgentRenderer
- [ ] Add `Ctrl+O` keybind to TUI
- [ ] Update tool components on toggle
- [ ] Persist expansion state across re-renders

---

## Open Questions

1. **Should compact be the default?** Pi-mono shows collapsed by default. Current rollouts shows ~5 lines.

2. **Per-tool vs global toggle?** Pi-mono does global only (`Ctrl+O`). Could add click-to-expand per tool.

3. **Syntax highlighting in expanded view?** Currently no highlighting. Could add for code blocks.

4. **Context file caching?** Reload on every session or cache? Files rarely change mid-session.
