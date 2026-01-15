# Tool Result Expansion Research

Research on how different CLI coding agents implement tool result expansion/collapse.

## Summary

| Project | Framework | Expansion Mechanism | Default State | Toggle |
|---------|-----------|---------------------|---------------|--------|
| rollouts | Python TUI | `set_expanded(bool)` on `ToolExecution` | Collapsed (10 lines) | Via formatter |
| Claude Code | Node/React (Ink) | Collapsible components | Collapsed | Unknown (minified) |
| opencode | SolidJS (Kobalte) | `<Collapsible>` + `<Accordion>` | Collapsed | `stepsExpanded` prop |
| pi-mono | TypeScript TUI | `TruncatedText` component | Truncated to viewport | No toggle |
| mistral-vibe | Python Textual | `collapsed` property on `ToolResultWidget` | Collapsed | Keyboard shortcut per-tool |
| amp | Node (minified) | Unknown | Unknown | Unknown |

---

## Detailed Analysis

### 1. rollouts (Python)
**File**: `rollouts/frontends/tui/components/tool_execution.py`

**Implementation**:
- `ToolExecution` component has `_expanded: bool` property
- `set_expanded(expanded: bool)` method to toggle
- Uses formatter functions from environment for display
- Default collapsed shows first 10 lines + "... (N more lines)"

```python
def set_expanded(self, expanded: bool) -> None:
    """Set whether to show expanded output."""
    self._expanded = expanded
    self._rebuild_display()

# In _format_generic():
max_lines = len(lines) if self._expanded else 10
for line in lines[:max_lines]:
    text += "\n  " + line
remaining = len(lines) - max_lines
if remaining > 0:
    text += f"\n  ... ({remaining} more lines)"
```

**Toggle**: Via custom formatter functions passed from environment.

---

### 2. Claude Code (Node/React)
**File**: `~/.bun/install/global/node_modules/@anthropic-ai/claude-code/cli.js` (minified)

**Implementation**:
- Uses React (Ink) for terminal UI
- Source is minified/bundled - exact expansion logic not easily readable
- Grep shows references to "collaps" and "expand" patterns in the minified code
- Likely uses React state for expansion toggle

**Observations from minified code**:
- Contains Observable/RxJS patterns for state management
- Has component-based architecture

---

### 3. opencode (SolidJS)
**Files**:
- `packages/ui/src/components/collapsible.tsx`
- `packages/ui/src/components/accordion.tsx`
- `packages/ui/src/components/session-turn.tsx`
- `packages/ui/src/components/basic-tool.tsx`

**Implementation**:
Uses Kobalte's primitives for accessibility-friendly collapsible/accordion:

```tsx
// Collapsible wrapper
<Collapsible open={open()} onOpenChange={setOpen}>
  <Collapsible.Trigger>
    <Icon name={props.icon} />
    {props.trigger}
    <Collapsible.Arrow />
  </Collapsible.Trigger>
  <Collapsible.Content>{props.children}</Collapsible.Content>
</Collapsible>

// Session turn with steps expansion
<SessionTurn
  stepsExpanded={props.stepsExpanded}
  onStepsExpandedToggle={...}
/>
```

**Features**:
- `stepsExpanded` boolean prop controls all tool steps visibility
- Individual tool results use `<BasicTool>` with `defaultOpen` and `forceOpen` props
- Accordion for multiple tools in a turn
- "Show steps" / "Hide steps" button toggle

---

### 4. pi-mono (TypeScript)
**Files**:
- `packages/tui/src/tui.ts`
- `packages/tui/src/components/truncated-text.ts`
- `packages/coding-agent/src/core/messages.ts`

**Implementation**:
Simple truncation without expand/collapse:

```typescript
class TruncatedText implements Component {
  render(width: number): string[] {
    // Truncate text to fit viewport width
    const displayText = truncateToWidth(singleLineText, availableWidth);
    // ...
  }
}
```

**Message compaction**:
```typescript
const COMPACTION_SUMMARY_PREFIX = `The conversation history before this point was compacted into the following summary:
<summary>
`;
```

**Features**:
- Truncates to viewport width, no expand/collapse toggle
- Has compaction/summarization for context management (different from display expansion)

---

### 5. mistral-vibe (Python/Textual)
**Files**:
- `vibe/cli/textual_ui/widgets/tools.py`
- `vibe/cli/textual_ui/widgets/tool_widgets.py`
- `vibe/core/tools/ui.py`

**Implementation**:
Uses Textual framework with per-tool keyboard shortcuts:

```python
class ToolResultWidget[TResult: BaseModel](Static):
    SHORTCUT = DEFAULT_TOOL_SHORTCUT  # 'e' by default

    def __init__(self, result, success, message, collapsed=True, warnings=None):
        self.collapsed = collapsed
        # ...

    def _hint(self) -> str:
        action = "expand" if self.collapsed else "collapse"
        return f"({self.SHORTCUT} to {action})"

    def compose(self) -> ComposeResult:
        if self.collapsed:
            yield Static(f"{self.message} {self._hint()}", markup=False)
        else:
            yield Static(self.message, markup=False)
            # ... render full content
```

**Features**:
- `collapsed: bool` property per widget
- Keyboard shortcuts configurable per tool type
- Shows hint text "(e to expand)" / "(e to collapse)"
- Different widgets for different tool types (BashResultWidget, WriteFileResultWidget, etc.)

---

### 6. amp (Sourcegraph)
**File**: `/opt/homebrew/lib/node_modules/@sourcegraph/amp/dist/main.js` (minified)

**Implementation**:
- Source is heavily minified, ~6.9MB bundle
- Uses Observable patterns (similar to RxJS)
- Cannot determine expansion implementation from minified code

---

## Key Patterns

### State Management
1. **Boolean flag**: All implementations use a simple `expanded/collapsed` boolean
2. **Per-component state**: Each tool result manages its own expansion state
3. **Hierarchical state**: opencode has both per-tool and per-turn expansion

### Toggle Mechanisms
1. **Keyboard shortcuts**: mistral-vibe uses per-tool shortcuts
2. **Click/button**: opencode uses clickable triggers
3. **Props/external control**: rollouts allows formatter functions to control display

### Display Patterns
1. **Truncated preview**: Show first N lines + "... (X more)"
2. **Summary + expand**: Show summary text, expand for details
3. **Hint text**: Show keyboard shortcut or action hint

### Framework Choices
- **Python**: Textual (mistral-vibe) or custom (rollouts)
- **JavaScript**: React/Ink (Claude Code), SolidJS (opencode)
- **TypeScript**: Custom rendering (pi-mono)

---

## Recommendations for Implementation

1. **Collapsed by default** - All implementations start collapsed
2. **Show hint about how to expand** - Important for discoverability
3. **Show preview content** - First few lines or summary
4. **Keyboard shortcut** - Quick toggle without clicking
5. **Per-tool vs global toggle** - Consider both like opencode
6. **Persist expansion state** - Across renders (use React/SolidJS state)
