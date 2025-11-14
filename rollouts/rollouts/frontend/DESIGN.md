# Agent Dev Loop Frontend - Design Specification

## Vision
A **live agent development environment** for iterating on rollouts-based agents. Not a static config builder, but an IDE-like experience for agent development.

## Layout Architecture

### Top Bar
```
┌─────────────────────────────────────────────────────┐
│  [≡ Config] [▶ Launch]           [Results ▼]       │
└─────────────────────────────────────────────────────┘
```

- `[≡ Config]` - Toggle config sidebar (left)
- `[▶ Launch]` - Start evaluation (validates first)
- `[Results ▼]` - Toggle results sidebar (right)

### Three Layout States

**1. Default (no sidebars)**
```
┌─────────────────────────────────────────────────────┐
│  [≡ Config] [▶ Launch]           [Results ▼]       │
├─────────────────────────────────────────────────────┤
│                                                     │
│               Main Pane                             │
│   "Select a result or launch an agent"             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**2. Config sidebar open**
```
┌──────────┬──────────────────────────────────────────┐
│          │  [≡ Config] [▶ Launch]    [Results ▼]   │
│          ├──────────────────────────────────────────┤
│  Config  │                                          │
│ Sidebar  │         Main Pane (dimmed)              │
│  (left)  │                                          │
│          │                                          │
└──────────┴──────────────────────────────────────────┘
```

**3. Results sidebar open**
```
┌──────────────────────────────────────────┬──────────┐
│  [≡ Config] [▶ Launch]    [Results ▼]   │          │
├──────────────────────────────────────────┤ Results  │
│                                          │ Sidebar  │
│         Main Pane                        │ (right)  │
│   (shows selected trace or live run)     │          │
│                                          │          │
└──────────────────────────────────────────┴──────────┘
```

## Main Pane Behavior

### During Live Run

**Top-level progress:**
```
┌─────────────────────────────────────────┐
│ Progress: [████████░░] 8/10 samples    │
└─────────────────────────────────────────┘
```

**Per-sample blocks (concurrent samples):**
```
┌─ Sample 1 ─────────────────────────────┐
│ Turn 2/5                     [▼ Expand]│
│                                         │
│ ┌─ user ─────────────────────┐         │
│ │ Write a matmul kernel...   │         │
│ └────────────────────────────┘         │
│                                         │
│ ┌─ assistant ─────────────────┐        │
│ │ Here's my implementation... │ [+]    │
│ │ (streaming text here...)    │        │
│ └─────────────────────────────┘        │
└─────────────────────────────────────────┘

┌─ Sample 2 ─────────────────────────────┐
│ Turn 1/5                     [▼ Expand]│
│ (collapsed - click to expand)          │
└─────────────────────────────────────────┘
```

**Message block behavior:**
- Each message is a collapsible/expandable box
- Streaming content appears in real-time
- Click `[+]` to expand full message
- User can scroll through multiple samples
- Each sample can be independently expanded/collapsed

### Viewing Old Trajectory

**Same format, but static:**
- No streaming
- Shows completed conversation
- Metrics displayed at top
- All messages available for expansion

## Config Sidebar (Left)

### Structure

**Tab bar at top:**
```
┌─────────────────────────────────────────┐
│ [02_agent] [03_custom +] [+ New]       │
├─────────────────────────────────────────┤
│ Config content (scrollable)             │
└─────────────────────────────────────────┘
```

**Multiple configs:**
- Tabs for each open config
- Scrollable if many tabs
- `[+ New]` to create new config
- Click tab to switch
- `[x]` on tab to close (with unsaved warning)

### Config Content

**Validation status at top:**
```
┌─────────────────────────────────────────┐
│ ⚠️ Validation Errors:                   │
│   • Tool name must use underscores      │
│   • SSH target format invalid           │
│                                         │
│ [Validate] [Create] (greyed until valid)│
├─────────────────────────────────────────┤
```

**Form sections:**

**1. Base Config**
```
Base Config: [02_agent_multiturn ▼]
Config Name: [03_my_experiment___]
```

**2. Model Settings**
```
Model: [gpt-4.1-mini ▼]
Temperature: [0.2]
System Prompt:
┌────────────────────────────────────┐
│ You are an expert GPU developer... │
│                                    │
└────────────────────────────────────┘
```

**3. Environment Settings**
```
SSH Target: [ubuntu@150.136.217.70:22]
GPU ID: [0]
Dataset: [datasets/nvfp4_matmul.json]
```

**4. Sampling Settings**
```
Seed: [42___]
Range: [0]════════════[100]
       └─ start_idx    end_idx ─┘
Max Turns: [5]
```

**5. Tools**
```
┌─ Tools ─────────────────────────────────┐
│                                         │
│ ┌─ read_file ──────────────────┐  [✎]  │
│ │ Read contents of a file      │       │
│ │ Parameters: filePath (str)   │       │
│ └──────────────────────────────┘       │
│                                         │
│ [+ Add Tool]                            │
└─────────────────────────────────────────┘
```

**Tool builder (inline, expands when adding/editing):**
```
┌─ Add Tool ──────────────────────────────┐
│ Name: [read_file_____________]          │
│ Description:                            │
│ [Read contents of a file___________]    │
│                                         │
│ Parameters:                             │
│ ┌─────────────────────────────────────┐ │
│ │ Name: filePath                      │ │
│ │ Type: [string ▼]                    │ │
│ │ Description: Path to file           │ │
│ │ Required: [✓]                       │ │
│ │ [Remove]                            │ │
│ ├─────────────────────────────────────┤ │
│ │ [+ Add Parameter]                   │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ Implementation:                         │
│ ○ Auto (return success)                 │
│ ○ Paste Python code                     │
│                                         │
│ Generated Code Preview:                 │
│ ┌─────────────────────────────────────┐ │
│ │ Tool(                               │ │
│ │   type="function",                  │ │
│ │   function=ToolFunction(            │ │
│ │     name="read_file",               │ │
│ │     ...                             │ │
│ │   )                                 │ │
│ │ )                                   │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ [Save Tool] [Cancel]                    │
└─────────────────────────────────────────┘
```

**Tool code paste option:**
```
Implementation:
○ Auto (return success)
● Paste Python code

Paste tool implementation:
┌─────────────────────────────────────────┐
│ async def exec_tool(self, ...):         │
│     if tool_call.name == "read_file":   │
│         path = tool_call.args["path"]   │
│         content = Path(path).read_text()│
│         return ToolResult(...)          │
│                                         │
│ ✓ Syntax valid                          │
└─────────────────────────────────────────┘
```

**6. Environment Logic**
```
┌─ Environment Pattern ────────────────────┐
│ ○ Tool-based (tools handle everything)  │
│ ● Code block extraction                 │
│ ○ JSON response parsing                 │
│ ○ Custom                                │
│                                         │
│ For code block extraction:              │
│ Languages: [✓] Python  [✓] Triton       │
│                                         │
│ Generated on_assistant_message():       │
│ ┌───────────────────────────────────┐   │
│ │ # Auto-generated template         │   │
│ │ code_blocks = extract_code_blocks(│   │
│ │     message.content,              │   │
│ │     languages=["python", "triton"]│   │
│ │ )                                 │   │
│ │ # TODO: Process code              │   │
│ └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Config Immutability

**Once deployed (launched):**
- Config becomes read-only
- Badge shows `[Deployed]` next to name
- To modify: Click `[Copy]` → Creates new config with incremented name
- Original config preserved for reproducibility

## Results Sidebar (Right)

### Structure

**Grouped by config/date:**
```
┌─ Results ───────────────────────────────┐
│                                         │
│ ▼ 02_agent_multiturn                    │
│   ┌─ 2025-11-14 ─────────────┐          │
│   │ 14:30 ● Running          │          │
│   │   [View Live Stream]     │          │
│   ├──────────────────────────┤          │
│   │ 12:15 ✓ Completed        │ [+]     │
│   │   Mean: 0.75  (10 samp)  │          │
│   └──────────────────────────┘          │
│                                         │
│ ▼ 03_my_experiment                      │
│   ┌─ 2025-11-13 ─────────────┐          │
│   │ 18:22 ✓ Completed        │ [+]     │
│   │   Mean: 0.82  (20 samp)  │          │
│   └──────────────────────────┘          │
│                                         │
└─────────────────────────────────────────┘
```

**Expandable quick metrics:**
```
┌─────────────────────────────────────────┐
│ 12:15 ✓ Completed              [▼]     │
│ ┌───────────────────────────────────┐   │
│ │ Mean Reward: 0.75                 │   │
│ │ Min: 0.25  Max: 1.00             │   │
│ │ Samples: 10                       │   │
│ │ Avg Turns: 3.2                   │   │
│ │                                   │   │
│ │ [View in Main Pane]              │   │
│ └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**Click behavior:**
- `[View Live Stream]` or `[View in Main Pane]` → Opens in main pane
- Main pane switches from live run to trajectory viewer
- Results sidebar stays open for comparison

## Validation System

### Config Validation

**Tool validation:**
- Name must be valid Python identifier (`/^[a-z_][a-z0-9_]*$/i`)
- Description non-empty
- Parameter names valid identifiers
- Parameter types from allowed list: `string`, `number`, `boolean`, `object`, `array`
- Required fields actually exist in parameters

**Environment validation:**
- SSH target format: `user@host:port` (warn if not reachable)
- Dataset path exists (warn if missing)
- GPU ID is integer >= 0

**Sampling validation:**
- Seed is integer
- start_idx < end_idx
- Max turns >= 1

**Display errors:**
```
⚠️ Validation Errors:
  • Tool 'my-tool': Use underscores, not dashes
  • Parameter 'file-path': Invalid name format
  • Dataset 'bad/path.json': File not found (warning)
```

### Code Validation

**For pasted tool code:**
- Run `compile(code, '<tool>', 'exec')` in Python
- Show syntax errors inline
- Highlight code with syntax highlighting

**For on_assistant_message:**
- Same compile check
- Show generated code preview
- Allow editing generated code

## Templates System

**"Templates" are just base configs:**
- Creating new config = copying existing config
- Dropdown shows all available configs
- No separate "template" concept
- User can designate favorite configs as "starter templates" (future)

## Launch Flow

1. User clicks `[▶ Launch]`
2. Frontend validates config
3. If errors: Show in config sidebar, don't launch
4. If valid:
   - Mark config as `[Deployed]` (immutable)
   - Send to backend
   - Main pane switches to live stream view
   - Results sidebar auto-opens showing "Running..."
   - Progress bar appears
5. Stream results in real-time
6. On completion: Update results sidebar with final metrics

## Technical Implementation Notes

### State Management

**State machine:**
```javascript
const state = {
  // UI state
  ui: {
    configSidebarOpen: false,
    resultsSidebarOpen: false,
    mainPaneMode: 'idle' | 'live-stream' | 'trajectory-view',
  },

  // Configs (multiple tabs)
  configs: [
    {
      id: 'uuid',
      name: '02_agent_multiturn',
      isDirty: false,
      isDeployed: false,
      baseName: '01_agent_eval',
      model: '...',
      tools: [...],
      // ... all config fields
    }
  ],
  activeConfigId: 'uuid',

  // Results
  results: [...],
  selectedResultId: null,

  // Live run
  liveRun: {
    configId: 'uuid',
    samples: [
      {
        id: 1,
        turn: 2,
        maxTurns: 5,
        messages: [...],
        isExpanded: true,
      }
    ],
    progress: { completed: 8, total: 10 }
  }
}
```

### Backend APIs

**New endpoints needed:**
- `POST /api/validate` - Validate config
- `POST /api/launch` - Start evaluation (returns run ID)
- `GET /api/stream/:runId` - SSE endpoint for live updates
- `POST /api/stop/:runId` - Stop running evaluation

### Frontend Components

**Major components:**
- `ConfigSidebar` - Tabbed config editor
- `ToolBuilder` - Inline tool form builder
- `ResultsSidebar` - Grouped results list
- `MainPane` - Switches between idle/live/trajectory
- `LiveStreamView` - Real-time sample blocks
- `TrajectoryView` - Static conversation viewer
- `MessageBlock` - Expandable message component

## Open Questions

1. **Concurrent evaluation runs:**
   - Allow multiple simultaneous launches?
   - How to switch between live streams?
   - Show all in main pane (tabs?) or queue?

2. **Config persistence:**
   - Auto-save drafts to browser localStorage?
   - Sync configs to server (save to configs/ directory)?
   - Export/import configs?

3. **Comparison mode:**
   - Side-by-side trajectory comparison?
   - Diff two config results?
   - Metric plots over time?

4. **Tool testing:**
   - Dry-run tool before saving?
   - Mock tool execution in UI?
   - Tool debugging/logging?

5. **Advanced features:**
   - Dataset preview in config?
   - SSH connection testing?
   - GPU availability checking?
   - Config templates library (community-shared)?

## Next Steps

1. ✅ Get design approval
2. Build config sidebar with tabs
3. Build tool builder (inline form)
4. Build live stream main pane
5. Build results sidebar
6. Wire up validation
7. Implement launch flow
8. Add SSE streaming
9. Polish & test

---

*This design follows:*
- **Progressive disclosure** (Notion/Cursor philosophy)
- **Compression-oriented** (start specific, compress later)
- **Explicit state** (no hidden magic)
- **User control** (immutability, validation, previews)
