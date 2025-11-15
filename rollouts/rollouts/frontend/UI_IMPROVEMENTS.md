# UI Improvements - User Feedback

## Layout & Interaction Model

### Sidebar Behavior (HIGH PRIORITY)
**Current**: Sidebars dim/block main pane when open
**Target**: Notion-style "side page" - sidebars don't stop interaction with main pane
- User can still scroll and interact with main pane while sidebars are open
- Remove overlay dimming effect
- Sidebars push content aside rather than overlaying
- Think: opening a database in Notion - you can still work on the main page

**Implementation**:
- Remove `overlay` element and dimming behavior
- Change sidebar positioning from `position: absolute` to part of flex layout
- Main pane should remain fully interactive when sidebars are open
- Sidebars slide in/out but don't create modal-like blocking

---

## Config Sidebar Improvements

### 1. Launch Button Placement
**Current**: Launch button in top bar
**Target**: Move Launch button to config sidebar action buttons area
- Place alongside Validate and Save Config buttons
- Order: `[Validate] [Save Config] [Launch]`
- Launch should launch the current active config being edited

**Rationale**: Launch is a config-specific action, not a global action

---

### 2. GPU ID Selection
**Current**: Number input with up/down arrows
**Target**: Visual button grid (0-7)

```
GPU ID:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
└───┴───┴───┴───┴───┴───┴───┴───┘
```

**Design**:
- 8 small square buttons in a row
- Active button highlighted with accent color
- Click to select
- Much faster than typing/incrementing

---

### 3. Sample Range Input
**Current**:
- Range slider
- Goes to infinity
- Unclear bounds

**Target**: Manual start/end index inputs with dataset bounds
- Read dataset file to get actual length
- Show bounds: `0 ≤ start < end ≤ {dataset_length}`
- Two number inputs side by side
- Display dataset length: "Dataset: 150 samples"
- Validate that start < end and both within dataset bounds

**Example**:
```
Sample Range (Dataset: 150 samples)
┌─────────────┬─────────────┐
│ Start: [0 ] │ End: [10  ] │
└─────────────┴─────────────┘
0 ≤ start < end ≤ 150
```

**Remove**: Range slider (not needed)

---

### 4. Seed Input
**Current**: Number input (correct)
**Target**: Keep as-is, but clarify it's manual input only
- Remove up/down increment arrows via CSS
- Just a text field for number entry
- Placeholder: "42 (default)"

---

### 5. Max Turns Input
**Current**: Number input with up/down arrows
**Target**: Manual text input with -1 for unlimited
- Remove increment arrows
- Placeholder: "5 (or -1 for unlimited)"
- Validation: Must be integer ≥ -1
- Show help text: "-1 = no limit"

---

## Tool Builder Improvements (MAJOR CHANGE)

### Current Flow Issues
- User must write valid JSON for parameters
- No connection to actual tool execution code
- No validation of parameter schemas

### New Tool Builder Flow

**Step 1: Tool Metadata**
```
Tool Name: [read_file_______]
Description: [Read contents of a file from the remote environment]
```

**Step 2: Parameter Builder (Form, not JSON)**
```
Parameters:
┌─────────────────────────────────────────────────┐
│ ┌─ Parameter 1 ─────────────────────────────┐  │
│ │ Name: [filePath]                          │  │
│ │ Type: [String ▼]                          │  │
│ │      String / Number / Boolean / Object   │  │
│ │ Description: [Path to file to read]       │  │
│ │ Required: [✓]                             │  │
│ │ [Remove Parameter]                        │  │
│ └───────────────────────────────────────────┘  │
│                                                 │
│ [+ Add Parameter]                               │
└─────────────────────────────────────────────────┘
```

**Detect OpenAI vs Anthropic Style**:
- Let user pick format or auto-detect from selected model
- Show preview of generated tool schema
- Support both formats internally

**Step 3: Tool Implementation (Python Function)**
```
Tool Implementation:
┌─────────────────────────────────────────────────┐
│ def execute_read_file(filePath: str) -> str:   │
│     """Read file from remote environment."""   │
│     # User writes implementation here           │
│     with open(filePath, 'r') as f:             │
│         return f.read()                         │
│                                                 │
└─────────────────────────────────────────────────┘

Preview Generated Tool:
┌─────────────────────────────────────────────────┐
│ Tool(                                           │
│   type="function",                              │
│   function=ToolFunction(                        │
│     name="read_file",                           │
│     description="Read contents of a file...",   │
│     parameters={                                │
│       "type": "object",                         │
│       "properties": {                           │
│         "filePath": {                           │
│           "type": "string",                     │
│           "description": "Path to file..."      │
│         }                                       │
│       },                                        │
│       "required": ["filePath"]                  │
│     }                                           │
│   ),                                            │
│   execute=execute_read_file                     │
│ )                                               │
└─────────────────────────────────────────────────┘
```

**Benefits**:
- No JSON writing required
- Visual form for parameters
- Direct Python code for execution
- Preview shows final generated tool
- Syntax highlighting for Python code

---

## Model Selection Improvements

### Current
- Hardcoded model dropdown

### Target: Query Live API Models
**Query endpoints**:
- OpenAI: `GET https://api.openai.com/v1/models`
- Anthropic: List of known models (or API if available)

**Display format**:
```
Model: [gpt-4o-mini ▼]
       gpt-4o-mini        ($0.150 / $0.600 per 1M tokens)
       gpt-4o             ($2.50 / $10.00 per 1M tokens)
       claude-3.5-sonnet  ($3.00 / $15.00 per 1M tokens)
       claude-3.5-haiku   ($0.80 / $4.00 per 1M tokens)
```

**Show costs**:
- Input cost / Output cost per 1M tokens
- Helps user make informed decisions
- Update costs from API or hardcoded pricing table

**Implementation**:
- Cache model list for 1 hour
- Show loading state while fetching
- Fallback to hardcoded list if API fails
- Group by provider (OpenAI / Anthropic)

---

## Dataset Path Improvements

### Current
- Text input for path

### Target: File Explorer Dropdown
**Behavior**:
- Click input → Opens file picker modal
- Start at project root or `datasets/` folder
- Browse directory tree
- Filter: Show only `.json`, `.jsonl` files
- Click file to select
- Show full path in input field

**Design**:
```
Dataset Path: [datasets/nvfp4_matmul.json] [Browse...]

[Browse...] opens:
┌─ Select Dataset ──────────────────┐
│ 📁 datasets/                      │
│   📄 nvfp4_matmul.json      (150) │
│   📄 calculator_tasks.json  (50)  │
│   📄 coding_eval.jsonl      (200) │
│                                   │
│ 📁 custom_data/                   │
│   📄 my_dataset.json        (75)  │
│                                   │
│ [Cancel] [Select]                 │
└───────────────────────────────────┘
```

**Show sample count** next to each file (read file, count samples)

**Validation**:
- File must exist
- File must be valid JSON/JSONL
- Show error if file is malformed

---

## Summary of Changes

### High Priority (User Experience)
1. **Sidebar non-blocking behavior** - Notion-style interaction
2. **Tool builder parameter form** - No JSON writing
3. **Tool implementation editor** - Python function input

### Medium Priority (Polish)
4. **GPU ID button grid** - Visual selection
5. **Dataset file picker** - Browse instead of type
6. **Model dropdown with costs** - Query live APIs
7. **Launch button placement** - Move to config sidebar

### Low Priority (Nice to Have)
8. **Sample range bounds** - Read from dataset
9. **Max turns -1 support** - Unlimited turns
10. **Better input styling** - Remove increment arrows

---

## Implementation Notes

### Tool Builder Refactor
- Most complex change
- Need Python code editor (CodeMirror or Monaco)
- Need parameter form builder
- Need to serialize to both OpenAI and Anthropic formats
- Need to inject user's Python function into config

### Dataset Reading
- Backend endpoint: `GET /api/dataset-info?path=...`
- Returns: `{length: 150, valid: true, format: "json"}`
- Use for validation and bounds checking

### Model API Queries
- Backend caches model lists
- Frontend shows in dropdown with costs
- Pricing table stored in backend config

### File Browser
- Backend endpoint: `GET /api/browse?path=...`
- Returns directory tree structure
- Filter by extension on frontend
- Modal component for file selection

---

## Next Steps

1. Prioritize sidebar behavior fix (biggest UX impact)
2. Implement tool builder form (removes JSON requirement)
3. Add GPU button grid (quick win)
4. Add file browser for datasets
5. Query model APIs with pricing
6. Polish inputs (remove arrows, add validation)
