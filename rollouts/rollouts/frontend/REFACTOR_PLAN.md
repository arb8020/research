# Frontend Refactor Plan - Production Readiness

## Current Status (Functional MVP ✅)

All 5 phases complete:
- ✅ Phase 1: Three-pane layout with sliding sidebars
- ✅ Phase 2: Config sidebar with full form auto-population
- ✅ Phase 3: Tool builder modal (JSON parameters)
- ✅ Phase 4: Results sidebar + trajectory viewer
- ✅ Phase 5: Launch (manual command copy)

**Working end-to-end**: User can select config → edit → add tools → save → launch manually → view results

## Refactor Priority (Bottom-Up Foundation)

### 1. State Machine Refactor (NEXT - In Progress)
**Goal**: Centralize all state updates, eliminate scattered DOM manipulation

**Current Issues**:
```javascript
// Scattered updates throughout code
state.tools.push(tool);
updateToolsList();
closeToolBuilder();

// Direct DOM manipulation mixed with state
document.getElementById('results-content').innerHTML = '...';
state.results = results;
```

**Target Architecture**:
```javascript
// Single state object
const state = {
    ui: {
        configSidebarOpen: false,
        resultsSidebarOpen: false,
        mainPaneMode: 'idle' | 'trajectory-view',
        toolBuilderOpen: false,
    },

    // Data
    configs: [],           // Available base configs
    activeConfig: {        // Current config being edited
        prefix: '03_',
        name: '',
        model: '',
        temperature: 0.2,
        prepareMessages: '',
        sshTarget: '',
        gpuId: 0,
        datasetPath: '',
        seed: 42,
        startIdx: 0,
        endIdx: 10,
        maxTurns: 5,
        tools: [],
    },

    results: [],           // Past evaluation results
    selectedResultId: null,
    selectedTrajectory: null,

    // Validation
    validationErrors: [],
};

// Single update function
function updateState(changes) {
    // Apply changes
    Object.assign(state, changes);

    // Re-render affected UI components
    render();
}

// Single render function
function render() {
    renderConfigSidebar();
    renderResultsSidebar();
    renderMainPane();
    renderToolBuilder();
}
```

**Files to Refactor**:
- `rollouts/frontend/index.html` (lines 830-1400, all JavaScript)

**Tasks**:
1. [ ] Create proper state object with nested structure
2. [ ] Write `updateState()` function
3. [ ] Write component render functions:
   - [ ] `renderConfigSidebar()` - updates form fields from state
   - [ ] `renderResultsSidebar()` - updates result list
   - [ ] `renderMainPane()` - updates main content area
   - [ ] `renderToolBuilder()` - updates modal visibility/content
4. [ ] Refactor all event handlers to use `updateState()`
5. [ ] Remove all scattered `innerHTML` assignments
6. [ ] Test: open config → edit → add tool → save → view result

**Success Criteria**:
- All state changes go through `updateState()`
- UI always reflects current state
- No direct DOM manipulation outside render functions
- State can be logged/inspected at any time

---

### 2. Backend System Design (NEXT - After State Refactor)
**Goal**: Add proper background execution, job queue, live streaming

**Current Issues**:
- Manual copy-paste launch command
- No background execution
- No live progress updates
- Results appear only after completion

**Target Architecture**:

#### Backend Components

**New API Endpoints**:
```python
# rollouts/frontend/server.py

POST /api/launch
  Body: {config_name: "03_my_experiment"}
  Returns: {run_id: "uuid-1234", status: "queued"}

  - Validates config exists
  - Creates run record in memory/DB
  - Spawns subprocess: python entrypoint.py configs/{name}.py
  - Returns run_id immediately

GET /api/runs
  Returns: [{run_id, config_name, status, started_at}, ...]

  - Lists all runs (queued, running, completed, failed)

GET /api/runs/:run_id
  Returns: {run_id, status, progress, logs, error}

  - Status: "queued" | "running" | "completed" | "failed"
  - Progress: {current_sample: 3, total_samples: 10}

GET /api/stream/:run_id (Server-Sent Events)
  Stream: data: {"type": "progress", "sample": 3, "total": 10}
          data: {"type": "log", "message": "Starting sample 3..."}
          data: {"type": "complete", "result_id": "..."}

  - Real-time updates as evaluation runs
  - Client reconnects if connection drops

POST /api/stop/:run_id
  Returns: {success: true}

  - Gracefully terminates subprocess
  - Marks run as "stopped"
```

**Backend State Management**:
```python
# In-memory for now, can move to DB later
running_processes = {}  # run_id -> subprocess.Popen

class RunState:
    run_id: str
    config_name: str
    status: str  # queued, running, completed, failed
    subprocess: Popen | None
    started_at: float
    completed_at: float | None
    result_id: str | None  # Points to results/ directory
    error: str | None
```

**Subprocess Management**:
```python
def launch_evaluation(config_name: str) -> str:
    """Launch evaluation in background subprocess."""
    run_id = generate_uuid()

    # Spawn subprocess
    cmd = ["python", "entrypoint.py", f"configs/{config_name}.py"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=project_root,
        text=True
    )

    # Track state
    running_processes[run_id] = RunState(
        run_id=run_id,
        config_name=config_name,
        status="running",
        subprocess=process,
        started_at=time.time()
    )

    # Monitor in background thread
    threading.Thread(target=monitor_run, args=(run_id,)).start()

    return run_id

def monitor_run(run_id: str):
    """Monitor subprocess and update state."""
    state = running_processes[run_id]

    # Stream output
    for line in state.subprocess.stdout:
        # Send to SSE clients
        broadcast_sse(run_id, {"type": "log", "message": line})

        # Parse progress if possible
        if "sample" in line:
            # Extract progress info
            broadcast_sse(run_id, {"type": "progress", ...})

    # Wait for completion
    returncode = state.subprocess.wait()

    if returncode == 0:
        # Find result in results/ directory
        result_id = find_latest_result(state.config_name)
        state.status = "completed"
        state.result_id = result_id
        broadcast_sse(run_id, {"type": "complete", "result_id": result_id})
    else:
        state.status = "failed"
        state.error = "Process exited with code " + str(returncode)
        broadcast_sse(run_id, {"type": "error", "error": state.error})

    state.completed_at = time.time()
```

**SSE Implementation**:
```python
# Server-Sent Events for live updates
sse_clients = {}  # run_id -> [response_queue, ...]

def stream_run(run_id: str):
    """SSE endpoint for streaming run updates."""
    queue = queue.Queue()
    sse_clients.setdefault(run_id, []).append(queue)

    def generate():
        try:
            while True:
                message = queue.get(timeout=30)  # 30s keepalive
                yield f"data: {json.dumps(message)}\n\n"
        except queue.Empty:
            yield ": keepalive\n\n"
        finally:
            sse_clients[run_id].remove(queue)

    return Response(generate(), mimetype='text/event-stream')

def broadcast_sse(run_id: str, message: dict):
    """Send message to all SSE clients for this run."""
    for queue in sse_clients.get(run_id, []):
        queue.put(message)
```

#### Frontend Integration

**New State**:
```javascript
state.runs = {
    active: null,      // Currently viewing run_id
    list: [],          // All runs
    streaming: false,  // EventSource connected
};
```

**Launch Flow**:
```javascript
async function launchAgent() {
    // Validate config
    if (!validateConfig()) return;

    // Get config name
    const configName = state.activeConfig.prefix + state.activeConfig.name;

    // Launch
    const response = await fetch('/api/launch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({config_name: configName})
    });

    const {run_id} = await response.json();

    // Switch to live stream view
    updateState({
        ui: {mainPaneMode: 'live-stream'},
        runs: {active: run_id}
    });

    // Connect to SSE stream
    connectToRunStream(run_id);
}

function connectToRunStream(run_id) {
    const eventSource = new EventSource(`/api/stream/${run_id}`);

    eventSource.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === 'progress') {
            updateState({
                runs: {
                    progress: {
                        current: message.sample,
                        total: message.total
                    }
                }
            });
        } else if (message.type === 'log') {
            // Append to log display
            appendLog(message.message);
        } else if (message.type === 'complete') {
            // Reload results, show completion
            loadResults();
            eventSource.close();
        }
    };

    eventSource.onerror = () => {
        // Handle disconnect, maybe reconnect
        console.error('SSE connection lost');
    };
}
```

**Live Stream UI** (in main pane):
```html
<div class="live-stream">
    <div class="live-stream-header">
        <h2>Running: {config_name}</h2>
        <button onclick="stopRun()">Stop</button>
    </div>

    <div class="progress-bar">
        <div class="progress-fill" style="width: {percent}%"></div>
        <span>{current} / {total} samples</span>
    </div>

    <div class="live-log">
        <!-- Streaming log output -->
    </div>
</div>
```

**Files to Modify**:
- Backend: `rollouts/frontend/server.py`
- Frontend: `rollouts/frontend/index.html` (add SSE client)

**Tasks**:
1. Backend:
   - [ ] Add `/api/launch` endpoint
   - [ ] Add subprocess spawning logic
   - [ ] Add background monitoring thread
   - [ ] Add `/api/stream/:run_id` SSE endpoint
   - [ ] Add `/api/runs` list endpoint
   - [ ] Add `/api/stop/:run_id` endpoint
   - [ ] Test: launch evaluation, see output stream

2. Frontend:
   - [ ] Add live stream state to state machine
   - [ ] Add `launchAgent()` to call new API
   - [ ] Add `connectToRunStream()` SSE client
   - [ ] Add `renderLiveStream()` for main pane
   - [ ] Add progress bar component
   - [ ] Add log streaming display
   - [ ] Test: launch → see progress → see completion

**Success Criteria**:
- Click Launch → evaluation starts in background
- See live progress bar updating
- See log output streaming
- Click result when complete → shows in main pane
- Multiple evaluations can queue/run concurrently

---

### 3. Error Handling & Assertions (LATER - After Core Refactor)
**Goal**: Add Tiger Style fail-fast assertions, proper error boundaries

**Current Issues**:
- Generic `try/catch` with alerts
- No precondition checks
- Silent failures possible
- Unclear error states

**Target**:
- Assertion library or console.assert wrapper
- Assert all function preconditions
- Assert state invariants
- Explicit error UI states
- Error logging/reporting

**Defer until**: State machine + backend are solid

---

### 4. Aesthetic Polish (LATER - After Functionality Solid)
**Goal**: Distinctive visual design, animations, typography

**Current**: Clean but generic Notion-like aesthetic

**Target**:
- Custom font choices (not system fonts)
- Micro-interactions and animations
- Distinctive color palette
- Polish transitions and hover states

**Defer until**: Everything else works perfectly

---

## Collaboration Points

### Where to Start
1. **You**: Review state machine refactor plan above
2. **Me**: Implement centralized state + render functions
3. **You**: Review backend architecture for launch system
4. **Me**: Implement subprocess spawning + SSE streaming
5. **Both**: Test end-to-end, iterate

### Questions for You

**State Management**:
- Is the proposed state structure clear enough?
- Do we need immutability (frozen objects) or can we mutate state?
- Should render functions be granular (per-component) or one big render?

**Backend**:
- Should we store run state in-memory or add a DB table?
- How should we handle server restart (lose running processes)?
- Concurrent evaluation limit (prevent spawning 100 processes)?
- Should logs be streamed from stdout or from result files?

**Error Handling**:
- Assertion library preference or just console.assert?
- Error boundary strategy for async operations?
- How to show errors in UI (toast? modal? inline?)?

### Handoff Artifacts

When ready for backend work:
- [ ] State machine refactor complete
- [ ] State structure documented with types
- [ ] Frontend ready to consume new APIs
- [ ] Example SSE client code tested

When ready for polish:
- [ ] Core functionality tested and stable
- [ ] No known bugs or edge cases
- [ ] Performance acceptable
- [ ] Ready for aesthetic layer

---

## Testing Strategy

After each refactor phase:

**State Machine**:
- [ ] Open/close sidebars (UI state syncs)
- [ ] Edit config fields (state updates)
- [ ] Add/remove tools (tools array syncs)
- [ ] Validation errors display correctly
- [ ] Save config (all state captured)

**Backend Launch**:
- [ ] Launch evaluation (process spawns)
- [ ] See progress updates (SSE works)
- [ ] Complete successfully (result appears)
- [ ] Handle errors (bad config, crash)
- [ ] Stop running evaluation (graceful exit)
- [ ] Multiple concurrent runs (no conflicts)

**Error Handling**:
- [ ] Invalid config caught early
- [ ] Network errors shown to user
- [ ] Backend errors displayed properly
- [ ] No silent failures

**Polish**:
- [ ] Smooth animations
- [ ] Responsive interactions
- [ ] Pleasant visual hierarchy
- [ ] Professional feel

---

## Timeline Estimate

- **State Machine Refactor**: 2-4 hours
- **Backend System Design**: 4-8 hours
- **Error Handling**: 2-3 hours
- **Aesthetic Polish**: 2-4 hours

**Total**: ~10-20 hours to production-ready

---

## Current Code Baseline

**Commit**: `8649227` - "Phase 5: Add launch functionality (manual for now)"

**Working Features**:
- Config sidebar with full auto-population ✅
- Tool builder modal ✅
- Results sidebar with trajectory viewer ✅
- Manual launch instructions ✅

**Next Commit Target**: State machine refactor complete

**Final Commit Target**: Full live streaming launch system working
