# Handoff: Live Streaming for Launch and Result Display

## Current State

### What Works ✓
- **Launch button** auto-saves config and executes in background
- Config validation before launch
- Process runs detached via `subprocess.Popen()`
- Results sidebar polls and shows completed runs
- Clean UI with proper loading states

### What's Missing ✗
- **No live streaming** - users can't see progress while run is executing
- **No process tracking** - frontend doesn't know if a run is currently executing
- **No real-time logs** - stdout/stderr is captured but not displayed
- **Result display issues** - need to verify/fix how results are shown

## Task 1: Implement Live Streaming for Launch

### Backend Changes Needed

#### 1. Process Registry
**File**: `rollouts/frontend/server.py`

Add a global registry to track running processes:

```python
# At module level
_active_runs = {}  # run_id -> {process, config_name, start_time, status}
_run_counter = 0
```

#### 2. Modify `_launch_config()` to Track Runs
**File**: `rollouts/frontend/server.py:1046`

Instead of fire-and-forget, store the process:

```python
def _launch_config(self):
    global _run_counter, _active_runs

    # ... existing validation ...

    # Generate unique run ID
    _run_counter += 1
    run_id = f"run_{_run_counter}_{int(time.time())}"

    # Launch with tracked pipes
    process = subprocess.Popen(
        command,
        cwd=self.project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        start_new_session=True
    )

    # Store in registry
    _active_runs[run_id] = {
        "process": process,
        "config_name": config_name,
        "start_time": time.time(),
        "status": "running"
    }

    self._json_response({
        "success": True,
        "run_id": run_id,
        "command": " ".join(command)
    })
```

#### 3. Add SSE Streaming Endpoint
**File**: `rollouts/frontend/server.py` (add to `do_GET()`)

```python
elif path.startswith("/api/stream/"):
    run_id = path.split("/api/stream/")[1]
    self._stream_run_output(run_id)
```

Add the streaming method:

```python
def _stream_run_output(self, run_id: str):
    """Stream output from a running process via SSE."""
    if run_id not in _active_runs:
        self.send_error(404, f"Run not found: {run_id}")
        return

    run_data = _active_runs[run_id]
    process = run_data["process"]

    # Set up SSE headers
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-cache")
    self.send_header("Connection", "keep-alive")
    self.end_headers()

    try:
        # Stream output line by line
        for line in iter(process.stdout.readline, ''):
            if not line:
                break

            # Send as SSE event
            event_data = json.dumps({"line": line.rstrip(), "type": "stdout"})
            self.wfile.write(f"data: {event_data}\n\n".encode())
            self.wfile.flush()

        # Wait for process to complete
        exit_code = process.wait()

        # Send completion event
        completion_data = json.dumps({
            "type": "complete",
            "exit_code": exit_code,
            "status": "success" if exit_code == 0 else "failed"
        })
        self.wfile.write(f"data: {completion_data}\n\n".encode())
        self.wfile.flush()

        # Update registry
        run_data["status"] = "completed" if exit_code == 0 else "failed"
        run_data["exit_code"] = exit_code

    except Exception as e:
        error_data = json.dumps({"type": "error", "message": str(e)})
        self.wfile.write(f"data: {error_data}\n\n".encode())
        self.wfile.flush()
```

#### 4. Add Status Check Endpoint
**File**: `rollouts/frontend/server.py` (add to `do_GET()`)

```python
elif path == "/api/runs":
    self._list_active_runs()
```

```python
def _list_active_runs(self):
    """List all active and recent runs."""
    runs = []
    for run_id, data in _active_runs.items():
        runs.append({
            "run_id": run_id,
            "config_name": data["config_name"],
            "start_time": data["start_time"],
            "status": data["status"],
            "exit_code": data.get("exit_code")
        })
    self._json_response({"runs": runs})
```

### Frontend Changes Needed

#### 1. Update State for Live Streaming
**File**: `rollouts/frontend/index.html:952`

```javascript
state.liveRun = {
    runId: null,
    configName: null,
    status: 'idle',  // 'idle' | 'running' | 'completed' | 'failed'
    output: [],      // Array of output lines
    startTime: null,
    eventSource: null
};
```

#### 2. Update `launchAgent()` to Connect to Stream
**File**: `rollouts/frontend/index.html:1523`

After successful launch response:

```javascript
const result = await response.json();
const { run_id } = result;

// Connect to live stream
state.liveRun.runId = run_id;
state.liveRun.configName = configName;
state.liveRun.status = 'running';
state.liveRun.output = [];
state.liveRun.startTime = Date.now();

// Switch to live-stream mode
state.ui.mainPaneMode = 'live-stream';
updateUI();

// Connect EventSource
const eventSource = new EventSource(`/api/stream/${run_id}`);
state.liveRun.eventSource = eventSource;

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'stdout') {
        // Append output line
        state.liveRun.output.push(data.line);
        renderLiveOutput();
    } else if (data.type === 'complete') {
        // Run finished
        state.liveRun.status = data.status;
        eventSource.close();
        launchBtn.textContent = data.status === 'success' ? '✓ Completed' : '✗ Failed';

        // Reload results
        setTimeout(() => {
            loadResults();
            launchBtn.textContent = originalText;
            launchBtn.disabled = false;
        }, 1000);
    } else if (data.type === 'error') {
        console.error('Stream error:', data.message);
        state.liveRun.status = 'failed';
        eventSource.close();
    }
};

eventSource.onerror = () => {
    console.error('EventSource error');
    state.liveRun.status = 'failed';
    eventSource.close();
};
```

#### 3. Add Live Output Rendering
**File**: `rollouts/frontend/index.html` (add new function)

```javascript
function renderLiveOutput() {
    const mainPane = document.querySelector('.main-pane');
    if (state.ui.mainPaneMode !== 'live-stream') return;

    const { configName, output, status } = state.liveRun;

    let html = `
        <div style="padding: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <h2 style="margin: 0;">🚀 Running: ${configName}</h2>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <div class="status-indicator ${status}">${status}</div>
                    <button class="btn" onclick="stopLiveRun()">Stop</button>
                </div>
            </div>

            <div class="live-output-container" style="
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 16px;
                border-radius: 4px;
                font-family: 'SF Mono', Monaco, monospace;
                font-size: 12px;
                line-height: 1.5;
                max-height: calc(100vh - 200px);
                overflow-y: auto;
            ">
    `;

    if (output.length === 0) {
        html += '<div style="color: #888;">Waiting for output...</div>';
    } else {
        output.forEach(line => {
            // Color code different types of output
            let color = '#d4d4d4';
            if (line.includes('ERROR') || line.includes('FAILED')) {
                color = '#f48771';
            } else if (line.includes('SUCCESS') || line.includes('PASSED')) {
                color = '#89d185';
            } else if (line.includes('WARNING')) {
                color = '#dcdcaa';
            }

            html += `<div style="color: ${color};">${escapeHtml(line)}</div>`;
        });
    }

    html += `
            </div>
        </div>
    `;

    mainPane.innerHTML = html;

    // Auto-scroll to bottom
    const container = mainPane.querySelector('.live-output-container');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function stopLiveRun() {
    if (state.liveRun.eventSource) {
        state.liveRun.eventSource.close();
    }
    state.liveRun.status = 'stopped';
    state.ui.mainPaneMode = 'idle';
    updateUI();

    // TODO: Send stop signal to backend to kill process
}
```

#### 4. Update `updateUI()` to Handle Live Stream Mode
**File**: `rollouts/frontend/index.html` (in `updateUI()` function)

```javascript
// Show/hide main pane based on mode
const mainPane = document.querySelector('.main-pane');
if (state.ui.mainPaneMode === 'live-stream') {
    mainPane.style.display = 'block';
    renderLiveOutput();
} else if (state.ui.mainPaneMode === 'trajectory-view') {
    mainPane.style.display = 'block';
    // ... existing trajectory view code ...
} else {
    mainPane.style.display = 'none';
}
```

## Task 2: Fix Result Display Issues

### Current Issues to Investigate

1. **Check if results are being loaded correctly**
   - Verify `loadResults()` function at `rollouts/frontend/index.html:~1440`
   - Check `/api/results` endpoint returns correct data

2. **Check result rendering**
   - Verify results sidebar displays loaded data
   - Check if clicking a result shows details

3. **Check result file paths**
   - Ensure results are written to expected directory
   - Verify backend can find and read result files

### Debug Steps

```bash
# 1. Check what results exist
ls -la results/

# 2. Test results API directly
curl http://localhost:9000/api/results | jq '.'

# 3. Check browser console for errors
# Open DevTools -> Console when viewing results sidebar
```

### Potential Fixes

**If results aren't showing:**
- Check `loadResults()` function handles response correctly
- Verify results directory path is correct
- Add error handling for missing/malformed result files

**If results show but can't view details:**
- Check trajectory view rendering
- Verify result file parsing in backend
- Add proper error messages for failed loads

## Implementation Plan

1. **Start with backend SSE streaming**
   - Add process registry and tracking
   - Implement `/api/stream/<run_id>` endpoint
   - Test with curl/SSE client

2. **Add frontend EventSource connection**
   - Update `launchAgent()` to connect
   - Implement `renderLiveOutput()`
   - Add stop functionality

3. **Test end-to-end**
   - Launch a config
   - Verify live output streams
   - Check completion handling

4. **Fix result display**
   - Debug what's not working
   - Implement fixes
   - Test result viewing

## Testing Checklist

- [ ] Launch config shows live output immediately
- [ ] Output streams line-by-line as process runs
- [ ] Colors/formatting work for different log types
- [ ] Completion updates button state correctly
- [ ] Results sidebar refreshes after completion
- [ ] Can stop a running process
- [ ] Multiple runs can be tracked simultaneously
- [ ] EventSource connection closes properly
- [ ] Results display correctly in sidebar
- [ ] Can click result to view trajectory/details

## Notes

- Use Server-Sent Events (SSE) not WebSockets - simpler, unidirectional
- Keep process registry in memory (could persist to file later)
- Consider max output buffer size (e.g., 10000 lines) to prevent memory issues
- Add timestamps to output lines for debugging
- Color-code output: errors (red), success (green), warnings (yellow)
- Auto-scroll output to bottom as new lines arrive
- Consider adding "Clear Output" and "Download Logs" buttons
