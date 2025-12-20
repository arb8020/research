# Browser Eval Runner

**DRI:** Chiraag
**Claude:** [this conversation]

## Context
Run agent evals from a browser UI with live streaming, using a thin Python backend for environment execution only.

## Out of Scope
- Replacing the Python eval runner (this is an alternative for interactive use/demos)
- Running environments in browser (Pyodide, WASM)
- Proxying LLM calls through backend

## Solution
**Input:** Eval sample (problem description, kernel code, etc.)
**Output:** Live-streamed agent trace with tool results, final reward

```
┌─────────────────────────────────────────────────────────┐
│  Browser                                                │
│  - Agent loop (messages array)                          │
│  - Direct Claude API calls (streaming)                  │
│  - UI rendering                                         │
└────────────────┬────────────────────────────────────────┘
                 │ POST /execute {tool_name, args}
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Python Backend (thin)                                  │
│  - Session management                                   │
│  - Routes to existing Environment classes               │
│  - Returns tool results                                 │
└─────────────────────────────────────────────────────────┘
```

## Usage

**Browser (JS):**
```javascript
const messages = [{ role: "user", content: problem }];

while (!done) {
  // Stream from Claude directly
  const response = await streamFromClaude(messages, tools);
  messages.push(response);

  if (response.tool_calls) {
    for (const call of response.tool_calls) {
      const result = await fetch("/execute", {
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId,
          tool_name: call.name,
          args: call.args,
          idempotency_key: call.id,
        }),
      });
      messages.push({ role: "tool", content: await result.json() });
    }
  } else {
    done = true;
  }
}
```

**Backend (Python):**
```python
@app.post("/sessions")
async def create_session(env_type: str) -> Session:
    env = create_environment(env_type)  # existing code
    return Session(id=uuid4(), env=env)

@app.post("/sessions/{id}/execute")
async def execute(id: str, req: ToolRequest) -> ToolResult:
    session = get_session(id)
    return await session.env.execute_tool(req.tool_name, req.args)
```

---

## Details

### Flow
1. User opens HTML file, pastes Anthropic API key (stored in localStorage)
2. User selects/pastes eval sample
3. Browser creates session: `POST /sessions {env_type: "kernelbench"}`
4. Browser starts agent loop:
   - Call Claude API directly (streaming to UI)
   - For each tool call: `POST /sessions/{id}/execute`
   - Append results to messages
   - Repeat until no tool calls
5. Display final trace + reward

### API

```
POST /sessions
  body: {env_type: "kernelbench" | "docs" | ...}
  → {session_id: string}

POST /sessions/{id}/execute
  body: {tool_name: string, args: object, idempotency_key: string}
  → {content: string, details?: object}

DELETE /sessions/{id}
  → {ok: true}
```

### Decisions
- **Long-running tools:** SSE for streaming results
- **Session cleanup:** Explicit `DELETE /sessions/{id}`
- **Server:** FastAPI + Hypercorn with `--worker-class trio` (trio-native, no asyncio)
- **Auth:** None for local dev, simple token for deployed

### Open Questions
- [ ] Auth for backend (simple token? none for local?)
- [ ] Bundle as single HTML file or separate JS?

### Files
**Read:**
- `rollouts/providers.py` - understand streaming format
- `research/evals/kernelbench/base_config.py` - existing eval runner
- `apps/trace-viewer/index.html` - existing HTML tool pattern

**Modify:**
- `apps/browser-runner/index.html` (new)
- `services/wafer-api/...` or new lightweight server (new)
