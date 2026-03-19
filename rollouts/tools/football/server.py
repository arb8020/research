"""NFL fact-checker agent — Modal server.

Exposes a single SSE endpoint: POST /query
Body: {"query": str, "api_key": str}

Streams typed SSE events:
  {"type": "text_delta",   "text": str}
  {"type": "tool_call",    "id": str, "code": str}
  {"type": "tool_result",  "id": str, "output": str, "dataframes": [...], "error": bool}
  {"type": "done"}
  {"type": "error",        "message": str}

The agent runs inside a Modal Sandbox (per-request) with pandas + nfl_data_py installed.
One tool: run_python(code). Model writes pandas code, sees stdout + captured DataFrames, loops.

PFR is stubbed — if the model tries pfr_scrape() it gets a clear "not yet" message.

Deploy:  modal deploy server.py
Dev:     modal serve server.py
"""

from __future__ import annotations

import json
import logging
import textwrap
from collections.abc import AsyncIterator

import anthropic
import modal
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# ── Modal app ────────────────────────────────────────────────────────────────

app = modal.App("nfl-factchecker")

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Image for the sandbox (data dependencies)
sandbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas>=2.0",
        "nfl_data_py>=0.3",
        "matplotlib>=3.7",
        "tabulate>=0.9",
        "pyarrow>=14.0",
    )
)

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an NFL fact-checker. Users give you claims or questions about \
football and you verify them using real data.

You have one tool: run_python(code).

The Python sandbox has pandas, nfl_data_py, matplotlib, and tabulate pre-installed.
nfl_data_py downloads nflFastR data directly. Key functions:

  import nfl_data_py as nfl

  nfl.import_pbp_data([2023])         # play-by-play (one or more seasons)
  nfl.import_weekly_data([2023])      # weekly player stats
  nfl.import_seasonal_data([2023])    # seasonal aggregates
  nfl.import_rosters([2023])          # rosters
  nfl.import_schedules([2023])        # game results
  nfl.import_snap_counts([2023])      # snap counts
  nfl.import_combine_data()           # combine measurements
  nfl.import_draft_picks()            # draft history

Play-by-play goes back to ~1999. Other tables vary.

Workflow:
1. Write focused pandas code. Print intermediate results to understand columns and data.
2. To surface a table to the user, print it as JSON:
     print(df.to_json(orient="records"))
3. When you have enough data, give a plain-text answer summarizing what it shows.
4. Only state what the data actually shows. If you can't verify something, say so.

PFR (Pro Football Reference) is not available yet. If you need it, say so explicitly."""

# ── Tool spec ────────────────────────────────────────────────────────────────

TOOL_SPEC: dict = {
    "name": "run_python",
    "description": (
        "Execute Python code in a sandbox. "
        "Has pandas, nfl_data_py, matplotlib, tabulate. "
        "stdout is returned. Print df.to_json(orient='records') to surface tables."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to run.",
            }
        },
        "required": ["code"],
    },
}

# ── Sandbox execution ────────────────────────────────────────────────────────

async def run_code(sandbox: modal.Sandbox, code: str) -> tuple[str, bool]:
    """Run code in the sandbox. Returns (output, is_error).

    Wraps the code in a try/except so tracebacks come back as output
    rather than killing the sandbox process.
    """
    # Write code to a temp file via stdin to avoid shell-quoting issues
    wrapped = textwrap.dedent(f"""\
        import sys, traceback
        _code = {repr(code)}
        try:
            exec(compile(_code, "<agent>", "exec"), {{"__name__": "__main__"}})
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
    """)

    process = await sandbox.exec.aio("python3", "-c", wrapped)
    await process.wait.aio()

    stdout_parts: list[str] = []
    async for line in process.stdout:
        stdout_parts.append(line)

    stderr_parts: list[str] = []
    async for line in process.stderr:
        stderr_parts.append(line)

    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)

    is_error = bool(stderr.strip())
    if is_error:
        output = (stdout + "\n" + stderr).strip() if stdout.strip() else stderr.strip()
    else:
        output = stdout

    if len(output) > 20_000:
        output = output[:20_000] + f"\n... (truncated, {len(output)} total chars)"

    return output, is_error


def extract_dataframes(output: str) -> tuple[str, list[list[dict]]]:
    """Pull JSON arrays from stdout, return (remaining_text, dataframes).

    The model is instructed to print df.to_json(orient='records') for tables
    it wants to surface. We extract those lines as structured data and strip
    them from the plain-text output shown to the user.
    """
    dataframes: list[list[dict]] = []
    kept_lines: list[str] = []

    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    dataframes.append(parsed)
                    continue
            except json.JSONDecodeError:
                pass
        kept_lines.append(line)

    return "\n".join(kept_lines).strip(), dataframes


# ── Agent loop ───────────────────────────────────────────────────────────────

async def run_agent(
    query: str,
    api_key: str,
    sandbox: modal.Sandbox,
    max_turns: int = 12,
) -> AsyncIterator[dict]:
    """Core agent loop. Yields typed event dicts."""
    client = anthropic.AsyncAnthropic(api_key=api_key)
    messages: list[dict] = [{"role": "user", "content": query}]

    for _turn in range(max_turns):
        async with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            tools=[TOOL_SPEC],
            messages=messages,
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield {"type": "text_delta", "text": event.delta.text}

            final_message = await stream.get_final_message()

        messages.append({"role": "assistant", "content": final_message.content})

        if final_message.stop_reason != "tool_use":
            yield {"type": "done"}
            return

        tool_results = []
        for block in final_message.content:
            if block.type != "tool_use":
                continue

            code = block.input.get("code", "")
            yield {"type": "tool_call", "id": block.id, "code": code}

            output, is_error = await run_code(sandbox, code)
            clean_output, dataframes = extract_dataframes(output)

            yield {
                "type": "tool_result",
                "id": block.id,
                "output": clean_output,
                "dataframes": dataframes,
                "error": is_error,
            }

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                # Send full output (including any JSON lines) back to model
                "content": output,
            })

        messages.append({"role": "user", "content": tool_results})

    yield {"type": "error", "message": f"Reached max turns ({max_turns}) without finishing."}


# ── FastAPI endpoint ──────────────────────────────────────────────────────────

@web_app.post("/query")
async def query_endpoint(request: Request) -> StreamingResponse:
    body = await request.json()
    user_query = (body.get("query") or "").strip()
    api_key = (body.get("api_key") or "").strip()

    if not user_query:
        return StreamingResponse(
            iter([b'data: {"type":"error","message":"query is required"}\n\n']),
            media_type="text/event-stream",
        )
    if not api_key:
        return StreamingResponse(
            iter([b'data: {"type":"error","message":"api_key is required"}\n\n']),
            media_type="text/event-stream",
        )

    async def event_stream() -> AsyncIterator[bytes]:
        sandbox = await modal.Sandbox.create.aio(
            app=app,
            image=sandbox_image,
            timeout=300,
        )
        try:
            async for event in run_agent(user_query, api_key, sandbox):
                yield f"data: {json.dumps(event)}\n\n".encode()
        except Exception as exc:
            logger.exception("Agent error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n".encode()
        finally:
            await sandbox.terminate.aio()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Modal function ────────────────────────────────────────────────────────────

@app.function(
    image=(
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("anthropic>=0.30", "fastapi[standard]>=0.110")
    ),
    timeout=600,
    allow_concurrent_inputs=20,
)
@modal.asgi_app()
def serve() -> FastAPI:
    return web_app
