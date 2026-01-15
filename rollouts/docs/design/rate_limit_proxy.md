# Rate Limit Proxy

> **Status**: Draft
> **DRI**: Chiraag
> **Date**: 2025-01-12

## Context

When running parallel rollouts, workers make independent API calls. Without coordination, they collectively exceed rate limits, causing cascading retries.

**Current:** Each worker has local `CapacityLimiter` + per-request retry with backoff.

**Problem:** 100 workers × 10 concurrent each = 1000 concurrent requests → instant rate limit.

## Solution

Simple passthrough proxy with centralized rate limiting.

```python
# rollouts/proxy.py

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import httpx
import os

app = FastAPI()

MAX_CONCURRENT = int(os.environ.get("PROXY_MAX_CONCURRENT", "50"))
# Use semaphore for sync context, or trio.CapacityLimiter if async
import asyncio
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

PROVIDER_URLS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}

@app.get("/health")
async def health():
    return {"status": "ok", "max_concurrent": MAX_CONCURRENT}

@app.api_route("/{provider}/{path:path}", methods=["GET", "POST"])
async def proxy(provider: str, path: str, request: Request):
    if provider not in PROVIDER_URLS:
        return Response(status_code=404, content=f"Unknown provider: {provider}")

    if semaphore.locked():
        return Response(status_code=429, headers={"Retry-After": "1"})

    async with semaphore:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.request(
                method=request.method,
                url=f"{PROVIDER_URLS[provider]}/{path}",
                headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
                content=await request.body(),
            )

            # Handle SSE streaming
            if "text/event-stream" in response.headers.get("content-type", ""):
                return StreamingResponse(
                    response.aiter_bytes(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
```

## Usage

```bash
# Start proxy
python -m rollouts.proxy
# or
rollouts proxy --port 8080 --max-concurrent 50

# Workers point to proxy
export ANTHROPIC_BASE_URL=http://localhost:8080/anthropic
rollouts eval dataset.jsonl --max-concurrent 100
```

## Files

**Create:**
- `rollouts/proxy.py` (~50 lines)

**Modify:**
- `rollouts/cli.py` - Add `rollouts proxy` subcommand

## Open Questions

- [ ] Python service vs Cloudflare Worker? (Start with Python, simpler)
- [ ] Should proxy log requests for debugging?
