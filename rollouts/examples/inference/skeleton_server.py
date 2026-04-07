"""
Skeleton inference server.

Implements the minimal HTTP contract expected by the rollouts eval harness:

    GET  /health                → {"status": "ok"}
    POST /v1/chat/completions   → OpenAI-compatible response

Students replace `generate_reply` with their inference logic. Everything else
(HTTP routing, request parsing, response shaping) is already correct and
should not need to change.

Usage:
    python skeleton_server.py --model <path-or-name> --port 30000

The benchmark/eval harness will:
  1. Start this process with --model and --port
  2. Poll GET /health until 200
  3. POST requests to /v1/chat/completions and parse responses
"""

import argparse
import logging
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# THE ONLY FUNCTION STUDENTS NEED TO IMPLEMENT
# ---------------------------------------------------------------------------


def generate_reply(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    **kwargs: object,
) -> tuple[str, str]:
    """Generate a reply for the given conversation.

    Args:
        messages:    Conversation so far. Each dict has "role" ("system" /
                     "user" / "assistant") and "content" (str).
        model:       Model name/path as passed on the command line.
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        **kwargs:    Any extra fields from the request (top_p, etc.) - safe
                     to ignore.

    Returns:
        (reply_text, finish_reason)
        finish_reason must be "stop" (hit EOS) or "length" (hit max_tokens).
    """
    # TODO: replace this stub with real inference
    raise NotImplementedError("implement generate_reply")


# ---------------------------------------------------------------------------
# HTTP SERVER - students should not need to touch anything below this line
# ---------------------------------------------------------------------------


def build_app(model_name: str) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict) -> dict:
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.0)

        # Strip keys we've already captured so kwargs doesn't duplicate them
        extra = {
            k: v
            for k, v in request.items()
            if k not in {"messages", "max_tokens", "temperature", "model", "stream"}
        }

        try:
            reply, finish_reason = generate_reply(
                messages=messages,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail="generate_reply not implemented") from e
        except Exception as e:
            logger.exception("generate_reply raised")
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Count tokens naively (whitespace split) - replace with a real
        # tokenizer if the benchmark tracks token counts precisely.
        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_tokens = len(prompt_text.split())
        completion_tokens = len(reply.split())

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting skeleton server: model={args.model} port={args.port}")

    app = build_app(model_name=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
