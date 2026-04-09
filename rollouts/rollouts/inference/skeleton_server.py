"""
Skeleton inference server.

Implements the minimal HTTP contract expected by the rollouts eval harness:

    GET  /health                → {"status": "ok"}
    POST /v1/chat/completions   → OpenAI-compatible response

Students replace `generate_reply` with their inference logic. Everything else
(HTTP routing, request parsing, response shaping) is already correct and
should not need to change.

Usage:
    python -m rollouts.inference.skeleton_server --model <path-or-name> --port 30000

The benchmark/eval harness will:
  1. Start this process with --model and --port
  2. Poll GET /health until 200
  3. POST requests to /v1/chat/completions and parse responses

See also:
    rollouts/inference/gold_server.py  - reference HF implementation (naive baseline)
    examples/inference/eval_skeleton_server.py  - eval config that tests this server
"""

import argparse
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StubTokenLogprob:
    token: str
    logprob: float
    top_logprobs: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GenerationResult:
    reply_text: str
    finish_reason: str
    token_logprobs: tuple[StubTokenLogprob, ...] = ()


def _fixed_stub_result() -> GenerationResult:
    token_logprobs = (
        StubTokenLogprob(
            token="<answer>",
            logprob=-0.01,
            top_logprobs=(("<answer>", -0.01), ("<think>", -4.2), ("stub", -5.1)),
        ),
        StubTokenLogprob(
            token="stub",
            logprob=-0.02,
            top_logprobs=(("stub", -0.02), ("hello", -4.4), ("world", -4.7)),
        ),
        StubTokenLogprob(
            token="</answer>",
            logprob=-0.01,
            top_logprobs=(("</answer>", -0.01), ("</think>", -4.0), (".", -5.3)),
        ),
    )
    return GenerationResult(
        reply_text="<answer>stub</answer>",
        finish_reason="stop",
        token_logprobs=token_logprobs,
    )


def _normalize_generation_result(
    result: GenerationResult | tuple[str, str],
) -> GenerationResult:
    if isinstance(result, GenerationResult):
        return result
    reply_text, finish_reason = result
    return GenerationResult(reply_text=reply_text, finish_reason=finish_reason)


def _format_choice_logprobs(token_logprobs: tuple[StubTokenLogprob, ...]) -> dict | None:
    if not token_logprobs:
        return None
    return {
        "content": [
            {
                "token": token_lp.token,
                "logprob": token_lp.logprob,
                "top_logprobs": [
                    {"token": alt_token, "logprob": alt_logprob}
                    for alt_token, alt_logprob in token_lp.top_logprobs
                ],
            }
            for token_lp in token_logprobs
        ]
    }


# ---------------------------------------------------------------------------
# THE ONLY FUNCTION STUDENTS NEED TO IMPLEMENT
# ---------------------------------------------------------------------------


def generate_reply(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    **kwargs: object,
) -> GenerationResult:
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
        GenerationResult(reply_text, finish_reason, token_logprobs)
        finish_reason must be "stop" (hit EOS) or "length" (hit max_tokens).
    """
    # TODO: replace this stub with real inference.
    # The HTTP/SSE/logprob contract is already correct; students should only
    # need to swap out this deterministic stub for a real engine.
    del messages, model, max_tokens, temperature, kwargs
    return _fixed_stub_result()


# ---------------------------------------------------------------------------
# HTTP SERVER - students should not need to touch anything below this line
# ---------------------------------------------------------------------------


def build_app(model_name: str) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    # FastAPI/Pydantic cannot materialize a response model from
    # `StreamingResponse | dict`, so keep the explicit union return value but
    # disable response-model generation at the route boundary.
    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: dict) -> StreamingResponse | dict:
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.0)
        stream = request.get("stream", False)

        # Strip keys we've already captured so kwargs doesn't duplicate them
        extra = {
            k: v
            for k, v in request.items()
            if k not in {"messages", "max_tokens", "temperature", "model", "stream"}
        }

        try:
            result = _normalize_generation_result(
                generate_reply(
                    messages=messages,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra,
                )
            )
        except Exception as e:
            logger.exception("generate_reply raised")
            raise HTTPException(status_code=500, detail=str(e)) from e

        reply = result.reply_text
        finish_reason = result.finish_reason
        choice_logprobs = _format_choice_logprobs(result.token_logprobs)

        # Count tokens naively (whitespace split) - replace with a real
        # tokenizer if the benchmark tracks token counts precisely.
        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_tokens = len(prompt_text.split())
        completion_tokens = len(reply.split())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        if stream:
            # SSE streaming format - eval harness sends stream=True by default.
            async def sse() -> AsyncGenerator[str, None]:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": reply},
                            "finish_reason": None,
                            "logprobs": choice_logprobs,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                done = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    "usage": usage,
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": finish_reason,
                    "logprobs": choice_logprobs,
                }
            ],
            "usage": usage,
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
