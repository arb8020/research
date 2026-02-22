"""HTTP server for inference engine.

Provides two API styles:
1. SGLang-style /generate endpoint (token IDs in/out, used by Miles)
2. OpenAI-style /v1/chat/completions endpoint (messages, used by QED-Nano)

Usage:
    python -m rollouts.inference.server --model <path> --port 8000

Or programmatically:
    from rollouts.inference.server import create_app, run_server
    app = create_app(engine)
    run_server(app, port=8000)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from .core import SamplingParams
from .engine_v2 import EngineConfig, InferenceEngineV2

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS (Pydantic-free for minimal deps)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GenerateRequest:
    """SGLang-style /generate request (Miles compatibility).

    Input is token IDs directly - no tokenization on server.
    """

    input_ids: list[int]
    sampling_params: dict[str, Any] = field(default_factory=dict)
    return_logprob: bool = False
    return_routed_experts: bool = False  # For MoE routing replay


@dataclass
class GenerateResponse:
    """SGLang-style /generate response."""

    text: str
    meta_info: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


class InferenceServer:
    """HTTP server wrapping InferenceEngineV2.

    Handles async request batching and provides both SGLang and OpenAI APIs.
    """

    def __init__(self, engine: InferenceEngineV2) -> None:
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.model_name = engine.config.model_path

        # Pending requests: uid -> (future, return_logprob)
        self._pending: dict[int, tuple[asyncio.Future, bool]] = {}
        self._lock = asyncio.Lock()
        self._step_event = asyncio.Event()

    def _convert_sampling_params(self, params: dict[str, Any]) -> SamplingParams:
        """Convert SGLang sampling params to our format."""
        return SamplingParams(
            temperature=params.get("temperature", 1.0),
            top_p=params.get("top_p", 1.0),
            top_k=params.get("top_k", -1),
            max_tokens=params.get("max_new_tokens", 256),
            ignore_eos=params.get("ignore_eos", False),
            return_logprobs=True,  # Always compute, filter in response
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Handle SGLang-style /generate request."""
        sampling_params = self._convert_sampling_params(request.sampling_params)

        # Add request to engine
        async with self._lock:
            uid = self.engine.add_request(request.input_ids, sampling_params)
            future: asyncio.Future = asyncio.Future()
            self._pending[uid] = (future, request.return_logprob)

        # Signal step loop
        self._step_event.set()

        # Wait for completion
        result = await future

        # Build response
        output_ids = result.input_ids[len(request.input_ids) :].tolist()
        text = self.tokenizer.decode(output_ids, skip_special_tokens=False)

        meta_info: dict[str, Any] = {
            "finish_reason": "stop"
            if result.input_ids[-1] == self.engine.eos_token_id
            else "length",
        }

        if request.return_logprob and result.logprobs:
            # Format: [[logprob, token_id], ...]
            meta_info["output_token_logprobs"] = [
                [lp, tid] for lp, tid in zip(result.logprobs, output_ids, strict=False)
            ]

        # TODO: return_routed_experts for MoE

        return GenerateResponse(text=text, meta_info=meta_info)

    async def chat_completions(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle OpenAI-style /v1/chat/completions request."""
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 16)
        temperature = request.get("temperature", 1.0)
        top_p = request.get("top_p", 1.0)
        top_k = request.get("top_k", -1)
        return_logprobs = request.get("logprobs", 0) > 0

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            return_logprobs=return_logprobs,
        )

        # Add request to engine
        async with self._lock:
            uid = self.engine.add_request(input_ids, sampling_params)
            future: asyncio.Future = asyncio.Future()
            self._pending[uid] = (future, return_logprobs)

        self._step_event.set()
        result = await future

        # Build response
        output_ids = result.input_ids[len(input_ids) :].tolist()
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        choice: dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
            if result.input_ids[-1] == self.engine.eos_token_id
            else "length",
        }

        if return_logprobs and result.logprobs:
            # Format token IDs as "token_id:123" per QED-Nano expectation
            choice["logprobs"] = {
                "content": [
                    {"token": f"token_id:{tid}", "logprob": lp}
                    for lp, tid in zip(result.logprobs, output_ids, strict=False)
                ]
            }

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(output_ids),
                "total_tokens": len(input_ids) + len(output_ids),
            },
        }

    async def step_loop(self) -> None:
        """Background task that runs engine steps and dispatches results."""
        while True:
            # Wait for work
            if not self.engine.has_pending():
                await self._step_event.wait()
                self._step_event.clear()

            # Run step
            finished = self.engine.step()

            # Dispatch results
            async with self._lock:
                for req in finished:
                    if req.uid in self._pending:
                        future, _ = self._pending.pop(req.uid)
                        if not future.done():
                            future.set_result(req)

            # Small yield to allow other coroutines
            await asyncio.sleep(0)


def create_app(engine: InferenceEngineV2) -> Any:
    """Create FastAPI app with inference endpoints."""
    from fastapi import FastAPI, HTTPException

    server = InferenceServer(engine)
    app = FastAPI(title="Rollouts Inference Server")

    @app.on_event("startup")
    async def startup() -> None:
        asyncio.create_task(server.step_loop())

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": server.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "rollouts",
                }
            ],
        }

    @app.post("/generate")
    async def generate(request: dict) -> dict:
        """SGLang-style generate endpoint."""
        try:
            gen_request = GenerateRequest(
                input_ids=request["input_ids"],
                sampling_params=request.get("sampling_params", {}),
                return_logprob=request.get("return_logprob", False),
                return_routed_experts=request.get("return_routed_experts", False),
            )
            response = await server.generate(gen_request)
            return {"text": response.text, "meta_info": response.meta_info}
        except Exception as e:
            logger.exception("Error in /generate")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict) -> dict:
        """OpenAI-style chat completions endpoint."""
        try:
            return await server.chat_completions(request)
        except Exception as e:
            logger.exception("Error in /v1/chat/completions")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def run_server(app: Any, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the inference server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entrypoint for running the inference server."""
    parser = argparse.ArgumentParser(description="Rollouts Inference Server")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF model ID")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Maximum batch size")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="auto",
        choices=["auto", "flashinfer", "flash", "reference"],
        help="Attention backend",
    )

    args = parser.parse_args()

    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    config = EngineConfig(
        model_path=args.model,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        dtype=dtype_map[args.dtype],
        attention_backend=args.attention_backend,
    )

    logger.info(f"Loading model: {args.model}")
    engine = InferenceEngineV2(config)
    logger.info("Model loaded, starting server...")

    app = create_app(engine)
    run_server(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
