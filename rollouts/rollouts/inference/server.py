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
import functools
import logging
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import torch
import trio

from .core import Req, SamplingParams
from .engine_v2 import EngineConfig, InferenceEngineV2
from .models.weight import load_weights

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE THREAD
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EngineResult:
    """Result from engine thread."""

    uid: int
    req: Req | None = None  # Set when request finishes
    new_token: int | None = None  # Set for streaming updates
    finish_reason: str | None = None


class EngineThread:
    """Dedicated thread running engine loop.

    The engine runs at GPU pace, pulling requests from a queue.
    Results are pushed to a result queue for the HTTP layer to dispatch.

    If the engine thread crashes, the exception is stored in `fatal_error`
    and all pending requests will timeout. Check this on health endpoints.
    """

    def __init__(self, engine: InferenceEngineV2) -> None:
        self.engine = engine

        # Thread-safe communication
        self._request_queue: queue.Queue[tuple[list[int], SamplingParams, bool] | None] = (
            queue.Queue()
        )
        self._result_queue: queue.Queue[EngineResult] = queue.Queue()

        # Track streaming uids (engine's uids)
        self._streaming_uids: set[int] = set()

        # Fatal error from engine thread (if any)
        self.fatal_error: BaseException | None = None

        # Thread state
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the engine thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Engine thread started")

    def stop(self) -> None:
        """Stop the engine thread."""
        self._stop_event.set()
        self._request_queue.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Engine thread stopped")

    def submit_request(
        self, input_ids: list[int], sampling_params: SamplingParams, streaming: bool = False
    ) -> int:
        """Submit request and return engine-assigned uid."""
        # Add to engine directly (in main thread before request_queue)
        # This ensures we get the uid immediately
        uid = self.engine.add_request(input_ids, sampling_params)
        logger.info(f"Submitted request uid={uid}, tokens={len(input_ids)}, streaming={streaming}")

        if streaming:
            self._streaming_uids.add(uid)

        # Signal the engine thread that new work is available
        self._request_queue.put((input_ids, sampling_params, streaming))
        return uid

    def get_results(self, timeout: float | None = None) -> list[EngineResult]:
        """Get available results (non-blocking or with timeout)."""
        results = []
        try:
            result = self._result_queue.get(timeout=timeout)
            results.append(result)
            # Drain remaining
            while True:
                try:
                    results.append(self._result_queue.get_nowait())
                except queue.Empty:
                    break
        except queue.Empty:
            pass
        return results

    def _run_loop(self) -> None:
        """Main engine loop.

        On crash, stores exception in self.fatal_error and exits.
        The HTTP layer should check this and fail health checks.
        """
        logger.info("Engine loop starting")
        step_count = 0

        try:
            while not self._stop_event.is_set():
                self._drain_queue()

                if not self.engine.has_pending():
                    try:
                        msg = self._request_queue.get(timeout=0.1)
                        if msg is None:
                            break
                    except queue.Empty:
                        continue

                step_count += 1
                finished = self.engine.step()

                # Push streaming token updates for decode requests
                for req in self.engine.state.decode_set:
                    if req.uid in self._streaming_uids:
                        if len(req.input_ids) > 0:
                            # req.input_ids is a tensor per core.py Req definition
                            token = req.input_ids[-1].item()
                            self._result_queue.put(EngineResult(uid=req.uid, new_token=token))

                # Push finished results
                for req in finished:
                    # req.input_ids is a tensor per core.py Req definition
                    last_token = req.input_ids[-1].item()
                    is_eos = last_token == self.engine.eos_token_id
                    finish_reason = "stop" if is_eos else "length"
                    self._result_queue.put(
                        EngineResult(uid=req.uid, req=req, finish_reason=finish_reason)
                    )
                    self._streaming_uids.discard(req.uid)

            logger.info(f"Engine loop stopped after {step_count} steps")

        except BaseException as e:
            # Store the exception so HTTP layer knows we crashed
            self.fatal_error = e
            logger.exception(f"Engine loop crashed after {step_count} steps")

    def _drain_queue(self) -> None:
        """Drain request queue (requests already added to engine)."""
        while True:
            try:
                msg = self._request_queue.get_nowait()
                if msg is None:
                    self._stop_event.set()
                    return
                # Request was already added in submit_request, just tracking
            except queue.Empty:
                break


# ═══════════════════════════════════════════════════════════════════════════════
# NCCL WEIGHT SYNC STATE
# ═══════════════════════════════════════════════════════════════════════════════

# Global state for NCCL weight sync (one process group per server)
_nccl_state: dict[str, Any] = {
    "process_group": None,
    "group_name": None,
    "rank": None,
    "world_size": None,
}


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
        logger.info("InferenceServer.__init__ starting")
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.model_name = engine.config.model_path
        logger.info("InferenceServer: basic attributes set")

        # Engine thread for GPU work
        logger.info("InferenceServer: creating EngineThread")
        self._engine_thread = EngineThread(engine)
        logger.info("InferenceServer: EngineThread created")

        # Pending requests: uid -> (event, return_logprob)
        self._pending: dict[int, tuple[trio.Event, bool]] = {}
        self._pending_result: dict[int, Req] = {}
        self._lock = trio.Lock()
        # Semaphore to limit concurrent requests to engine capacity
        self._request_slots = trio.Semaphore(engine.scheduler_config.max_batch_size)

        # Streaming requests: uid -> (send channel, prompt_len, seen_len)
        # Send channel receives (token_id, is_done, finish_reason) tuples
        self._streaming: dict[int, tuple[trio.MemorySendChannel, int, int]] = {}
        logger.info("InferenceServer.__init__ complete")

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

        # Wait for available slot (backpressure when engine is full)
        await self._request_slots.acquire()

        try:
            # Submit to engine thread and get uid
            async with self._lock:
                uid = self._engine_thread.submit_request(
                    request.input_ids, sampling_params, streaming=False
                )
                event = trio.Event()
                self._pending[uid] = (event, request.return_logprob)

            # Wait for completion
            await event.wait()
            result = self._pending_result.pop(uid, None)
            if result is None:
                raise RuntimeError(f"Missing result for uid={uid}")
        finally:
            # Release slot for next request
            self._request_slots.release()

        # Build response
        output_ids = result.input_ids[len(request.input_ids) :].tolist()
        text = self.tokenizer.decode(output_ids, skip_special_tokens=False)

        meta_info: dict[str, Any] = {
            "finish_reason": "stop"
            if result.input_ids[-1] == self.engine.eos_token_id
            else "length",
        }

        if request.return_logprob and result.logprobs is not None:
            # Format: [[logprob, token_id], ...]
            # Convert logprobs tensor to Python floats
            logprobs_list = result.logprobs.tolist()
            meta_info["output_token_logprobs"] = [
                [float(lp), tid] for lp, tid in zip(logprobs_list, output_ids, strict=False)
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

        # Wait for available slot (backpressure when engine is full)
        await self._request_slots.acquire()

        try:
            # Submit to engine thread
            async with self._lock:
                uid = self._engine_thread.submit_request(
                    input_ids, sampling_params, streaming=False
                )
                event = trio.Event()
                self._pending[uid] = (event, return_logprobs)

            await event.wait()
            result = self._pending_result.pop(uid, None)
            if result is None:
                raise RuntimeError(f"Missing result for uid={uid}")
        finally:
            # Release slot for next request
            self._request_slots.release()

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

        if return_logprobs and result.logprobs is not None:
            # Format token IDs as "token_id:123" per QED-Nano expectation
            # Convert logprobs tensor to Python floats
            logprobs_list = result.logprobs.tolist()
            choice["logprobs"] = {
                "content": [
                    {"token": f"token_id:{tid}", "logprob": float(lp)}
                    for lp, tid in zip(logprobs_list, output_ids, strict=False)
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

    async def chat_completions_stream(self, request: dict[str, Any]) -> AsyncGenerator[str, None]:
        """Handle streaming OpenAI-style /v1/chat/completions request.

        Yields SSE-formatted chunks as tokens are generated.
        """

        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 16)
        temperature = request.get("temperature", 1.0)
        top_p = request.get("top_p", 1.0)
        top_k = request.get("top_k", -1)

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(input_ids)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            return_logprobs=False,
        )

        # Create unique ID for this completion
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Wait for available slot
        await self._request_slots.acquire()

        try:
            # Create token queue for streaming
            token_send, token_recv = trio.open_memory_channel[Any](
                self._engine_thread.engine.scheduler_config.max_batch_size
            )

            # Submit to engine thread
            async with self._lock:
                uid = self._engine_thread.submit_request(input_ids, sampling_params, streaming=True)
                # Register for streaming: (queue, prompt_len, seen_len)
                self._streaming[uid] = (token_send, prompt_len, prompt_len)

            # Stream tokens as they arrive
            while True:
                # Get next token or done signal
                token_id, is_done, finish_reason = await token_recv.receive()

                if is_done:
                    # Final chunk
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    yield f"data: {self._json_dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    break

                # Decode single token
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": self.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {self._json_dumps(chunk)}\n\n"

        finally:
            # Cleanup
            self._streaming.pop(uid, None)
            token_send.close()
            self._request_slots.release()

    def _json_dumps(self, obj: Any) -> str:
        """JSON serialize without whitespace."""
        import json

        return json.dumps(obj, separators=(",", ":"))

    async def result_dispatcher(self, startup_complete: trio.Event | None = None) -> None:
        """Background task that dispatches results from engine thread.

        Polls the engine thread's result queue and dispatches to pending
        futures and streaming queues.
        """
        logger.info("Result dispatcher starting")
        if startup_complete is not None:
            startup_complete.set()

        while True:
            # Poll engine thread for results (blocking with timeout)
            # Use functools.partial since trio.to_thread.run_sync doesn't pass args
            results = await trio.to_thread.run_sync(
                functools.partial(self._engine_thread.get_results, 0.01)
            )

            if not results:
                # No results, yield to other coroutines
                await trio.sleep(0.001)
                continue

            logger.info(f"Dispatching {len(results)} results")

            async with self._lock:
                for result in results:
                    uid = result.uid

                    if result.req is not None:
                        # Request finished
                        if uid in self._pending:
                            event, _ = self._pending.pop(uid)
                            self._pending_result[uid] = result.req
                            if not event.is_set():
                                event.set()

                        elif uid in self._streaming:
                            # Signal completion to streaming handler
                            token_queue, prompt_len, seen_len = self._streaming[uid]
                            # Push final done signal
                            await token_queue.send((None, True, result.finish_reason))
                            logger.debug(f"Streaming complete for uid={uid}")

                    elif result.new_token is not None:
                        # Streaming token update
                        if uid in self._streaming:
                            token_queue, prompt_len, seen_len = self._streaming[uid]
                            await token_queue.send((result.new_token, False, None))


def create_app(engine: InferenceEngineV2) -> Any:
    """Create FastAPI app with inference endpoints."""
    logger.info("create_app: starting")

    from fastapi import FastAPI, HTTPException

    logger.info("create_app: creating InferenceServer")
    server = InferenceServer(engine)
    logger.info("create_app: InferenceServer created")

    # Start engine thread immediately (synchronous, no event loop needed)
    logger.info("create_app: starting engine thread")
    server._engine_thread.start()
    logger.info("create_app: engine thread started")

    logger.info("create_app: creating FastAPI app")
    app = FastAPI(title="Rollouts Inference Server")
    app.state.inference_server = server
    app.state.startup_complete = None
    logger.info("create_app: FastAPI app created")

    @app.get("/health")
    async def health() -> dict:
        # Fail health check until startup is complete
        startup_complete = app.state.startup_complete
        if startup_complete is None or not startup_complete.is_set():
            raise HTTPException(
                status_code=503,
                detail="Server starting up, result dispatcher not ready",
            )
        # Fail health check if engine thread crashed
        if server._engine_thread.fatal_error is not None:
            err = server._engine_thread.fatal_error
            raise HTTPException(
                status_code=503,
                detail=f"Engine thread crashed: {type(err).__name__}: {err}",
            )
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
    async def chat_completions(request: dict) -> Any:
        """OpenAI-style chat completions endpoint.

        Supports both streaming (stream=true) and non-streaming modes.
        """
        from fastapi.responses import StreamingResponse

        try:
            if request.get("stream", False):
                # Streaming mode: return SSE stream
                return StreamingResponse(
                    server.chat_completions_stream(request),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming mode: return complete response
                return await server.chat_completions(request)
        except Exception as e:
            logger.exception("Error in /v1/chat/completions")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ═══════════════════════════════════════════════════════════════════════════
    # WEIGHT SYNC ENDPOINTS (for RL on-policy training)
    # ═══════════════════════════════════════════════════════════════════════════

    @app.post("/update_weights_from_disk")
    async def update_weights_from_disk(request: dict) -> dict:
        """Load weights from checkpoint directory.

        Request: {"model_path": "/path/to/checkpoint"}
        """
        model_path = request.get("model_path")
        if not model_path:
            raise HTTPException(status_code=400, detail="model_path required")

        try:
            logger.info(f"Loading weights from {model_path}")
            state_dict = load_weights(model_path, engine.device, engine.config.dtype)
            engine.reload_weights(state_dict)
            logger.info("Weight reload complete")
            return {"status": "ok", "model_path": model_path}
        except Exception as e:
            logger.exception("Error in /update_weights_from_disk")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/init_weights_update_group")
    async def init_weights_update_group(request: dict) -> dict:
        """Initialize NCCL process group for weight sync.

        Request: {
            "master_address": "127.0.0.1",
            "master_port": 29500,
            "rank_offset": 1,
            "world_size": 2,
            "group_name": "weight_sync",
            "backend": "nccl"
        }
        """
        import os

        import torch.distributed as dist

        master_addr = request.get("master_address", "127.0.0.1")
        master_port = request.get("master_port", 29500)
        rank_offset = request.get("rank_offset", 1)
        world_size = request.get("world_size", 2)
        group_name = request.get("group_name", "weight_sync")
        backend = request.get("backend", "nccl")

        # Set environment for NCCL
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        try:
            logger.info(f"Initializing NCCL group: rank={rank_offset}, world_size={world_size}")

            # Initialize process group
            dist.init_process_group(
                backend=backend,
                rank=rank_offset,
                world_size=world_size,
            )

            # Store state
            _nccl_state["process_group"] = dist.group.WORLD
            _nccl_state["group_name"] = group_name
            _nccl_state["rank"] = rank_offset
            _nccl_state["world_size"] = world_size

            logger.info("NCCL group initialized")
            return {"status": "ok", "rank": rank_offset, "world_size": world_size}
        except Exception as e:
            logger.exception("Error in /init_weights_update_group")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/update_weights_from_distributed")
    async def update_weights_from_distributed(request: dict) -> dict:
        """Receive weights via NCCL broadcast from trainer.

        Request: {
            "names": ["model.embed_tokens.weight", ...],
            "shapes": [[32000, 4096], ...],
            "dtypes": ["bfloat16", ...],
        }
        """
        import torch.distributed as dist

        if _nccl_state["process_group"] is None:
            raise HTTPException(status_code=400, detail="NCCL group not initialized")

        names = request.get("names", [])
        shapes = request.get("shapes", [])
        dtypes = request.get("dtypes", [])

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        try:
            logger.info(f"Receiving {len(names)} weight tensors via NCCL")
            new_state_dict = {}

            for name, shape, dtype_str in zip(names, shapes, dtypes, strict=True):
                dtype = dtype_map.get(dtype_str, torch.bfloat16)
                # Allocate tensor and receive broadcast from rank 0
                tensor = torch.empty(shape, dtype=dtype, device=engine.device)
                dist.broadcast(tensor, src=0)
                new_state_dict[name] = tensor

            # Apply weights
            engine.reload_weights(new_state_dict)
            logger.info("NCCL weight update complete")
            return {"status": "ok", "num_tensors": len(names)}
        except Exception as e:
            logger.exception("Error in /update_weights_from_distributed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/destroy_weights_update_group")
    async def destroy_weights_update_group(request: dict) -> dict:
        """Destroy NCCL process group."""
        import torch.distributed as dist

        group_name = request.get("group_name", "weight_sync")

        try:
            if _nccl_state["process_group"] is not None:
                logger.info(f"Destroying NCCL group: {group_name}")
                dist.destroy_process_group()
                _nccl_state["process_group"] = None
                _nccl_state["group_name"] = None
                _nccl_state["rank"] = None
                _nccl_state["world_size"] = None

            return {"status": "ok"}
        except Exception as e:
            logger.exception("Error in /destroy_weights_update_group")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def run_server(app: Any, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the inference server."""
    from hypercorn.config import Config
    from hypercorn.trio import serve

    async def serve_http() -> None:
        server = getattr(app.state, "inference_server", None)

        if server is None:
            raise RuntimeError("create_app must be used before run_server")

        # Create startup gate inside trio event loop
        startup_complete = trio.Event()
        app.state.startup_complete = startup_complete

        config = Config()
        config.bind = [f"{host}:{port}"]

        async with trio.open_nursery() as nursery:
            nursery.start_soon(server.result_dispatcher, startup_complete)
            logger.info("Started result dispatcher task")
            try:
                await serve(app, config)
            finally:
                server._engine_thread.stop()

    try:
        trio.run(serve_http)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


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
    logger.info("Model loaded successfully")

    logger.info("Creating app...")
    app = create_app(engine)
    logger.info("App created, starting hypercorn (trio runtime)...")
    run_server(app, host=args.host, port=args.port)
    logger.info("Server stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
