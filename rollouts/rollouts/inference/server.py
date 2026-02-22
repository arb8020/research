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
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import torch

from .core import Req, SamplingParams
from .engine_v2 import EngineConfig, InferenceEngineV2
from .models.weight import load_weights
from .overlap import ForwardInput, ForwardOutput, get_or_create_streams, overlap_step
from .scheduler import SchedulerState, has_pending_work, schedule_step

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE THREAD (runs overlap loop at GPU pace)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EngineRequest:
    """Request to add to engine."""

    uid: int
    input_ids: list[int]
    sampling_params: SamplingParams


@dataclass
class EngineResult:
    """Result from engine (finished request or token update)."""

    uid: int
    req: Req | None = None  # Set when request finishes
    new_token: int | None = None  # Set for streaming token updates
    current_len: int | None = None  # Current sequence length for streaming
    finish_reason: str | None = None  # Set when request finishes


class EngineThread:
    """Dedicated thread running overlap loop at GPU pace.

    Communication with HTTP layer via thread-safe queues:
    - request_queue: HTTP layer pushes new requests
    - result_queue: Engine pushes finished requests and token updates

    The engine runs its own pace, pulling requests when it has capacity,
    processing batches with overlap, and pushing results back.
    """

    def __init__(self, engine: InferenceEngineV2) -> None:
        self.engine = engine
        self._overlap_state: dict[str, Any] = {}

        # Thread-safe communication
        self._request_queue: queue.Queue[EngineRequest | None] = queue.Queue()
        self._result_queue: queue.Queue[EngineResult] = queue.Queue()

        # Track streaming requests (by client uid)
        self._streaming_uids: set[int] = set()

        # Map client uid -> engine uid (engine generates its own)
        self._uid_map: dict[int, int] = {}
        # Reverse map: engine uid -> client uid
        self._reverse_uid_map: dict[int, int] = {}

        # Thread management
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
        self._request_queue.put(None)  # Unblock the queue
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Engine thread stopped")

    def submit_request(
        self,
        uid: int,
        input_ids: list[int],
        sampling_params: SamplingParams,
        streaming: bool = False,
    ) -> None:
        """Submit a request to the engine (non-blocking)."""
        if streaming:
            self._streaming_uids.add(uid)
        self._request_queue.put(
            EngineRequest(uid=uid, input_ids=input_ids, sampling_params=sampling_params)
        )

    def get_results(self, timeout: float | None = None) -> list[EngineResult]:
        """Get all available results (non-blocking or with timeout)."""
        results = []
        try:
            # Get first result (may block)
            result = self._result_queue.get(timeout=timeout)
            results.append(result)
            # Drain any additional results
            while True:
                try:
                    result = self._result_queue.get_nowait()
                    results.append(result)
                except queue.Empty:
                    break
        except queue.Empty:
            pass
        return results

    def _run_loop(self) -> None:
        """Main engine loop - runs in dedicated thread."""
        logger.info("Engine loop starting")

        # Initialize CUDA graphs if enabled
        if self.engine._use_cuda_graphs and "graphs" not in self.engine._graph_state:
            logger.info("Capturing CUDA graphs...")
            self.engine.capture_cuda_graphs_now()
            logger.info("CUDA graphs captured")

        last_output: ForwardOutput | None = None

        while not self._stop_event.is_set():
            # Pull new requests from queue (non-blocking)
            self._pull_requests()

            # Check if we have work
            if not has_pending_work(self.engine.state) and last_output is None:
                # Wait for new requests
                try:
                    req = self._request_queue.get(timeout=0.1)
                    if req is None:  # Stop signal
                        break
                    self._add_request(req)
                except queue.Empty:
                    continue

            # Run one overlap step
            current_output = self._overlap_step(last_output)

            # Note: streaming updates are pushed in _process_output after
            # the forward pass completes and tokens are available

            last_output = current_output

        # Process any remaining output
        if last_output is not None:
            last_output.copy_done.synchronize()
            self._process_output(last_output)

        logger.info("Engine loop stopped")

    def _pull_requests(self) -> None:
        """Pull all pending requests from queue."""
        while True:
            try:
                req = self._request_queue.get_nowait()
                if req is None:  # Stop signal
                    self._stop_event.set()
                    return
                self._add_request(req)
            except queue.Empty:
                break

    def _add_request(self, req: EngineRequest) -> None:
        """Add request to engine scheduler."""
        # Engine generates its own uid, we track mapping both ways
        engine_uid = self.engine.add_request(
            req.input_ids,
            req.sampling_params,
        )
        # Map both directions
        self._uid_map[req.uid] = engine_uid
        self._reverse_uid_map[engine_uid] = req.uid

    def _overlap_step(self, last_output: ForwardOutput | None) -> ForwardOutput | None:
        """Run one overlap step."""
        return overlap_step(
            state=self._overlap_state,
            device=self.engine.device,
            last_output=last_output,
            forward_fn=self._forward,
            schedule_fn=self._schedule,
            process_fn=self._process_output,
        )

    def _schedule(self) -> ForwardInput | None:
        """Schedule next batch (CPU work)."""
        if not has_pending_work(self.engine.state):
            return None

        result = schedule_step(
            state=self.engine.state,
            config=self.engine.scheduler_config,
            num_free_pages=self.engine._get_num_free_pages(),
            device=self.engine.device,
            allocate_pages=self.engine._allocate_pages,
        )

        if result.batch is None:
            return None

        batch = result.batch
        self.engine.state = result.new_state

        self.engine._update_page_table(batch)
        attn_metadata = self.engine._build_attention_metadata(batch)

        return ForwardInput(batch=batch, attn_metadata=attn_metadata)

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """Run model forward (GPU work)."""
        from .graph import can_use_graph, replay_graph
        from .overlap import create_forward_output

        batch = forward_input.batch
        attn_metadata = forward_input.attn_metadata

        use_graph = (
            self.engine._use_cuda_graphs
            and batch.is_decode
            and "graphs" in self.engine._graph_state
            and can_use_graph(self.engine._graph_state, batch)
        )

        with torch.no_grad():
            if use_graph:
                logits = replay_graph(self.engine._graph_state, batch, attn_metadata)
            else:
                logits = self.engine.model(
                    input_ids=batch.input_ids,
                    positions=batch.positions,
                    attn_backend=self.engine.attn_backend,
                    attn_metadata=attn_metadata,
                    out_loc=batch.out_loc,
                )
                last_indices = attn_metadata.cu_seqlens_q[1:] - 1
                logits = logits[last_indices]

            next_tokens_gpu, logprobs = self.engine._sample_batch_gpu(logits, batch)

        scheduler_stream, _ = get_or_create_streams(self._overlap_state, self.engine.device)

        return create_forward_output(
            next_tokens_gpu=next_tokens_gpu,
            stream=scheduler_stream,
            batch=batch,
            logprobs=logprobs,
        )

    def _process_output(self, output: ForwardOutput) -> None:
        """Process forward output (CPU work after GPU done)."""
        # Push streaming updates BEFORE state update (tokens are in output)
        self._push_streaming_updates(output)

        # Update request states using engine's overlap process method
        self.engine._overlap_process(output)

        # Push finished requests to result queue
        for req in self.engine.state.finished:
            # Translate engine uid to client uid
            client_uid = self._reverse_uid_map.get(req.uid, req.uid)
            is_eos = req.input_ids[-1].item() == self.engine.eos_token_id
            finish_reason = "stop" if is_eos else "length"
            self._result_queue.put(
                EngineResult(uid=client_uid, req=req, finish_reason=finish_reason)
            )
            self._streaming_uids.discard(client_uid)
            # Clean up uid maps
            self._reverse_uid_map.pop(req.uid, None)
            self._uid_map.pop(client_uid, None)

        # Clear finished from state
        if self.engine.state.finished:
            self.engine.state = SchedulerState(
                prefill_queue=self.engine.state.prefill_queue,
                decode_set=self.engine.state.decode_set,
                finished=(),
            )

    def _push_streaming_updates(self, output: ForwardOutput) -> None:
        """Push token updates for streaming requests.

        After processing output, push the new token for each streaming request
        that was in this batch.
        """
        batch = output.batch
        next_tokens = output.next_tokens_cpu

        for i, req in enumerate(batch.reqs):
            # Translate engine uid to client uid
            client_uid = self._reverse_uid_map.get(req.uid, req.uid)
            if client_uid in self._streaming_uids:
                # Push the new token that was just generated
                new_token = next_tokens[i].item()
                self._result_queue.put(
                    EngineResult(
                        uid=client_uid,
                        new_token=new_token,
                    )
                )


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

    Uses dedicated engine thread running overlap loop at GPU pace.
    HTTP layer communicates via thread-safe queues.
    """

    def __init__(self, engine: InferenceEngineV2) -> None:
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.model_name = engine.config.model_path

        # Engine thread for overlap execution
        self._engine_thread = EngineThread(engine)
        self._next_uid = 0

        # Pending requests: uid -> (future, return_logprob)
        self._pending: dict[int, tuple[asyncio.Future, bool]] = {}
        self._lock = asyncio.Lock()
        # Semaphore to limit concurrent requests to engine capacity
        self._request_slots = asyncio.Semaphore(engine.scheduler_config.max_batch_size)

        # Streaming requests: uid -> (queue, prompt_len, seen_len)
        # Queue receives (token_id, is_done, finish_reason) tuples
        self._streaming: dict[int, tuple[asyncio.Queue, int, int]] = {}

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

    def _get_next_uid(self) -> int:
        """Get next request UID (thread-safe via asyncio lock)."""
        uid = self._next_uid
        self._next_uid += 1
        return uid

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Handle SGLang-style /generate request."""
        sampling_params = self._convert_sampling_params(request.sampling_params)

        # Wait for available slot (backpressure when engine is full)
        await self._request_slots.acquire()

        try:
            # Create future and submit request
            async with self._lock:
                uid = self._get_next_uid()
                future: asyncio.Future = asyncio.Future()
                self._pending[uid] = (future, request.return_logprob)

            # Submit to engine thread (non-blocking)
            self._engine_thread.submit_request(uid, request.input_ids, sampling_params)

            # Wait for completion
            result = await future
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
            # Create future and submit request
            async with self._lock:
                uid = self._get_next_uid()
                future: asyncio.Future = asyncio.Future()
                self._pending[uid] = (future, return_logprobs)

            # Submit to engine thread (non-blocking)
            self._engine_thread.submit_request(uid, input_ids, sampling_params)
            result = await future
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
            token_queue: asyncio.Queue = asyncio.Queue()

            # Create future and submit request
            async with self._lock:
                uid = self._get_next_uid()
                # Register for streaming: (queue, prompt_len, seen_len)
                self._streaming[uid] = (token_queue, prompt_len, prompt_len)

            # Submit to engine thread with streaming flag
            self._engine_thread.submit_request(uid, input_ids, sampling_params, streaming=True)

            # Stream tokens as they arrive
            while True:
                # Get next token or done signal
                token_id, is_done, finish_reason = await token_queue.get()

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
            self._request_slots.release()

    def _json_dumps(self, obj: Any) -> str:
        """JSON serialize without whitespace."""
        import json

        return json.dumps(obj, separators=(",", ":"))

    async def result_dispatcher(self) -> None:
        """Background task that dispatches results from engine thread.

        Polls the engine thread's result queue and dispatches to pending
        futures and streaming queues.
        """
        while True:
            # Poll for results from engine thread
            # Use asyncio.to_thread to avoid blocking the event loop
            results = await asyncio.to_thread(self._engine_thread.get_results, timeout=0.01)

            async with self._lock:
                for result in results:
                    if result.req is not None:
                        # Finished request
                        if result.uid in self._pending:
                            future, _ = self._pending.pop(result.uid)
                            if not future.done():
                                future.set_result(result.req)

                        elif result.uid in self._streaming:
                            # Signal completion to streaming handler
                            token_queue, _, _ = self._streaming[result.uid]
                            await token_queue.put((None, True, result.finish_reason))

                    elif result.new_token is not None:
                        # Streaming token update
                        if result.uid in self._streaming:
                            token_queue, _, _ = self._streaming[result.uid]
                            await token_queue.put((result.new_token, False, None))

            # Small yield to allow other coroutines
            await asyncio.sleep(0)


def create_app(engine: InferenceEngineV2) -> Any:
    """Create FastAPI app with inference endpoints."""
    from fastapi import FastAPI, HTTPException

    server = InferenceServer(engine)
    app = FastAPI(title="Rollouts Inference Server")

    @app.on_event("startup")
    async def startup() -> None:
        # Start engine thread (runs overlap loop at GPU pace)
        server._engine_thread.start()
        # Start result dispatcher (polls engine thread for results)
        asyncio.create_task(server.result_dispatcher())

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
