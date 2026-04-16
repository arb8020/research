"""
Student inference server — personal learning progression.

Run against B200 over SSH:
    .venv/bin/python -m argus run --config rollouts/examples/inference/evals/configs/eval_student_server_b200_ssh.py --force-deploy-committed

    # Monitor:
    tail -f results/eval/<run>/events.jsonl | jq .

Implements the same HTTP contract as skeleton_server.py. Only generate_reply
needs to change at each step; the HTTP layer is frozen.

    GET  /health                → {"status": "ok"}
    POST /v1/chat/completions   → OpenAI-compatible response

Usage:
    python -m rollouts.inference.student --model <path-or-name> --port 30001

    # Run the eval harness against this server:
    python -m argus run --config rollouts/examples/inference/evals/configs/eval_student_server_b200_ssh.py --force-deploy-committed

--------------------------------------------------------------------------------
UPFRONT RULES — invariants that hold for every step below
--------------------------------------------------------------------------------

- Prefill and decode are always separate code paths. Never collapse them into
  one function, even in the naive torch reference. The compute/memory profiles
  are fundamentally different and the split must be visible in the code.

- Every op is tensor-in tensor-out, named, explicit. No nn.Module. No hidden
  state inside an op. If it has lifecycle or resource ownership, name it
  explicitly — don't bury it in a class.

- Observability is not optional. Every step has before/after numbers logged to
  JSONL. You do not proceed to the next step without a dashboard diff showing
  what moved and why you predicted it would move.

- No optimizations without profiler motivation. If nsight doesn't point at it,
  don't touch it.

--------------------------------------------------------------------------------
PROGRESSION
--------------------------------------------------------------------------------

BREAKPOINT A — "in the room"
    Steps 1-3 + 5-6 done well puts you in every architectural discussion about
    forking SGLang/vLLM for MoE serving. You understand the IR, the routing
    decisions, and where KV cache performance lives. Step 4 is optional at this
    point — do it in a day if the IR feels shaky, skip it if not.

BREAKPOINT B — research contributions
    Steps 7-9 (activation harvesting, spec decode, quantization) are why the
    fork is worth doing. You do not need these to start forking, but they are
    your surface area for original work on top of a working engine.

BREAKPOINT C — kernel opinions
    Step 10 (roofline) gives you the theoretical ceiling. Step 11 (TK kernel)
    is one implementation. Your kernel engineer owns step 11 — you need the
    roofline intuition to have opinions, not the implementation.

--------------------------------------------------------------------------------

Step 1 — Dense inference at all
    Model: Qwen/Qwen3.5-27B
    Goal:  generate_reply produces correct outputs. Final logits match HF
           AutoModelForCausalLM reference within floating point tolerance.
           Prefill and decode are already separate functions.
    Done when: eval exact_match == 1.0 on reverse-text tasks.

Step 2 — Observability
    Before touching anything else, instrument the engine:
    - Per-op: wall time, theoretical FLOP count, measured FLOP/s, theoretical
      bytes moved, measured memory bandwidth, MFU, roofline position
      (compute-bound or memory-bound?).
    - Per forward pass: end-to-end latency, tokens/sec, peak memory.
    - CUDA events bracket each op; sync happens off the hot path in an async
      consumer thread. CPU timestamps measure launch overhead only — not kernel
      execution.
    - Everything writes to a JSONL log (one line per forward pass, appended).
      A small CLI (obs.py diff / obs.py history <op>) is enough dashboard.
    Done when: you can run two back-to-back requests and obs.py diff shows a
    meaningful per-op breakdown with MFU numbers you can reason about.

Step 3 — Internal model denotation (list-of-ops IR) + own KV cache
    Model: Qwen/Qwen3.5-27B (same)
    Goal:  Express the model as an explicit list of named ops, each a pure
           tensor-in tensor-out callable. The forward pass is a loop over that
           list. No magic, no implicit graph. Replace HuggingFace past_key_values
           with an explicit KV cache layout (flat preallocated first) at the same
           time — once you own the forward pass, you own the cache layout too.
    Invariant: activation at layer N must still match HF reference after the
               refactor. Correctness diff is part of the definition of done.
    Done when: the model is a list, every op is named, KV cache layout is
    explicit and documented, and the observability dashboard still produces
    the same numbers as step 2.

Step 4 — Second dense model
    Model: google/gemma-3-27b-it  (Gemma4 31B-IT)
    Goal:  Prove the IR generalizes. Add Gemma4 without changing the op loop —
           only the op implementations and config change.
    Done when: Gemma4 evals pass and the IR required no structural changes.

Step 5 — MoE
    Models (in order of increasing complexity):
      - THUDM/glm-4-9b-chat  (GLM-4.7-Flash — start here)
      - BAAI/bge-reranker-v2-m3 / GPT-OSS-20B
      - deepseek-ai/DeepSeek-V3  (production target, do last)
    Goal:  Expert routing as explicit ops in the list. Router, dispatch,
           expert GEMMs, and combine are each separate named ops — not fused
           into one black box. Wrong expert assignments produce plausible output
           so correctness requires activation diff at the routing layer, not
           just final logits.
    Done when: MoE model evals pass AND router activation at each layer matches
    HF reference.

Step 6 — KV cache
    Goal:  Explicit layout decision (choose and write down: [batch, heads, seq,
           d_head] vs [batch, seq, heads, d_head] and why for your access
           pattern). Flat preallocated cache first, then paged attention,
           then continuous batching. Each sub-step has before/after MFU numbers.
    Invariant: activation harvesting from step 7 must work with whatever layout
               you choose here — decide layout with that constraint in mind.
    Done when: continuous batching works, memory utilization is measurable under
    variable-length workloads, and you can explain the layout decision from the
    hardware access pattern up.

Step 7 — Activation harvesting
    Prerequisite: continuous batching from step 6 must be fully working first.
           Naive harvesting breaks under continuous batching because sequences
           are interleaved in the batch — you cannot simply hook the model and
           save tensors. Goodfire's SGLang fork solved exactly this problem.
           Do not start this step until sequence reconstruction under a live
           batch is correct.
    Goal:  Hook points are first-class — residual stream, attn output, expert
           output are all harvestable with minimal overhead. Harvested activations
           at layer N match HF reference activations at layer N.
           This is the Goodfire problem in miniature: fast engine + correct
           activations is harder than either alone.
    Done when: you can harvest activations without the harvesting overhead
    dominating latency, and a spot-check diff against HF passes.

Step 8 — Speculative decode
    Goal:  Draft model runs against the same KV cache infrastructure. Measure
           draft acceptance rate. Compare residual stream of accepted vs rejected
           draft tokens — this is original research territory.
    Done when: spec decode produces correct outputs faster than greedy decode
    on your target workload profile.

Step 9 — Quantization
    Goal:  FP8 weights, FP4 weights, KV cache quantization. For each: measure
           capability degradation via activation diff (which features survive?)
           not just perplexity. This is the "how much downstream capability
           stays alive" research question.
    Done when: you have a table — quantization scheme, MFU gain, activation
    fidelity loss — and a hypothesis about which ops are most sensitive.

Step 10 — Roofline checkpoint
    Goal:  For each op: theoretical peak (from hardware spec + your step 2
           calibration), measured throughput, gap, and a written hypothesis for
           the gap. No code change — just arithmetic and observation.
    Done when: you can predict which op a kernel change will move before
    measuring it.

Step 11 — First TK attention kernel
    Goal:  Replace the torch attention op with a ThunderKittens kernel, motivated
           by the step 10 roofline. Observability dashboard shows the before/after.
           Activation diff still passes.
    Done when: TK attention is faster than torch baseline on your decode profile
    AND you can explain every tile decision in the kernel.

--------------------------------------------------------------------------------
"""

import argparse
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from rollouts.inference.student import engine
from rollouts.inference.student.engine import Engine
from rollouts.inference.student.observability import JsonlObserver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenLogprob:
    token: str
    logprob: float
    top_logprobs: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GenerationResult:
    reply_text: str
    finish_reason: str
    token_logprobs: tuple[TokenLogprob, ...] = ()


def _format_choice_logprobs(token_logprobs: tuple[TokenLogprob, ...]) -> dict | None:
    if not token_logprobs:
        return None
    return {
        "content": [
            {
                "token": tlp.token,
                "logprob": tlp.logprob,
                "top_logprobs": [
                    {"token": alt_token, "logprob": alt_logprob}
                    for alt_token, alt_logprob in tlp.top_logprobs
                ],
            }
            for tlp in token_logprobs
        ]
    }


# ---------------------------------------------------------------------------
# IMPLEMENT THIS
# ---------------------------------------------------------------------------


def generate_reply(
    eng: Engine,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    observer: JsonlObserver | None = None,
) -> GenerationResult:
    """Call engine.generate and wrap the result."""
    req_id = uuid.uuid4().hex
    reply_text, finish_reason = engine.generate(
        eng, messages, max_tokens, temperature, req_id=req_id, observer=observer
    )
    return GenerationResult(reply_text=reply_text, finish_reason=finish_reason)


# ---------------------------------------------------------------------------
# HTTP SERVER — do not modify
# ---------------------------------------------------------------------------


def build_app(model_name: str, eng: Engine, observer: JsonlObserver | None = None) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: dict) -> StreamingResponse | dict:
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.0)
        stream = request.get("stream", False)

        if max_tokens <= 0:
            raise HTTPException(status_code=400, detail=f"max_tokens must be > 0, got {max_tokens}")

        try:
            result = generate_reply(eng, messages, max_tokens, temperature, observer=observer)
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail="generate_reply not implemented") from e
        except Exception as e:
            logger.exception("generate_reply raised")
            raise HTTPException(status_code=500, detail=str(e)) from e

        reply = result.reply_text
        finish_reason = result.finish_reason
        choice_logprobs = _format_choice_logprobs(result.token_logprobs)

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
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--trace-path", default=None, help="Path to write per-request engine trace JSONL"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger.info(f"Loading model {args.model} ...")
    eng = engine.load(args.model)
    logger.info("Model loaded, starting server.")

    observer = JsonlObserver(Path(args.trace_path)) if args.trace_path else None

    app = build_app(model_name=args.model, eng=eng, observer=observer)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        if observer:
            observer.close()
