"""
Gold (reference) inference server.

Implements generate_reply using HuggingFace transformers directly.
This is the naive baseline - correct, but not optimized. Students
are expected to beat this on throughput and latency.

Supported models:
    Qwen/Qwen3-0.6B          - 0.6B, single GPU, smoke test
    zai-org/GLM-4.7-Flash     - 30B MoE (3B active), multi-GPU, assignment target

Usage:
    # Smoke test (Qwen3-0.6B, single GPU)
    python -m rollouts.inference.gold_server --model Qwen/Qwen3-0.6B --port 30001

    # Assignment target (GLM-4.7-Flash, multi-GPU via device_map=auto)
    python -m rollouts.inference.gold_server --model zai-org/GLM-4.7-Flash --port 30001

    # Run the eval against this server
    python -m rollouts.eval.run --config examples/inference/eval_skeleton_server.py

What students should improve:
    - KV cache management (this reloads weights but has no paged KV cache)
    - Batching (this processes one request at a time)
    - MoE expert parallelism (device_map="auto" is naive tensor slicing, not EP)
    - Continuous batching (this blocks until each request is fully decoded)
    - Speculative decoding, quantization, etc.
"""

import argparse
import logging
import time
import uuid
from threading import Lock

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_model_name = None
_lock = Lock()  # one request at a time - the naive baseline has no batching


def _load_model(model_name: str) -> None:
    global _model, _tokenizer, _model_name
    logger.info(f"Loading model {model_name} ...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # naive: lets HF decide, not expert-parallel
    )
    _model.eval()
    _model_name = model_name
    logger.info(f"Model loaded: {model_name}")


# ---------------------------------------------------------------------------
# Per-model generate_reply implementations
# ---------------------------------------------------------------------------

_QWEN3_THINK_TOKEN_ID = 151668  # </think>


def _generate_qwen3(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    """Qwen3 generation. Strips <think>...</think> blocks from output."""
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # disable thinking for speed; set True for reasoning tasks
    )
    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        generated = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    output_ids = generated[0][prompt_len:].tolist()

    # Strip thinking block if present
    try:
        think_end = len(output_ids) - output_ids[::-1].index(_QWEN3_THINK_TOKEN_ID)
    except ValueError:
        think_end = 0

    reply = _tokenizer.decode(output_ids[think_end:], skip_special_tokens=True)
    finish_reason = "length" if len(output_ids) >= max_tokens else "stop"
    return reply.strip(), finish_reason


def _generate_glm(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    """GLM-4.7-Flash generation. Standard chat template, no special tokens."""
    inputs = _tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    output_ids = generated[0][prompt_len:]
    # Log raw token ids to diagnose empty output issues (GLM-4.7-Flash MoE)
    logger.info(f"GLM output token count: {len(output_ids)}, ids[:10]: {output_ids[:10].tolist()}")
    reply = _tokenizer.decode(output_ids, skip_special_tokens=True)
    logger.info(f"GLM decoded reply: {repr(reply[:200])}")
    finish_reason = "length" if len(output_ids) >= max_tokens else "stop"
    return reply.strip(), finish_reason


# ---------------------------------------------------------------------------
# generate_reply dispatch - the function students replace in skeleton_server.py
# ---------------------------------------------------------------------------


def generate_reply(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    **kwargs: object,
) -> tuple[str, str]:
    assert _model is not None, "model not loaded"
    assert _tokenizer is not None, "tokenizer not loaded"

    with _lock:  # serialize requests - no batching in the naive baseline
        if "qwen3" in model.lower() or "qwen2" in model.lower():
            return _generate_qwen3(messages, max_tokens, temperature)
        elif "glm" in model.lower():
            return _generate_glm(messages, max_tokens, temperature)
        else:
            # Fallback: try standard chat template, may not work for all models
            logger.warning(f"Unknown model family {model!r}, using generic generation path")
            return _generate_glm(messages, max_tokens, temperature)


# ---------------------------------------------------------------------------
# HTTP server (same structure as skeleton_server.py)
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
        except Exception as e:
            logger.exception("generate_reply raised")
            raise HTTPException(status_code=500, detail=str(e)) from e

        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_tokens = (
            len(_tokenizer.encode(prompt_text)) if _tokenizer else len(prompt_text.split())
        )
        completion_tokens = len(_tokenizer.encode(reply)) if _tokenizer else len(reply.split())

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
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    _load_model(args.model)

    app = build_app(model_name=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
