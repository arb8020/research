"""Inference engine for LLM serving.

Designed for evaluations and RL rollouts. Alternative to vLLM/SGLang
with explicit state management and no global context.

Architecture derived from mini-sglang (Apache 2.0).

Two engine variants:
- InferenceEngine (engine.py): HuggingFace model backend (simple, compatible)
- InferenceEngineV2 (engine_v2.py): Custom model with FlashAttention (fast)

Usage:
    from rollouts.inference import InferenceEngineV2, EngineConfig, SamplingParams

    engine = InferenceEngineV2(EngineConfig(model_path="..."))
    results = engine.generate(["Hello, world"], SamplingParams(max_tokens=10))
    for req in results:
        print(req.input_ids)
    engine.shutdown()
"""

from .core import (
    Batch,
    Req,
    SamplingParams,
    create_req,
    make_batch,
    req_after_decode_step,
    req_after_forward,
    req_append_token,
)
from .engine import EngineConfig, InferenceEngine
from .engine_v2 import InferenceEngineV2

# CUDA graph capture (functional API)
from .graph import (
    GraphBuffers,
    can_use_graph,
    capture_graphs,
    replay_graph,
)
from .kv_cache import (
    CacheConfig,
    KVCachePool,
    RequestCache,
    empty_request_cache,
    extend_request_cache,
)

# Overlap scheduling (functional API)
from .overlap import (
    ForwardInput,
    ForwardOutput,
    create_forward_output,
    get_or_create_streams,
    overlap_step,
    run_overlap_loop,
)

# Radix cache (functional API)
from .radix import (
    CacheHandle,
    RadixNode,
    evict,
    get_stats,
    init_radix_state,
    insert_prefix,
    lock,
    match_prefix,
    unlock,
)
from .scheduler import (
    SchedulerConfig,
    SchedulerState,
    add_request,
    empty_scheduler_state,
    has_pending_work,
    schedule_step,
    update_after_forward,
)

__all__ = [
    # Core types
    "Batch",
    "Req",
    "SamplingParams",
    # Core functions
    "create_req",
    "make_batch",
    "req_after_decode_step",
    "req_after_forward",
    "req_append_token",
    # Engine
    "EngineConfig",
    "InferenceEngine",
    "InferenceEngineV2",
    # KV Cache
    "CacheConfig",
    "KVCachePool",
    "RequestCache",
    "empty_request_cache",
    "extend_request_cache",
    # Scheduler
    "SchedulerConfig",
    "SchedulerState",
    "add_request",
    "empty_scheduler_state",
    "has_pending_work",
    "schedule_step",
    "update_after_forward",
    # Radix cache (functional API)
    "CacheHandle",
    "RadixNode",
    "init_radix_state",
    "match_prefix",
    "lock",
    "unlock",
    "insert_prefix",
    "evict",
    "get_stats",
    # CUDA graph (functional API)
    "GraphBuffers",
    "capture_graphs",
    "can_use_graph",
    "replay_graph",
    # Overlap scheduling (functional API)
    "ForwardInput",
    "ForwardOutput",
    "create_forward_output",
    "get_or_create_streams",
    "overlap_step",
    "run_overlap_loop",
]
