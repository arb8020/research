"""Integration test for inference engine.

Verifies our engine produces same outputs as HuggingFace.
Run with GPU: python -m rollouts.inference.test_engine
"""

from __future__ import annotations

import torch


def test_greedy_equivalence():
    """Test that greedy decoding matches HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .core import SamplingParams
    from .engine import EngineConfig, InferenceEngine

    # Use a small model for testing
    model_name = "HuggingFaceTB/SmolLM2-135M"
    prompt = "The capital of France is"
    max_tokens = 10

    print(f"Testing with {model_name}")
    print(f"Prompt: {prompt!r}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Our engine
    print("Running our engine...")
    config = EngineConfig(
        model_path=model_name,
        max_batch_size=1,
        max_tokens_per_batch=512,
        max_seq_len=512,
    )
    engine = InferenceEngine(config)

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    finished = engine.generate([prompt], params)

    assert len(finished) == 1
    our_output = finished[0].input_ids.tolist()
    our_text = engine.tokenizer.decode(our_output)

    print(f"Our output tokens: {our_output}")
    print(f"Our output text: {our_text!r}")
    print()

    engine.shutdown()

    # HuggingFace reference
    print("Running HuggingFace reference...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(hf_model.device)

    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=hf_tokenizer.eos_token_id,
        )

    hf_tokens = hf_output[0].tolist()
    hf_text = hf_tokenizer.decode(hf_tokens)

    print(f"HF output tokens: {hf_tokens}")
    print(f"HF output text: {hf_text!r}")
    print()

    # Compare
    if our_output == hf_tokens:
        print("PASS: Outputs match exactly!")
        return True
    else:
        print("FAIL: Outputs differ")
        print(f"  Our length: {len(our_output)}")
        print(f"  HF length: {len(hf_tokens)}")

        # Find first difference
        for i, (a, b) in enumerate(zip(our_output, hf_tokens, strict=False)):
            if a != b:
                print(f"  First difference at position {i}: ours={a}, HF={b}")
                break

        return False


def test_multiple_requests():
    """Test batching multiple requests."""
    from .core import SamplingParams
    from .engine import EngineConfig, InferenceEngine

    model_name = "HuggingFaceTB/SmolLM2-135M"
    prompts = [
        "Hello, my name is",
        "The weather today is",
        "Python is a",
    ]

    print(f"Testing multiple requests with {model_name}")
    print(f"Prompts: {prompts}")
    print()

    config = EngineConfig(
        model_path=model_name,
        max_batch_size=8,
        max_tokens_per_batch=512,
        max_seq_len=512,
    )
    engine = InferenceEngine(config)

    params = SamplingParams(temperature=0.0, max_tokens=5)
    finished = engine.generate(prompts, params)

    assert len(finished) == len(prompts), f"Expected {len(prompts)} results, got {len(finished)}"

    print("Results:")
    for req in finished:
        text = engine.tokenizer.decode(req.input_ids.tolist())
        print(f"  uid={req.uid}: {text!r}")

    engine.shutdown()
    print()
    print("PASS: Multiple requests completed")
    return True


def test_scheduler_state_immutability():
    """Test that scheduler state is truly immutable."""
    from .core import SamplingParams, create_req
    from .scheduler import add_request, empty_scheduler_state

    print("Testing scheduler state immutability...")

    state1 = empty_scheduler_state()
    req = create_req(
        uid=0,
        prompt_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        table_idx=0,
    )

    state2 = add_request(state1, req)

    # state1 should be unchanged
    assert len(state1.prefill_queue) == 0, "state1 was mutated!"
    assert len(state2.prefill_queue) == 1, "state2 should have request"

    print("PASS: State is immutable")
    return True


def test_scheduler_prefill_avoids_head_of_line_blocking():
    """Test prefill scheduling skips oversized head request."""
    from .core import SamplingParams, create_req
    from .scheduler import SchedulerConfig, add_request, empty_scheduler_state, schedule_prefill

    state = empty_scheduler_state()
    oversized = create_req(
        uid=0,
        prompt_ids=[1, 2, 3, 4, 5],  # extend_len=5
        sampling_params=SamplingParams(max_tokens=1),
        table_idx=0,
    )
    small = create_req(
        uid=1,
        prompt_ids=[9, 8],  # extend_len=2
        sampling_params=SamplingParams(max_tokens=1),
        table_idx=1,
    )
    state = add_request(state, oversized)
    state = add_request(state, small)

    config = SchedulerConfig(max_batch_size=4, max_tokens_per_batch=4, max_seq_len=32)

    def allocate_pages(n: int):
        return torch.arange(n, dtype=torch.int32)

    result = schedule_prefill(
        state=state,
        config=config,
        num_free_pages=32,
        device=torch.device("cpu"),
        allocate_pages=allocate_pages,
    )

    assert result.batch is not None
    scheduled_uids = {req.uid for req in result.batch.reqs}
    assert scheduled_uids == {1}, f"Expected only small request scheduled, got {scheduled_uids}"
    assert tuple(req.uid for req in result.new_state.prefill_queue) == (0,)
    return True


def test_chunked_prefill_state_transition():
    """Test chunked prefill updates state without decoding intermediate chunks."""
    from .chunked_prefill import ChunkedPrefillManager
    from .core import SamplingParams, create_req, make_batch
    from .engine_v2 import InferenceEngineV2
    from .scheduler import SchedulerState

    params = SamplingParams(temperature=0.0, max_tokens=2)

    req_first_chunk = create_req(
        uid=7,
        prompt_ids=[10, 11],
        sampling_params=params,
        table_idx=0,
    )
    req_full_prompt = create_req(
        uid=7,
        prompt_ids=[10, 11, 12, 13],
        sampling_params=params,
        table_idx=0,
    )

    engine = InferenceEngineV2.__new__(InferenceEngineV2)
    engine.eos_token_id = 2
    engine._prefill_chunk_size = 2
    engine._chunked_prefill_mgr = ChunkedPrefillManager(2)
    engine._chunked_prefill_mgr.maybe_chunk(req_full_prompt)
    engine._chunk_pending_tokens = {7: torch.tensor([12, 13], dtype=torch.int32)}

    batch1 = make_batch(
        reqs=(req_first_chunk,),
        phase="prefill",
        out_loc=torch.tensor([0, 1], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    state1 = SchedulerState(
        prefill_queue=(),
        decode_set=frozenset({req_first_chunk}),
        finished=(),
    )
    updated1 = engine._update_state_after_batch(
        state=state1,
        batch=batch1,
        next_tokens=torch.tensor([99], dtype=torch.int32),
    )

    assert len(updated1.finished) == 0
    assert len(updated1.decode_set) == 0
    assert len(updated1.prefill_queue) == 1
    queued_req = updated1.prefill_queue[0]
    assert queued_req.uid == 7
    assert queued_req.cached_len == 2
    assert queued_req.input_ids.tolist() == [10, 11, 12, 13]

    batch2 = make_batch(
        reqs=(queued_req,),
        phase="prefill",
        out_loc=torch.tensor([2, 3], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    state2 = SchedulerState(
        prefill_queue=(),
        decode_set=frozenset({queued_req}),
        finished=(),
    )
    updated2 = engine._update_state_after_batch(
        state=state2,
        batch=batch2,
        next_tokens=torch.tensor([77], dtype=torch.int32),
    )

    assert len(updated2.prefill_queue) == 0
    assert len(updated2.finished) == 0
    assert len(updated2.decode_set) == 1
    final_req = next(iter(updated2.decode_set))
    assert final_req.uid == 7
    assert final_req.cached_len == 4
    assert final_req.input_ids.tolist() == [10, 11, 12, 13, 77]
    return True


def test_kv_cache_correctness():
    """Test that KV cache produces correct output.

    Verifies that our cached forward passes produce the same
    logits as HuggingFace's generate with use_cache=True.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .core import SamplingParams
    from .engine import EngineConfig, InferenceEngine

    model_name = "HuggingFaceTB/SmolLM2-135M"
    prompt = "Once upon a time"
    max_tokens = 5

    print(f"Testing KV cache correctness with {model_name}")
    print(f"Prompt: {prompt!r}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Our engine with KV cache
    print("Running our engine (with KV cache)...")
    config = EngineConfig(
        model_path=model_name,
        max_batch_size=1,
        max_tokens_per_batch=512,
        max_seq_len=512,
    )
    engine = InferenceEngine(config)

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    finished = engine.generate([prompt], params)

    assert len(finished) == 1
    our_output = finished[0].input_ids.tolist()
    our_text = engine.tokenizer.decode(our_output)

    print(f"Our output: {our_text!r}")
    print(f"Our tokens: {our_output}")
    print()

    engine.shutdown()

    # HuggingFace reference with use_cache=True (their default)
    print("Running HuggingFace reference (with cache)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(hf_model.device)

    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=hf_tokenizer.eos_token_id,
        )

    hf_tokens = hf_output[0].tolist()
    hf_text = hf_tokenizer.decode(hf_tokens)

    print(f"HF output: {hf_text!r}")
    print(f"HF tokens: {hf_tokens}")
    print()

    # Compare
    if our_output == hf_tokens:
        print("PASS: KV cache outputs match HuggingFace!")
        return True
    else:
        print("FAIL: Outputs differ")
        print(f"  Our length: {len(our_output)}")
        print(f"  HF length: {len(hf_tokens)}")

        for i, (a, b) in enumerate(zip(our_output, hf_tokens, strict=False)):
            if a != b:
                print(f"  First difference at position {i}: ours={a}, HF={b}")
                break

        return False


def test_kv_cache_state_tracking():
    """Test that request cache state is tracked correctly."""
    from .kv_cache import (
        CacheConfig,
        KVCachePool,
        empty_request_cache,
        extend_request_cache,
    )

    print("Testing KV cache state tracking...")

    config = CacheConfig(
        num_layers=2,
        num_heads=4,
        head_dim=16,
        num_slots=100,
    )
    device = torch.device("cpu")
    pool = KVCachePool(config, device)

    # Test allocation
    slots1 = pool.allocate_slots(5)
    assert len(slots1) == 5
    assert pool.num_free_slots == 95

    slots2 = pool.allocate_slots(3)
    assert len(slots2) == 3
    assert pool.num_free_slots == 92

    # Test request cache
    cache = empty_request_cache(uid=0, device=device)
    assert cache.cached_len == 0

    cache = extend_request_cache(cache, slots1)
    assert cache.cached_len == 5

    cache = extend_request_cache(cache, slots2)
    assert cache.cached_len == 8

    # Test reset
    pool.reset()
    assert pool.num_free_slots == 100

    print("PASS: KV cache state tracking works")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Inference Engine Tests")
    print("=" * 60)
    print()

    results = []

    # Tests without GPU
    results.append(("immutability", test_scheduler_state_immutability()))
    print()
    results.append(("scheduler_no_hol", test_scheduler_prefill_avoids_head_of_line_blocking()))
    print()
    results.append(("chunked_prefill_state", test_chunked_prefill_state_transition()))
    print()
    results.append(("kv_cache_state", test_kv_cache_state_tracking()))
    print()

    # GPU tests
    if torch.cuda.is_available():
        results.append(("greedy_equivalence", test_greedy_equivalence()))
        print()
        results.append(("kv_cache_correctness", test_kv_cache_correctness()))
        print()
        results.append(("multiple_requests", test_multiple_requests()))
    else:
        print("Skipping GPU tests (no CUDA available)")
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    exit(0 if all_passed else 1)
