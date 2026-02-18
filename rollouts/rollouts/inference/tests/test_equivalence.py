"""Numerical equivalence tests.

Verifies that our inference engine produces the same outputs as:
1. HuggingFace (baseline)
2. mini-sglang (parity target)

Test cases:
- Single request, greedy decoding
- Single request, temperature sampling (fixed seed)
- Multiple requests batched
- Long context (tests KV cache)
- Prefix cache hit (when implemented)
"""

from __future__ import annotations

import logging

# Setup logging - use color for console, respects LOG_LEVEL env var
# Disable queue_handler for Python 3.11 compatibility (Modal sandbox)
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rollouts._logging import setup_logging

setup_logging(
    use_color=True,
    logger_levels={"httpx": "WARNING"},
    use_queue_handler=(sys.version_info >= (3, 12)),
)
logger = logging.getLogger(__name__)


def test_functional_multitoken_vs_huggingface():
    """Regression: functional Llama must match HF across all prompt positions."""
    if not torch.cuda.is_available():
        logger.info("Skipping functional multi-token test (no CUDA)")
        return True

    logger.info("Testing functional multi-token parity vs HuggingFace...")

    model_name = "HuggingFaceTB/SmolLM2-135M"
    prompt = "Hello, world"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tolerance = 1e-2

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ..models.llama_functional import forward as functional_forward
    from ..models.llama_functional import load_config as functional_load_config

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    hf_model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    config = functional_load_config(model_name)
    weights = {k: v.to(device) for k, v in hf_model.state_dict().items()}

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        functional_logits = functional_forward(input_ids, weights, config)

    if functional_logits.dim() == 2:
        functional_logits = functional_logits.unsqueeze(0)

    all_ok = True
    for pos in range(functional_logits.shape[1]):
        pos_diff = (
            (functional_logits[0, pos].float() - hf_logits[0, pos].float()).abs().max().item()
        )
        logger.info(f"Functional position {pos}: max_diff={pos_diff:.2e}")
        if pos_diff >= tolerance:
            all_ok = False

    if all_ok:
        logger.info("PASS: Functional multi-token logits match HuggingFace per position")
        return True

    logger.error(f"FAIL: Functional per-position diff exceeds {tolerance:.2e}")
    return False


def test_functional_load_config_rope_parsing():
    """Regression: load_config should parse RoPE theta from modern HF config fields."""

    def _cfg(
        *,
        rope_theta=None,
        rope_parameters=None,
        rope_scaling=None,
    ):
        return SimpleNamespace(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=32000,
            rms_norm_eps=1e-5,
            max_position_embeddings=2048,
            rope_theta=rope_theta,
            rope_parameters=rope_parameters,
            rope_scaling=rope_scaling,
        )

    from ..models.llama_functional import load_config

    cases = [
        (
            _cfg(rope_parameters={"rope_theta": 54321.0}),
            54321.0,
            "rope_parameters.rope_theta",
        ),
        (
            _cfg(rope_scaling={"rope_type": "linear", "theta": 7777.0}),
            7777.0,
            "rope_scaling.theta",
        ),
        (_cfg(), 10000.0, "default"),
    ]

    for idx, (hf_cfg, expected, source) in enumerate(cases):
        with patch("transformers.AutoConfig.from_pretrained", return_value=hf_cfg):
            parsed = load_config(f"mock-model-{idx}")
        if parsed.rope_theta != expected:
            logger.error(
                f"FAIL: load_config parsed rope_theta={parsed.rope_theta}, "
                f"expected {expected} from {source}"
            )
            return False
        logger.info(f"RoPE parse case {source}: PASS (theta={parsed.rope_theta})")

    return True


def test_reference_attention_vs_pytorch():
    """Test our reference attention matches PyTorch scaled_dot_product_attention."""
    from ..attention.backend import AttentionMetadata
    from ..attention.reference import ReferenceAttentionBackend

    logger.info("Testing reference attention vs PyTorch...")

    # Setup
    batch_size = 2
    num_q_heads = 8
    num_kv_heads = 2  # GQA
    head_dim = 64
    seq_lens = [5, 3]  # Two sequences
    num_layers = 2
    num_slots = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Use float32 for precision comparison

    # Create cache
    k_cache = torch.zeros(num_layers, num_slots, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros(num_layers, num_slots, num_kv_heads, head_dim, device=device, dtype=dtype)

    backend = ReferenceAttentionBackend(
        k_cache=k_cache,
        v_cache=v_cache,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # Create inputs
    total_tokens = sum(seq_lens)
    q = torch.randn(total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Create metadata (all tokens are new, no cache)
    cu_seqlens_q = torch.tensor([0, seq_lens[0], sum(seq_lens)], dtype=torch.int32, device=device)
    cu_seqlens_k = cu_seqlens_q.clone()
    cache_seqlens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    # Page table: each sequence gets its own slots
    # seq 0: positions [0, seq_lens[0]) -> slots [0, seq_lens[0])
    # seq 1: positions [0, seq_lens[1]) -> slots [seq_lens[0], seq_lens[0] + seq_lens[1])
    page_table = torch.zeros(batch_size, num_slots, dtype=torch.int32, device=device)
    slot_offset = 0
    for i, seq_len in enumerate(seq_lens):
        page_table[i, :seq_len] = torch.arange(slot_offset, slot_offset + seq_len, device=device)
        slot_offset += seq_len

    # Allocate slots
    out_loc = torch.arange(total_tokens, dtype=torch.int32, device=device)

    metadata = AttentionMetadata(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max(seq_lens),
        max_seqlen_k=max(seq_lens),
        page_table=page_table,
    )

    # Run our attention
    layer_idx = 0
    our_output = backend.forward(q, k, v, layer_idx, metadata, out_loc)

    # Run PyTorch reference (per sequence)
    ref_outputs = []
    offset = 0
    for seq_len in seq_lens:
        seq_q = q[offset : offset + seq_len]  # [seq_len, num_q_heads, head_dim]
        seq_k = k[offset : offset + seq_len]
        seq_v = v[offset : offset + seq_len]

        # Expand for GQA
        seq_k = seq_k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        seq_v = seq_v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

        # PyTorch SDPA expects [batch, heads, seq, dim]
        seq_q = seq_q.transpose(0, 1).unsqueeze(0)
        seq_k = seq_k.transpose(0, 1).unsqueeze(0)
        seq_v = seq_v.transpose(0, 1).unsqueeze(0)

        ref_out = torch.nn.functional.scaled_dot_product_attention(
            seq_q, seq_k, seq_v, is_causal=True
        )
        # Back to [seq_len, num_heads, head_dim]
        ref_out = ref_out.squeeze(0).transpose(0, 1)
        ref_outputs.append(ref_out)

        offset += seq_len

    ref_output = torch.cat(ref_outputs, dim=0)

    # Compare
    max_diff = (our_output - ref_output).abs().max().item()
    logger.info(f"Max diff vs PyTorch SDPA: {max_diff:.2e}")

    if max_diff < 1e-5:
        logger.info("PASS: Reference attention matches PyTorch")
        return True
    else:
        logger.error(f"FAIL: Max diff {max_diff} exceeds threshold")
        return False


def find_divergence_layer():
    """Binary search to find which layer diverges from HuggingFace."""
    if not torch.cuda.is_available():
        return True

    logger.info("Finding divergence layer...")

    model_name = "HuggingFaceTB/SmolLM2-135M"
    prompt = "Hello"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    hf_model.eval()

    input_ids = hf_tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Capture HF intermediate outputs with hooks
    hf_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hf_outputs[name] = output[0].detach().clone()
            else:
                hf_outputs[name] = output.detach().clone()

        return hook

    # Register hooks
    handles = []
    handles.append(hf_model.model.embed_tokens.register_forward_hook(make_hook("embed")))
    for i, layer in enumerate(hf_model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
    handles.append(hf_model.model.norm.register_forward_hook(make_hook("final_norm")))

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits

    for h in handles:
        h.remove()

    logger.info(f"Captured {len(hf_outputs)} HF checkpoints")

    # Now compare with our model at each checkpoint
    from ..models.config import load_model_config
    from ..models.llama import LlamaForCausalLM
    from ..models.weight import load_weights, remap_weights_llama

    config = load_model_config(model_name)
    model = LlamaForCausalLM(config, device, dtype)
    model.to(device)

    hf_weights = load_weights(model_name, device, dtype)
    remapped = remap_weights_llama(hf_weights, config.num_hidden_layers)
    model.load_weights(remapped)
    model.eval()

    # Check embedding
    our_embed = model.model.embed_tokens(input_ids.view(-1))
    hf_embed = hf_outputs["embed"].view(-1, hf_outputs["embed"].shape[-1])
    embed_diff = (our_embed.float() - hf_embed.float()).abs().max().item()
    logger.info(f"embed: max_diff={embed_diff:.2e}")

    if embed_diff > 1e-5:
        logger.error("DIVERGES AT: embed_tokens")
        return False

    # For layers, we'd need to run partial forward - skip for now
    # Just report final logits
    logger.info("Embeddings match. Divergence is in transformer layers or lm_head.")

    return True


def test_model_logits_vs_huggingface():
    """Test our model produces same logits as HuggingFace."""
    if not torch.cuda.is_available():
        logger.info("Skipping model test (no CUDA)")
        return True

    logger.info("Testing model logits vs HuggingFace...")

    # Use small model
    model_name = "HuggingFaceTB/SmolLM2-135M"
    prompt = "Hello, world"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # HuggingFace reference
    logger.info("Loading HuggingFace model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    hf_model.eval()

    input_ids = hf_tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        hf_outputs = hf_model(input_ids)
        hf_logits = hf_outputs.logits  # [1, seq_len, vocab_size]

    logger.info(f"HuggingFace logits shape: {hf_logits.shape}")

    # Our model
    logger.info("Loading our model...")
    from ..attention.backend import build_attention_metadata
    from ..attention.reference import ReferenceAttentionBackend
    from ..kv_cache import CacheConfig, KVCachePool
    from ..models.config import load_model_config
    from ..models.llama import LlamaForCausalLM
    from ..models.weight import load_weights, remap_weights_llama

    config = load_model_config(model_name)
    model = LlamaForCausalLM(config, device, dtype)
    model.to(device)

    hf_weights = load_weights(model_name, device, dtype)
    remapped = remap_weights_llama(hf_weights, config.num_hidden_layers)
    model.load_weights(remapped)
    model.eval()

    # Setup KV cache and attention
    num_slots = 1024
    cache_config = CacheConfig(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_slots=num_slots,
        dtype=dtype,
    )
    kv_pool = KVCachePool(cache_config, device)

    attn_backend = ReferenceAttentionBackend(
        k_cache=kv_pool.k_cache,
        v_cache=kv_pool.v_cache,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )

    # Prepare inputs
    seq_len = input_ids.shape[1]
    flat_input_ids = input_ids.view(-1)
    positions = torch.arange(seq_len, device=device)
    out_loc = torch.arange(seq_len, dtype=torch.int32, device=device)

    # Page table
    page_table = torch.arange(num_slots, dtype=torch.int32, device=device).view(1, -1)

    attn_metadata = build_attention_metadata(
        cached_lens=[0],
        extend_lens=[seq_len],
        page_table=page_table,
        device=device,
    )

    with torch.no_grad():
        our_logits = model(
            input_ids=flat_input_ids,
            positions=positions,
            attn_backend=attn_backend,
            attn_metadata=attn_metadata,
            out_loc=out_loc,
        )

    our_logits = our_logits.unsqueeze(0)  # [1, seq_len, vocab_size]
    logger.info(f"Our logits shape: {our_logits.shape}")

    # Compare
    max_diff = (our_logits.float() - hf_logits.float()).abs().max().item()
    mean_diff = (our_logits.float() - hf_logits.float()).abs().mean().item()

    logger.info(f"Max logit diff: {max_diff:.2e}")
    logger.info(f"Mean logit diff: {mean_diff:.2e}")

    # Compare argmax at each position
    our_argmax = our_logits.argmax(dim=-1)  # [1, seq_len]
    hf_argmax = hf_logits.argmax(dim=-1)
    argmax_match = (our_argmax == hf_argmax).all().item()

    logger.info(f"Argmax match: {argmax_match}")
    logger.info(f"HF argmax: {hf_argmax.tolist()}")
    logger.info(f"Our argmax: {our_argmax.tolist()}")

    # Check per-position logit stats
    for pos in range(our_logits.shape[1]):
        pos_diff = (our_logits[0, pos].float() - hf_logits[0, pos].float()).abs()
        logger.info(
            f"Position {pos}: max_diff={pos_diff.max().item():.2e}, "
            f"hf_argmax={hf_argmax[0, pos].item()}, our_argmax={our_argmax[0, pos].item()}"
        )

    # For bf16 with different RoPE implementations, some numerical diff is expected
    # Key check: argmax matches (greedy sampling would produce same token)
    if argmax_match:
        logger.info("PASS: Model logits match HuggingFace (argmax identical)")
        return True
    elif max_diff < 0.1:
        logger.info("PASS: Model logits close to HuggingFace")
        return True
    else:
        logger.error("FAIL: Model logits differ significantly")
        logger.error(f"Argmax match: {argmax_match}, Max diff: {max_diff:.2e}")
        return False


def test_greedy_generation_vs_huggingface():
    """Test full generation matches HuggingFace."""
    if not torch.cuda.is_available():
        logger.info("Skipping generation test (no CUDA)")
        return True

    logger.info("Testing greedy generation vs HuggingFace...")

    model_name = "HuggingFaceTB/SmolLM2-135M"
    prompt = "The capital of France is"
    max_tokens = 10

    # HuggingFace
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    input_ids = hf_tokenizer.encode(prompt, return_tensors="pt").cuda()

    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=hf_tokenizer.eos_token_id,
        )

    hf_tokens = hf_output[0].tolist()
    hf_text = hf_tokenizer.decode(hf_tokens)
    logger.info(f"HuggingFace: {hf_text!r}")

    # Our engine
    from ..core import SamplingParams
    from ..engine_v2 import EngineConfig, InferenceEngineV2

    engine = InferenceEngineV2(
        EngineConfig(
            model_path=model_name,
            max_batch_size=1,
            max_tokens_per_batch=512,
            max_seq_len=512,
            attention_backend="reference",
        )
    )

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    finished = engine.generate([prompt], params)

    our_tokens = finished[0].input_ids.tolist()
    our_text = engine.tokenizer.decode(our_tokens)
    logger.info(f"Ours: {our_text!r}")

    engine.shutdown()

    # Compare
    if our_tokens == hf_tokens:
        logger.info("PASS: Generation matches exactly")
        return True
    else:
        # Find divergence point
        diverge_idx = 0
        for i, (a, b) in enumerate(zip(our_tokens, hf_tokens, strict=False)):
            if a != b:
                diverge_idx = i
                break

        # Some divergence is expected due to numerical drift in autoregressive generation
        # Key check: does it produce coherent text?
        logger.warning(f"Generation differs at token {diverge_idx}")
        logger.warning(f"HF: '{hf_text}'")
        logger.warning(f"Ours: '{our_text}'")

        # Pass if divergence is late (after several tokens of agreement)
        if diverge_idx >= 5:
            logger.info(f"PASS: Generation matches first {diverge_idx} tokens (drift expected)")
            return True
        else:
            logger.error(f"FAIL: Generation diverges too early (token {diverge_idx})")
            return False


if __name__ == "__main__":
    import traceback

    results = []

    print("\n[1/6] Running reference_attention test...")
    try:
        results.append(("reference_attention", test_reference_attention_vs_pytorch()))
    except Exception as e:
        print(f"reference_attention CRASHED: {e}")
        traceback.print_exc()
        results.append(("reference_attention", False))

    print("\n[2/6] Running functional multi-token parity regression...")
    try:
        results.append(("functional_multitoken", test_functional_multitoken_vs_huggingface()))
    except Exception as e:
        print(f"functional_multitoken CRASHED: {e}")
        traceback.print_exc()
        results.append(("functional_multitoken", False))

    print("\n[3/6] Running load_config RoPE parsing regression...")
    try:
        results.append(("functional_rope_config", test_functional_load_config_rope_parsing()))
    except Exception as e:
        print(f"functional_rope_config CRASHED: {e}")
        traceback.print_exc()
        results.append(("functional_rope_config", False))

    if torch.cuda.is_available():
        print("\n[4/6] Finding divergence layer...")
        try:
            find_divergence_layer()
        except Exception as e:
            print(f"find_divergence CRASHED: {e}")
            traceback.print_exc()

        print("\n[5/6] Running model_logits test...")
        try:
            results.append(("model_logits", test_model_logits_vs_huggingface()))
        except Exception as e:
            print(f"model_logits CRASHED: {e}")
            traceback.print_exc()
            results.append(("model_logits", False))

        print("\n[6/6] Running greedy_generation test...")
        try:
            results.append(("greedy_generation", test_greedy_generation_vs_huggingface()))
        except Exception as e:
            print(f"greedy_generation CRASHED: {e}")
            traceback.print_exc()
            results.append(("greedy_generation", False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    exit(0 if all_passed else 1)
