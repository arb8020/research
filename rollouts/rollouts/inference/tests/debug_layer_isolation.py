"""Debug tool to isolate first HF-vs-ours divergence layer.

Usage:
    uv run python -m rollouts.inference.tests.debug_layer_isolation
    uv run python -m rollouts.inference.tests.debug_layer_isolation --debug-mode
"""

from __future__ import annotations

import argparse

import torch

from ..attention.backend import build_attention_metadata
from ..attention.reference import ReferenceAttentionBackend
from ..kv_cache import CacheConfig, KVCachePool
from ..models.config import load_model_config
from ..models.llama import LlamaForCausalLM
from ..models.weight import load_weights, remap_weights_llama


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolate first mismatch layer vs HuggingFace")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument(
        "--prompt",
        default="Write me a very long fantasy story about a dragon and a lighthouse.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Compute dtype for both HF and our model",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Force fp32 + deterministic algorithms for tighter numeric debugging",
    )
    parser.add_argument(
        "--layer-threshold",
        type=float,
        default=5e-3,
        help="Report first layer with max abs diff above this threshold",
    )
    return parser.parse_args()


def _dtype_from_args(args: argparse.Namespace) -> torch.dtype:
    if args.debug_mode:
        return torch.float32
    return torch.float32 if args.dtype == "fp32" else torch.bfloat16


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is required for this debug tool"

    if args.debug_mode:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    dtype = _dtype_from_args(args)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device,
    )
    hf_model.eval()

    config = load_model_config(args.model)
    model = LlamaForCausalLM(config, device, dtype)
    model.to(device)
    hf_weights = load_weights(args.model, device, dtype)
    remapped = remap_weights_llama(hf_weights, config.num_hidden_layers)
    model.load_weights(remapped)
    model.eval()

    num_slots = max(8192, len(tokenizer.encode(args.prompt)) + args.max_new_tokens + 64)
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

    def our_last_logits(prefix_ids: torch.Tensor) -> torch.Tensor:
        kv_pool.reset()
        seq_len = prefix_ids.shape[1]
        flat_input_ids = prefix_ids.view(-1)
        positions = torch.arange(seq_len, device=device, dtype=torch.int32)
        out_loc = torch.arange(seq_len, dtype=torch.int32, device=device)
        page_table = torch.arange(num_slots, dtype=torch.int32, device=device).view(1, -1)
        metadata = build_attention_metadata(
            cached_lens=[0],
            extend_lens=[seq_len],
            page_table=page_table,
            device=device,
        )
        logits = model(
            input_ids=flat_input_ids,
            positions=positions,
            attn_backend=attn_backend,
            attn_metadata=metadata,
            out_loc=out_loc,
        )
        return logits[-1]

    prefix = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    mismatch_step: int | None = None
    hf_next = -1
    our_next = -1

    for step in range(args.max_new_tokens):
        hf_logits = hf_model(prefix).logits[0, -1]
        our_logits = our_last_logits(prefix)
        hf_next = int(hf_logits.argmax().item())
        our_next = int(our_logits.argmax().item())
        if hf_next != our_next:
            mismatch_step = step
            break
        prefix = torch.cat([prefix, torch.tensor([[hf_next]], device=device)], dim=1)

    if mismatch_step is None:
        print(f"No mismatch found in first {args.max_new_tokens} decode steps")
        return

    print(
        f"First mismatch at step={mismatch_step}, prefix_len={prefix.shape[1]}, "
        f"hf_next={hf_next}, our_next={our_next}"
    )

    hf_outputs: dict[str, torch.Tensor] = {}
    our_outputs: dict[str, torch.Tensor] = {}

    def make_hook(dst: dict[str, torch.Tensor], name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            dst[name] = tensor.detach()

        return hook

    handles = []
    handles.append(
        hf_model.model.embed_tokens.register_forward_hook(make_hook(hf_outputs, "embed"))
    )
    for i, layer in enumerate(hf_model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(hf_outputs, f"layer_{i}")))
    handles.append(hf_model.model.norm.register_forward_hook(make_hook(hf_outputs, "final_norm")))
    hf_last_logits = hf_model(prefix).logits[0, -1].float()
    for h in handles:
        h.remove()

    handles = []
    handles.append(model.model.embed_tokens.register_forward_hook(make_hook(our_outputs, "embed")))
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(our_outputs, f"layer_{i}")))
    handles.append(model.model.norm.register_forward_hook(make_hook(our_outputs, "final_norm")))
    our_last = our_last_logits(prefix).float()
    for h in handles:
        h.remove()

    first_layer_over = None
    for i in range(config.num_hidden_layers):
        hf_layer = hf_outputs[f"layer_{i}"][0, -1].float()
        our_layer = our_outputs[f"layer_{i}"][-1].float()
        diff = (hf_layer - our_layer).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        print(f"layer_{i:02d}: max_diff={max_diff:.3e} mean_diff={mean_diff:.3e}")
        if first_layer_over is None and max_diff > args.layer_threshold:
            first_layer_over = i

    print(f"first_layer_over_{args.layer_threshold:.1e}={first_layer_over}")

    logit_diff = (hf_last_logits - our_last).abs()
    print(
        f"logits: max_diff={float(logit_diff.max().item()):.3e} "
        f"mean_diff={float(logit_diff.mean().item()):.3e}"
    )
    hf_topv, hf_topi = torch.topk(hf_last_logits, 5)
    our_topv, our_topi = torch.topk(our_last, 5)
    print(f"hf_top5={hf_topi.tolist()} gap12={float((hf_topv[0] - hf_topv[1]).item()):.3e}")
    print(f"our_top5={our_topi.tolist()} gap12={float((our_topv[0] - our_topv[1]).item()):.3e}")


if __name__ == "__main__":
    main()
