#!/usr/bin/env python3
"""Test functional GLM-4.5 MoE implementation against HuggingFace model.

Downloads only the required shards for testing partial layers.

Run locally (requires GPU with ~40GB VRAM for 5 layers):
    python -m tools.functional_extractor.test_glm4_moe

Run on remote GPU:
    python -m tools.functional_extractor.test_glm4_moe --remote
    python -m tools.functional_extractor.test_glm4_moe --remote --gpu-id <id>

Options:
    --num-layers N    Number of layers to test (default: 5)
    --layer-by-layer  Test each layer individually to find divergence
"""

from __future__ import annotations

import argparse
import sys


def download_partial_weights(num_layers: int) -> dict:
    """Download only the shards needed for testing num_layers.

    Shard mapping for GLM-4.5:
    - Shard 1: embed_tokens + layer 0 (3.75 GB)
    - Shard 2: layer 1 (0.65 GB)
    - Shard 3: layer 2 (0.65 GB)
    - Shard 4+: MoE layers 3+ (7.87 GB each)
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    # Determine which shards we need
    # Layer 0 -> shard 1, Layer 1 -> shard 2, etc.
    # But we also need the final norm weight which is in the last shard
    shards_needed = num_layers + 1  # +1 because layer 0 is in shard 1

    print(f"\nDownloading shards 1-{shards_needed} for {num_layers} layers...")

    weights = {}

    for shard_idx in range(1, shards_needed + 1):
        shard_name = f"model-{shard_idx:05d}-of-00093.safetensors"
        print(f"  Downloading {shard_name}...")

        shard_path = hf_hub_download(
            "zai-org/GLM-4.5",
            filename=shard_name,
        )

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    # Also need the final norm weight - it's in shard 93
    # But for partial testing, we can create a dummy or skip the final forward
    # Actually, let's check if model.norm.weight is in one of our shards
    if "model.norm.weight" not in weights:
        # model.norm.weight is in shard 92 (not 93!)
        print("  Downloading final norm weight from shard 92...")
        shard_path = hf_hub_download(
            "zai-org/GLM-4.5",
            filename="model-00092-of-00093.safetensors",
        )
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            weights["model.norm.weight"] = f.get_tensor("model.norm.weight")
            print("    Loaded model.norm.weight")

    return weights


def debug_moe_layer(hf_model, weights_gpu, layer_input, layer_idx: int, cos, sin) -> None:
    """Debug MoE layer to find exact divergence point.

    Note: layer_input is the input to the FULL layer (before attention).
    We need to run attention first to get the MoE input.
    """
    import torch
    import torch.nn.functional as F
    from glm4_moe_functional import (
        N_ROUTED_EXPERTS,
        attention,
        dense_mlp,
        expert_forward,
        moe_router,
        rms_norm,
    )

    print(f"\n  ### Debugging MoE Layer {layer_idx} ###")
    prefix = f"model.layers.{layer_idx}.mlp"
    hf_layer = hf_model.model.layers[layer_idx]
    hf_moe = hf_layer.mlp

    # First, compute the MoE input by running attention
    # MoE input = post_attention_layernorm(layer_input + attention(input_layernorm(layer_input)))
    with torch.no_grad():
        residual = layer_input
        normed = rms_norm(
            layer_input, weights_gpu[f"model.layers.{layer_idx}.input_layernorm.weight"]
        )
        attn_out = attention(
            normed,
            q_weight=weights_gpu[f"model.layers.{layer_idx}.self_attn.q_proj.weight"],
            q_bias=weights_gpu[f"model.layers.{layer_idx}.self_attn.q_proj.bias"],
            k_weight=weights_gpu[f"model.layers.{layer_idx}.self_attn.k_proj.weight"],
            k_bias=weights_gpu[f"model.layers.{layer_idx}.self_attn.k_proj.bias"],
            v_weight=weights_gpu[f"model.layers.{layer_idx}.self_attn.v_proj.weight"],
            v_bias=weights_gpu[f"model.layers.{layer_idx}.self_attn.v_proj.bias"],
            o_weight=weights_gpu[f"model.layers.{layer_idx}.self_attn.o_proj.weight"],
            q_norm_weight=weights_gpu[f"model.layers.{layer_idx}.self_attn.q_norm.weight"],
            k_norm_weight=weights_gpu[f"model.layers.{layer_idx}.self_attn.k_norm.weight"],
            cos=cos,
            sin=sin,
        )
        post_attn = residual + attn_out
        # This is the input to post_attention_layernorm
        moe_input_normed = rms_norm(
            post_attn, weights_gpu[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
        )

    # Now debug using moe_input_normed
    hidden_states = moe_input_normed
    batch_size, seq_len, hidden_size = hidden_states.shape
    hidden_flat = hidden_states.view(-1, hidden_size)

    with torch.no_grad():
        # Compare router outputs
        hf_router_logits = hf_moe.gate(hidden_states)
        func_router_logits = F.linear(
            hidden_flat.float(), weights_gpu[f"{prefix}.gate.weight"].float()
        )
        router_diff = (
            (hf_router_logits.view(-1, N_ROUTED_EXPERTS) - func_router_logits).abs().max().item()
        )
        print(f"    Router logits: max_diff={router_diff:.2e}")

        # Compare routing decisions
        hf_topk_idx, hf_topk_weights = hf_moe.route_tokens_to_experts(hf_router_logits)
        func_topk_idx, func_topk_weights = moe_router(
            hidden_flat,
            router_weight=weights_gpu[f"{prefix}.gate.weight"],
            e_score_correction_bias=weights_gpu[f"{prefix}.gate.e_score_correction_bias"],
        )

        # Check if same experts are selected
        hf_sorted = hf_topk_idx.sort(dim=-1)[0]
        func_sorted = func_topk_idx.sort(dim=-1)[0]
        same_experts = (hf_sorted == func_sorted).all().item()
        print(f"    Same experts selected: {same_experts}")
        if not same_experts:
            print(f"      HF experts: {hf_topk_idx[0].tolist()}")
            print(f"      Func experts: {func_topk_idx[0].tolist()}")

        # Compare routing weights
        weight_diff = (hf_topk_weights - func_topk_weights).abs().max().item()
        print(f"    Router weights: max_diff={weight_diff:.2e}")
        print(f"      HF weights sample: {hf_topk_weights[0, :3].tolist()}")
        print(f"      Func weights sample: {func_topk_weights[0, :3].tolist()}")

        # Compare shared expert output
        hf_shared_out = hf_moe.shared_experts(hidden_states)
        func_shared_out = dense_mlp(
            hidden_states,
            gate_weight=weights_gpu[f"{prefix}.shared_experts.gate_proj.weight"],
            up_weight=weights_gpu[f"{prefix}.shared_experts.up_proj.weight"],
            down_weight=weights_gpu[f"{prefix}.shared_experts.down_proj.weight"],
        )
        shared_diff = (hf_shared_out - func_shared_out).abs().max().item()
        print(f"    Shared expert: max_diff={shared_diff:.2e}")

        # Compare routed expert output
        expert_gate_weights = [
            weights_gpu[f"{prefix}.experts.{i}.gate_proj.weight"] for i in range(N_ROUTED_EXPERTS)
        ]
        expert_up_weights = [
            weights_gpu[f"{prefix}.experts.{i}.up_proj.weight"] for i in range(N_ROUTED_EXPERTS)
        ]
        expert_down_weights = [
            weights_gpu[f"{prefix}.experts.{i}.down_proj.weight"] for i in range(N_ROUTED_EXPERTS)
        ]

        func_routed_out = expert_forward(
            hidden_flat,
            func_topk_idx,
            func_topk_weights,
            expert_gate_weights,
            expert_up_weights,
            expert_down_weights,
        ).view(batch_size, seq_len, hidden_size)

        # Get HF routed output by capturing it
        hf_routed_out = hf_moe.experts(hidden_flat, hf_topk_idx, hf_topk_weights).view(
            batch_size, seq_len, hidden_size
        )

        routed_diff = (hf_routed_out - func_routed_out).abs().max().item()
        print(f"    Routed experts: max_diff={routed_diff:.2e}")

        # Compare final output
        hf_final = hf_routed_out + hf_shared_out
        func_final = func_routed_out + func_shared_out
        final_diff = (hf_final - func_final).abs().max().item()
        print(f"    Final MoE output: max_diff={final_diff:.2e}")


def test_on_gpu(num_layers: int = 5, layer_by_layer: bool = True) -> None:  # noqa: PLR0915
    """Run verification test on GPU."""
    # NOTE: layer_by_layer defaults to True for debugging - remote execution doesn't pass CLI args
    import gc
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import torch
    from glm4_moe_functional import (
        FIRST_K_DENSE_REPLACE,
        glm4_moe_forward,
    )
    from transformers import AutoConfig

    print("=" * 60)
    print(f"GLM-4.5 MoE Functional Implementation Test ({num_layers} layers)")
    print("=" * 60)

    # Determine which shards we need
    shards_needed = num_layers + 1  # +1 for embeddings in shard 1
    print(
        f"\nWill download shards 1-{shards_needed} + shard 93 (~{estimate_download_size(shards_needed)} GB)"
    )

    # Download weights to CPU first
    weights_cpu = download_partial_weights(num_layers)
    print(f"Loaded {len(weights_cpu)} weight tensors to CPU")

    # Test inputs (on CPU for now)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    # === Phase 1: Run HuggingFace model ===
    print("\n### Phase 1: HuggingFace Model ###")
    config = AutoConfig.from_pretrained("zai-org/GLM-4.5", trust_remote_code=True)
    original_num_layers = config.num_hidden_layers
    config.num_hidden_layers = num_layers

    from transformers import Glm4MoeForCausalLM

    model = Glm4MoeForCausalLM(config)
    model.load_state_dict(weights_cpu, strict=False)
    model = model.to(torch.bfloat16).cuda()
    model.eval()

    print(f"HF model ready with {num_layers} layers (original: {original_num_layers})")

    # Capture layer-by-layer outputs from HF if debugging
    hf_intermediates = {}
    if layer_by_layer:
        print("Capturing HF layer outputs...")
        # Capture embedding
        hf_intermediates["embed"] = model.model.embed_tokens(input_ids.cuda()).cpu()

        # Capture each layer output
        hidden = hf_intermediates["embed"].cuda()
        # Need position embeddings for RoPE
        from glm4_moe_functional import compute_rope_embeddings

        positions = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
        cos, sin = compute_rope_embeddings(positions, dtype=hidden.dtype)

        from glm4_moe_functional import FIRST_K_DENSE_REPLACE

        for i in range(num_layers):
            layer_out = model.model.layers[i](hidden, position_embeddings=(cos, sin))
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            hf_intermediates[f"layer_{i}"] = hidden.cpu()
            print(f"  Layer {i}: {hidden.shape}, sample={hidden[0, 0, :3].tolist()}")

            # Capture MoE subcomponent outputs for the first MoE layer (layer 3)
            if i == FIRST_K_DENSE_REPLACE:
                moe = model.model.layers[i].mlp
                hf_layer = model.model.layers[i]

                # Use hooks to capture MoE input, shared expert output, and routed expert output
                moe_captures = {}

                def make_hook(name):
                    def hook(module, inp, out):
                        if isinstance(inp, tuple) and len(inp) > 0:
                            moe_captures[name + "_in"] = inp[0].clone() if hasattr(inp[0], 'clone') else inp[0]
                        elif hasattr(inp, 'clone'):
                            moe_captures[name + "_in"] = inp.clone()
                        if isinstance(out, tuple):
                            # gate returns (topk_indices, topk_weights)
                            moe_captures[name + "_out"] = tuple(x.clone() if hasattr(x, 'clone') else x for x in out)
                        elif hasattr(out, 'clone'):
                            moe_captures[name + "_out"] = out.clone()
                    return hook

                hooks = []
                hooks.append(hf_layer.mlp.register_forward_hook(make_hook("moe")))
                hooks.append(hf_layer.mlp.shared_experts.register_forward_hook(make_hook("shared")))
                hooks.append(hf_layer.mlp.experts.register_forward_hook(make_hook("experts")))
                hooks.append(hf_layer.mlp.gate.register_forward_hook(make_hook("gate")))

                # Re-run the layer forward to capture via hooks
                layer_i_input = hf_intermediates[f"layer_{i-1}"].cuda() if i > 0 else hf_intermediates["embed"].cuda()
                with torch.no_grad():
                    _ = hf_layer(layer_i_input, position_embeddings=(cos, sin))

                for h in hooks:
                    h.remove()

                if "moe_in" in moe_captures:
                    hf_intermediates[f"moe_{i}_input"] = moe_captures["moe_in"].cpu()
                if "gate_out" in moe_captures:
                    gate_out = moe_captures["gate_out"]
                    if isinstance(gate_out, tuple):
                        # gate returns (topk_indices, topk_weights)
                        hf_intermediates[f"moe_{i}_topk_idx"] = gate_out[0].cpu()
                        hf_intermediates[f"moe_{i}_topk_weights"] = gate_out[1].cpu()
                    else:
                        hf_intermediates[f"moe_{i}_router_logits"] = gate_out.cpu()
                if "shared_out" in moe_captures:
                    hf_intermediates[f"moe_{i}_shared_out"] = moe_captures["shared_out"].cpu()
                if "experts_out" in moe_captures:
                    hf_intermediates[f"moe_{i}_routed_out"] = moe_captures["experts_out"].cpu()

                print(f"    Captured MoE subcomponents for layer {i}: {list(moe_captures.keys())}")

        # Capture final norm
        hf_intermediates["norm"] = model.model.norm(hidden).cpu()

    with torch.no_grad():
        hf_logits = model(input_ids.cuda()).logits.cpu()  # Save to CPU

    print(f"HF logits shape: {hf_logits.shape}")

    # Free GPU memory from HF model before loading weights
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("HF model deleted, GPU memory freed")

    # Move weights to GPU for functional forward
    weights_gpu = {k: v.cuda() for k, v in weights_cpu.items()}
    del weights_cpu
    gc.collect()

    # === Phase 2: Run functional implementation ===
    print("\n### Phase 2: Functional Implementation ###")

    # Layer-by-layer comparison if debugging
    if layer_by_layer and hf_intermediates:
        import torch.nn.functional as F
        from glm4_moe_functional import (
            compute_rope_embeddings,
            transformer_layer,
        )

        print("\nComparing layer-by-layer...")

        # Compare embeddings
        func_embed = F.embedding(input_ids.cuda(), weights_gpu["model.embed_tokens.weight"])
        embed_diff = (hf_intermediates["embed"].cuda() - func_embed).abs().max().item()
        print(
            f"  Embeddings: max_diff={embed_diff:.2e} [{'PASS' if embed_diff < 1e-4 else 'FAIL'}]"
        )

        if embed_diff >= 1e-4:
            print("    STOPPING: Embeddings diverged")
        else:
            # Compare each layer
            hidden = func_embed
            positions = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
            cos, sin = compute_rope_embeddings(positions, dtype=hidden.dtype)

            for i in range(num_layers):
                # Get input hidden state for this layer (before the layer)
                layer_input = hidden.clone()

                hidden = transformer_layer(hidden, weights_gpu, i, cos, sin)
                hf_hidden = hf_intermediates[f"layer_{i}"].cuda()
                layer_diff = (hf_hidden - hidden).abs().max().item()
                layer_type = "dense" if i < FIRST_K_DENSE_REPLACE else "MoE"
                status = "PASS" if layer_diff < 1e-4 else "FAIL"
                print(f"  Layer {i} ({layer_type}): max_diff={layer_diff:.2e} [{status}]")

                if layer_diff >= 1e-4:
                    print(f"    STOPPING: Layer {i} diverged")
                    print(f"    HF sample: {hf_hidden[0, 0, :5].tolist()}")
                    print(f"    Func sample: {hidden[0, 0, :5].tolist()}")

                    # Debug MoE subcomponents using captured HF outputs
                    if i >= FIRST_K_DENSE_REPLACE and f"moe_{i}_input" in hf_intermediates:
                        print(f"\n    ### Debugging MoE Layer {i} ###")
                        from glm4_moe_functional import (
                            moe_router,
                            expert_forward,
                            dense_mlp,
                            N_ROUTED_EXPERTS,
                        )

                        prefix = f"model.layers.{i}.mlp"

                        # Compare MoE input (should match since it's from post_attention_layernorm)
                        hf_moe_input = hf_intermediates[f"moe_{i}_input"].cuda()

                        # Compute functional routing
                        bias_key = f"{prefix}.gate.e_score_correction_bias"
                        print(f"      e_score_correction_bias shape: {weights_gpu[bias_key].shape}")
                        print(f"      e_score_correction_bias sample: {weights_gpu[bias_key][:5].tolist()}")
                        func_topk_idx, func_topk_weights = moe_router(
                            hf_moe_input.view(-1, hf_moe_input.shape[-1]),
                            router_weight=weights_gpu[f"{prefix}.gate.weight"],
                            e_score_correction_bias=weights_gpu[bias_key],
                            debug=True,
                        )

                        # Compare topk indices if captured
                        if f"moe_{i}_topk_idx" in hf_intermediates:
                            hf_topk_idx = hf_intermediates[f"moe_{i}_topk_idx"].cuda()
                            hf_topk_weights = hf_intermediates[f"moe_{i}_topk_weights"].cuda()

                            # Check if same experts are selected
                            hf_sorted = hf_topk_idx.sort(dim=-1)[0]
                            func_sorted = func_topk_idx.sort(dim=-1)[0]
                            same_experts = (hf_sorted == func_sorted).all().item()
                            print(f"      Same experts selected: {same_experts}")
                            if not same_experts:
                                print(f"        HF experts: {hf_topk_idx[0].tolist()}")
                                print(f"        Func experts: {func_topk_idx[0].tolist()}")

                            # Compare routing weights
                            weight_diff = (hf_topk_weights - func_topk_weights).abs().max().item()
                            print(f"      Router weights diff: max_diff={weight_diff:.2e}")
                            print(f"        HF weights sample: {hf_topk_weights[0,:3].tolist()}")
                            print(f"        Func weights sample: {func_topk_weights[0,:3].tolist()}")
                        else:
                            # Compare router logits if captured (older style)
                            if f"moe_{i}_router_logits" in hf_intermediates:
                                hf_router_logits = hf_intermediates[f"moe_{i}_router_logits"].cuda()
                                func_router_logits = F.linear(
                                    hf_moe_input.view(-1, hf_moe_input.shape[-1]).float(),
                                    weights_gpu[f"{prefix}.gate.weight"].float()
                                )
                                router_diff = (hf_router_logits.view(-1, N_ROUTED_EXPERTS) - func_router_logits).abs().max().item()
                                print(f"      Router logits: max_diff={router_diff:.2e}")
                            print(f"      Func routing: experts={func_topk_idx[0].tolist()}, weights={func_topk_weights[0,:3].tolist()}")

                        # Compare shared expert
                        if f"moe_{i}_shared_out" in hf_intermediates:
                            hf_shared = hf_intermediates[f"moe_{i}_shared_out"].cuda()
                            func_shared = dense_mlp(
                                hf_moe_input,
                                gate_weight=weights_gpu[f"{prefix}.shared_experts.gate_proj.weight"],
                                up_weight=weights_gpu[f"{prefix}.shared_experts.up_proj.weight"],
                                down_weight=weights_gpu[f"{prefix}.shared_experts.down_proj.weight"],
                            )
                            shared_diff = (hf_shared - func_shared).abs().max().item()
                            print(f"      Shared expert: max_diff={shared_diff:.2e}")
                        else:
                            func_shared = dense_mlp(
                                hf_moe_input,
                                gate_weight=weights_gpu[f"{prefix}.shared_experts.gate_proj.weight"],
                                up_weight=weights_gpu[f"{prefix}.shared_experts.up_proj.weight"],
                                down_weight=weights_gpu[f"{prefix}.shared_experts.down_proj.weight"],
                            )

                        # Compare routed experts
                        if f"moe_{i}_routed_out" in hf_intermediates:
                            hf_routed = hf_intermediates[f"moe_{i}_routed_out"].cuda()
                            expert_gate_weights = [weights_gpu[f"{prefix}.experts.{j}.gate_proj.weight"] for j in range(N_ROUTED_EXPERTS)]
                            expert_up_weights = [weights_gpu[f"{prefix}.experts.{j}.up_proj.weight"] for j in range(N_ROUTED_EXPERTS)]
                            expert_down_weights = [weights_gpu[f"{prefix}.experts.{j}.down_proj.weight"] for j in range(N_ROUTED_EXPERTS)]

                            func_routed = expert_forward(
                                hf_moe_input.view(-1, hf_moe_input.shape[-1]),
                                func_topk_idx,
                                func_topk_weights,
                                expert_gate_weights,
                                expert_up_weights,
                                expert_down_weights,
                            ).view_as(hf_routed)
                            routed_diff = (hf_routed - func_routed).abs().max().item()
                            print(f"      Routed experts: max_diff={routed_diff:.2e}")

                            # Compare final MoE output
                            hf_moe_out = hf_routed + hf_shared
                            func_moe_out = func_routed + func_shared
                            moe_diff = (hf_moe_out - func_moe_out).abs().max().item()
                            print(f"      Final MoE output: max_diff={moe_diff:.2e}")

                    break

    with torch.no_grad():
        func_logits = glm4_moe_forward(input_ids.cuda(), weights_gpu, num_layers=num_layers).cpu()

    print(f"Func logits shape: {func_logits.shape}")

    # === Phase 3: Compare ===
    print("\n### Results ###")
    matches = torch.allclose(hf_logits, func_logits, rtol=1e-4, atol=1e-4)
    max_diff = (hf_logits - func_logits).abs().max().item()

    status = "PASS" if matches else "FAIL"
    print(f"  Full forward: max_diff={max_diff:.2e} [{status}]")

    if not matches:
        print(f"\n  HF logits sample: {hf_logits[0, 0, :5]}")
        print(f"  Func logits sample: {func_logits[0, 0, :5]}")
        sys.exit(1)
    else:
        print("\n  SUCCESS: Functional implementation matches HuggingFace!")


def estimate_download_size(shards: int) -> float:
    """Estimate download size in GB."""
    # Shard 1: 3.75 GB, Shards 2-3: 0.65 GB each, Shards 4+: 7.87 GB each
    size = 3.75
    if shards >= 2:
        size += 0.65
    if shards >= 3:
        size += 0.65
    if shards >= 4:
        size += (shards - 3) * 7.87
    return round(size, 1)


def test_full_forward(
    model: AutoModelForCausalLM,
    weights: dict,
    input_ids: torch.Tensor,
    num_layers: int,
) -> None:
    """Test full forward pass."""
    import torch
    from glm4_moe_functional import glm4_moe_forward

    print("\n### Full Forward Pass Test ###")

    with torch.no_grad():
        # HuggingFace forward
        hf_output = model(input_ids)
        hf_logits = hf_output.logits

        # Functional forward
        func_logits = glm4_moe_forward(input_ids, weights, num_layers=num_layers)

    # Compare
    matches = torch.allclose(hf_logits, func_logits, rtol=1e-4, atol=1e-4)
    max_diff = (hf_logits - func_logits).abs().max().item()

    status = "PASS" if matches else "FAIL"
    print(f"  Full forward: max_diff={max_diff:.2e} [{status}]")

    if not matches:
        print(f"\n  HF logits shape: {hf_logits.shape}")
        print(f"  Func logits shape: {func_logits.shape}")
        print(f"  HF logits sample: {hf_logits[0, 0, :5]}")
        print(f"  Func logits sample: {func_logits[0, 0, :5]}")

        # Find where the max diff is
        diff = (hf_logits - func_logits).abs()
        max_idx = diff.argmax()
        batch_idx = max_idx // (diff.shape[1] * diff.shape[2])
        seq_idx = (max_idx % (diff.shape[1] * diff.shape[2])) // diff.shape[2]
        vocab_idx = max_idx % diff.shape[2]
        print(f"  Max diff at: batch={batch_idx}, seq={seq_idx}, vocab={vocab_idx}")

        sys.exit(1)


def test_layer_by_layer(
    model: AutoModelForCausalLM,
    weights: dict,
    input_ids: torch.Tensor,
    num_layers: int,
) -> None:
    """Test each layer individually to find divergence point."""
    import torch
    import torch.nn.functional as F
    from glm4_moe_functional import (
        FIRST_K_DENSE_REPLACE,
        compute_rope_embeddings,
        rms_norm,
        transformer_layer,
    )

    print("\n### Layer-by-Layer Test ###")

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Test embeddings
    with torch.no_grad():
        hf_embed = model.model.embed_tokens(input_ids)
        func_embed = F.embedding(input_ids, weights["model.embed_tokens.weight"])

    embed_diff = (hf_embed - func_embed).abs().max().item()
    embed_match = embed_diff < 1e-5
    print(f"  Embeddings: max_diff={embed_diff:.2e} [{'PASS' if embed_match else 'FAIL'}]")

    if not embed_match:
        print("  Stopping at embeddings divergence")
        return

    # Compute RoPE
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    cos, sin = compute_rope_embeddings(positions, dtype=hf_embed.dtype)

    # Test each layer
    hf_hidden = hf_embed
    func_hidden = func_embed.clone()

    for layer_idx in range(num_layers):
        layer_type = "dense" if layer_idx < FIRST_K_DENSE_REPLACE else "MoE"

        with torch.no_grad():
            # HF layer forward
            hf_layer = model.model.layers[layer_idx]
            hf_out = hf_layer(hf_hidden, position_embeddings=(cos, sin))
            if isinstance(hf_out, tuple):
                hf_out = hf_out[0]

            # Functional layer forward
            func_out = transformer_layer(func_hidden, weights, layer_idx, cos, sin)

        layer_diff = (hf_out - func_out).abs().max().item()
        layer_match = layer_diff < 1e-4

        status = "PASS" if layer_match else "FAIL"
        print(f"  Layer {layer_idx:2d} ({layer_type:5s}): max_diff={layer_diff:.2e} [{status}]")

        if not layer_match:
            print(f"\n  ### DIVERGENCE AT LAYER {layer_idx} ({layer_type}) ###")
            debug_layer(model, weights, layer_idx, hf_hidden, cos, sin)
            return

        # Update hidden states for next layer
        hf_hidden = hf_out
        func_hidden = func_out

    # Test final norm
    with torch.no_grad():
        hf_normed = model.model.norm(hf_hidden)
        func_normed = rms_norm(func_hidden, weights["model.norm.weight"])

    norm_diff = (hf_normed - func_normed).abs().max().item()
    norm_match = norm_diff < 1e-4
    print(f"  Final norm: max_diff={norm_diff:.2e} [{'PASS' if norm_match else 'FAIL'}]")

    # Test LM head
    with torch.no_grad():
        hf_logits = model.lm_head(hf_normed)
        func_logits = F.linear(func_normed, weights["model.embed_tokens.weight"])

    logits_diff = (hf_logits - func_logits).abs().max().item()
    logits_match = logits_diff < 1e-4
    print(f"  LM head: max_diff={logits_diff:.2e} [{'PASS' if logits_match else 'FAIL'}]")

    if embed_match and norm_match and logits_match:
        print(f"\n  All {num_layers} layers PASSED!")


def debug_layer(
    model: AutoModelForCausalLM,
    weights: dict,
    layer_idx: int,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    """Debug a specific layer to find component divergence."""
    import torch
    from glm4_moe_functional import (
        FIRST_K_DENSE_REPLACE,
        attention,
        dense_mlp,
        moe_block,
        rms_norm,
    )

    prefix = f"model.layers.{layer_idx}"
    hf_layer = model.model.layers[layer_idx]

    print("\n  Debugging layer components...")

    with torch.no_grad():
        # Input layernorm
        hf_normed = hf_layer.input_layernorm(hidden_states)
        func_normed = rms_norm(hidden_states, weights[f"{prefix}.input_layernorm.weight"])
        norm_diff = (hf_normed - func_normed).abs().max().item()
        print(f"    input_layernorm: max_diff={norm_diff:.2e}")

        # Attention
        hf_attn_out = None

        def capture_attn(module, input, output):
            nonlocal hf_attn_out
            hf_attn_out = output[0] if isinstance(output, tuple) else output

        hook = hf_layer.self_attn.register_forward_hook(capture_attn)
        _ = hf_layer(hidden_states, position_embeddings=(cos, sin))
        hook.remove()

        func_attn_out = attention(
            func_normed,
            q_weight=weights[f"{prefix}.self_attn.q_proj.weight"],
            q_bias=weights[f"{prefix}.self_attn.q_proj.bias"],
            k_weight=weights[f"{prefix}.self_attn.k_proj.weight"],
            k_bias=weights[f"{prefix}.self_attn.k_proj.bias"],
            v_weight=weights[f"{prefix}.self_attn.v_proj.weight"],
            v_bias=weights[f"{prefix}.self_attn.v_proj.bias"],
            o_weight=weights[f"{prefix}.self_attn.o_proj.weight"],
            q_norm_weight=weights[f"{prefix}.self_attn.q_norm.weight"],
            k_norm_weight=weights[f"{prefix}.self_attn.k_norm.weight"],
            cos=cos,
            sin=sin,
        )

        if hf_attn_out is not None:
            attn_diff = (hf_attn_out - func_attn_out).abs().max().item()
            print(f"    attention: max_diff={attn_diff:.2e}")

        # Post-attention residual
        hf_post_attn = hidden_states + hf_attn_out
        func_post_attn = hidden_states + func_attn_out

        # Post-attention layernorm
        hf_post_normed = hf_layer.post_attention_layernorm(hf_post_attn)
        func_post_normed = rms_norm(
            func_post_attn, weights[f"{prefix}.post_attention_layernorm.weight"]
        )
        post_norm_diff = (hf_post_normed - func_post_normed).abs().max().item()
        print(f"    post_attention_layernorm: max_diff={post_norm_diff:.2e}")

        # MLP/MoE
        if layer_idx < FIRST_K_DENSE_REPLACE:
            func_mlp_out = dense_mlp(
                func_post_normed,
                gate_weight=weights[f"{prefix}.mlp.gate_proj.weight"],
                up_weight=weights[f"{prefix}.mlp.up_proj.weight"],
                down_weight=weights[f"{prefix}.mlp.down_proj.weight"],
            )
            print("    MLP type: dense")
        else:
            func_mlp_out = moe_block(func_post_normed, weights, layer_idx)
            print("    MLP type: MoE")

        # Capture HF MLP output
        hf_mlp_out = None

        def capture_mlp(module, input, output):
            nonlocal hf_mlp_out
            hf_mlp_out = output

        hook = hf_layer.mlp.register_forward_hook(capture_mlp)
        _ = hf_layer(hidden_states, position_embeddings=(cos, sin))
        hook.remove()

        if hf_mlp_out is not None:
            mlp_diff = (hf_mlp_out - func_mlp_out).abs().max().item()
            print(f"    mlp: max_diff={mlp_diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test functional GLM-4.5 MoE implementation")
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    parser.add_argument("--gpu-id", type=str, help="Reuse existing GPU instance")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU alive after test")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of layers to test")
    parser.add_argument(
        "--layer-by-layer", action="store_true", default=True, help="Test each layer individually"
    )
    args = parser.parse_args()

    if args.remote:
        from tools.functional_extractor.config import DeploymentConfig
        from tools.functional_extractor.verify import run_on_gpu

        # GLM-4.5 needs more VRAM and disk - estimate based on layers
        vram_needed = 24 if args.num_layers <= 3 else 48 if args.num_layers <= 5 else 80
        # Model weights + HF cache + PyTorch needs ~50GB minimum
        disk_needed = 100

        run_on_gpu(
            script_path=__file__,
            deployment=DeploymentConfig(vram_gb=vram_needed, container_disk=disk_needed),
            gpu_id=args.gpu_id,
            keep_alive=args.keep_alive or bool(args.gpu_id),
        )
    else:
        import torch

        if not torch.cuda.is_available():
            print("No GPU available locally. Use --remote to run on remote GPU.")
            sys.exit(1)

        test_on_gpu(num_layers=args.num_layers, layer_by_layer=args.layer_by_layer)


if __name__ == "__main__":
    main()
