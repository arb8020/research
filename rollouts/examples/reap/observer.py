"""MoE activation observer using PyTorch hooks.

Collects per-expert statistics during forward passes for REAP scoring.

Architecture-specific hook strategies:

  Qwen3 (transformers 5.x):
    Hooks Qwen3MoeExperts — the fused expert module whose args include
    (hidden_states, top_k_index, top_k_weights) from the router. EAN is
    computed directly from the fused gate_up_proj/down_proj weight tensors,
    matching the per-expert computation in Qwen3MoeExperts.forward.

  Mixtral / ModuleList-based:
    Hooks the sparse MoE block; iterates experts as a ModuleList.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class MoELayerConfig:
    """Attributes needed to find MoE components within a decoder layer."""

    moe_block_attr: str  # attr on decoder layer, e.g. "mlp"
    experts_attr: str  # attr on moe_block holding experts, e.g. "experts"
    router_attr: str  # attr on moe_block holding the router, e.g. "gate"
    num_experts: int
    num_experts_per_tok: int
    fused: bool  # True if experts are fused (Qwen3MoeExperts-style)


# Registry: model class name -> config extractor
_MOE_CONFIG_REGISTRY: dict[str, Any] = {}


def register_moe_config(model_class: str) -> Any:
    def decorator(fn: Any) -> Any:
        _MOE_CONFIG_REGISTRY[model_class] = fn
        return fn

    return decorator


@register_moe_config("Qwen3MoeForCausalLM")
def _qwen3_moe_config(model: nn.Module) -> MoELayerConfig:
    config = model.config
    return MoELayerConfig(
        moe_block_attr="mlp",
        experts_attr="experts",
        router_attr="gate",
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        fused=True,
    )


@register_moe_config("MixtralForCausalLM")
def _mixtral_moe_config(model: nn.Module) -> MoELayerConfig:
    config = model.config
    return MoELayerConfig(
        moe_block_attr="block_sparse_moe",
        experts_attr="experts",
        router_attr="gate",
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        fused=False,
    )


@register_moe_config("DeepseekV2ForCausalLM")
def _deepseek_moe_config(model: nn.Module) -> MoELayerConfig:
    config = model.config
    return MoELayerConfig(
        moe_block_attr="mlp",
        experts_attr="experts",
        router_attr="gate",
        num_experts=config.n_routed_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        fused=False,
    )


@dataclass
class LayerObservation:
    """Collected statistics for one MoE layer."""

    expert_frequency: Tensor  # [num_experts]
    ean_sum: Tensor  # [num_experts] sum of (EAN * routing_weight) per routed token
    routing_weight_sum: Tensor  # [num_experts] sum of routing weights
    max_activations: Tensor  # [num_experts] peak activation value per expert


class MoEObserver:
    """Observes MoE activations during forward passes.

    For fused expert architectures (Qwen3): hooks the fused Experts module,
    reads routing info from args, computes EAN from weight tensors.

    For ModuleList architectures (Mixtral): hooks the MoE block, iterates
    individual expert modules.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.hooks: list[Any] = []
        self.observations: dict[int, LayerObservation] = {}

        model_class = model.__class__.__name__
        assert model_class in _MOE_CONFIG_REGISTRY, (
            f"Unsupported model architecture: {model_class}. "
            f"Supported: {list(_MOE_CONFIG_REGISTRY.keys())}"
        )
        self.moe_config = _MOE_CONFIG_REGISTRY[model_class](model)
        self._setup_hooks()

    def _get_moe_layers(self) -> list[tuple[int, nn.Module]]:
        """Return (layer_idx, moe_block) for each MoE layer."""
        base = self.model.model if hasattr(self.model, "model") else self.model
        assert hasattr(base, "layers"), "Could not find decoder layers in model"

        layers = []
        for idx, layer in enumerate(base.layers):
            moe_block = getattr(layer, self.moe_config.moe_block_attr, None)
            if moe_block is not None and hasattr(moe_block, self.moe_config.experts_attr):
                layers.append((idx, moe_block))
        return layers

    def _setup_hooks(self) -> None:
        moe_layers = self._get_moe_layers()
        logger.info(f"Found {len(moe_layers)} MoE layers")
        assert len(moe_layers) > 0, (
            f"No MoE layers found using moe_block_attr='{self.moe_config.moe_block_attr}'"
        )

        for layer_idx, moe_block in moe_layers:
            num_experts = self.moe_config.num_experts
            self.observations[layer_idx] = LayerObservation(
                expert_frequency=torch.zeros(num_experts, device="cpu"),
                ean_sum=torch.zeros(num_experts, device="cpu", dtype=torch.float64),
                routing_weight_sum=torch.zeros(num_experts, device="cpu", dtype=torch.float64),
                max_activations=torch.zeros(num_experts, device="cpu"),
            )

            experts_module = getattr(moe_block, self.moe_config.experts_attr)
            if self.moe_config.fused:
                # Hook the fused Experts module — args carry routing decisions
                hook = experts_module.register_forward_pre_hook(
                    self._make_fused_hook(layer_idx), with_kwargs=False
                )
            else:
                # Hook the MoE block — iterate ModuleList experts
                hook = moe_block.register_forward_hook(self._make_block_hook(layer_idx))
            self.hooks.append(hook)

    # ── Fused expert hook (Qwen3) ────────────────────────────────────────────

    def _make_fused_hook(self, layer_idx: int) -> Any:
        def hook(module: nn.Module, args: tuple) -> None:
            # Qwen3MoeExperts.forward(hidden_states, top_k_index, top_k_weights)
            assert len(args) == 3, (
                f"Expected 3 args for fused experts hook, got {len(args)}: "
                f"{[type(a) for a in args]}"
            )
            hidden_states, top_k_index, top_k_weights = args
            # top_k_index: [num_tokens, top_k]  top_k_weights: [num_tokens, top_k]

            with torch.no_grad():
                self._process_fused(layer_idx, module, hidden_states, top_k_index, top_k_weights)

        return hook

    def _process_fused(
        self,
        layer_idx: int,
        module: nn.Module,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> None:
        """Compute EAN from fused weight tensors for each active expert."""
        obs = self.observations[layer_idx]
        num_experts = self.moe_config.num_experts

        # Flatten to [num_tokens, hidden_dim]
        hidden_dim = hidden_states.shape[-1]
        flat_hidden = hidden_states.view(-1, hidden_dim)
        flat_index = top_k_index.view(-1, top_k_index.shape[-1])  # [num_tokens, top_k]
        flat_weights = top_k_weights.view(-1, top_k_weights.shape[-1])  # [num_tokens, top_k]

        expert_frequency = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        ean_sum = torch.zeros(num_experts, device="cpu", dtype=torch.float64)
        routing_weight_sum = torch.zeros(num_experts, device="cpu", dtype=torch.float64)
        max_activations = obs.max_activations.clone()

        # Replicate Qwen3MoeExperts.forward per-expert computation
        for expert_idx in range(num_experts):
            # Find tokens routed to this expert and which top-k slot
            top_k_pos, token_idx = torch.where(flat_index == expert_idx)
            if token_idx.numel() == 0:
                continue

            current_state = flat_hidden[token_idx]  # [num_routed, hidden_dim]
            routing_w = flat_weights[token_idx, top_k_pos]  # [num_routed]

            # Expert forward: gate_up -> silu gate -> down
            gate_up = F.linear(current_state, module.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = module.act_fn(gate) * up  # [num_routed, intermediate_dim]

            ean_norms = expert_out.norm(dim=-1)  # [num_routed]

            expert_frequency[expert_idx] = token_idx.numel()
            ean_sum[expert_idx] = (ean_norms * routing_w).sum().to(dtype=torch.float64).cpu()
            routing_weight_sum[expert_idx] = routing_w.sum().to(dtype=torch.float64).cpu()
            max_act = expert_out.max().cpu()
            if max_act > max_activations[expert_idx]:
                max_activations[expert_idx] = max_act

        obs.expert_frequency += expert_frequency
        obs.ean_sum += ean_sum
        obs.routing_weight_sum += routing_weight_sum
        obs.max_activations = max_activations

    # ── ModuleList block hook (Mixtral) ──────────────────────────────────────

    def _make_block_hook(self, layer_idx: int) -> Any:
        def hook(module: nn.Module, args: tuple, output: Any) -> None:
            # output last element should be router_logits for non-fused architectures
            assert isinstance(output, tuple) and len(output) >= 2, (
                f"Expected tuple output from MoE block, got {type(output)}"
            )
            hidden_states = args[0]
            router_logits = output[-1]
            assert isinstance(router_logits, Tensor)

            with torch.no_grad():
                self._process_block(layer_idx, module, hidden_states, router_logits)

        return hook

    def _process_block(
        self,
        layer_idx: int,
        module: nn.Module,
        hidden_states: Tensor,
        router_logits: Tensor,
    ) -> None:
        obs = self.observations[layer_idx]
        num_experts = self.moe_config.num_experts
        top_k = self.moe_config.num_experts_per_tok
        device = hidden_states.device

        hidden_dim = hidden_states.shape[-1]
        flat_hidden = hidden_states.view(-1, hidden_dim)
        flat_logits = router_logits.view(-1, num_experts)

        routing_weights = F.softmax(flat_logits, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        experts = getattr(module, self.moe_config.experts_attr)
        activations = torch.zeros((num_experts, flat_hidden.shape[0], hidden_dim), device=device)
        for i, expert in enumerate(experts):
            activations[i] = expert(flat_hidden)

        expert_frequency = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        ean_sum = torch.zeros(num_experts, device="cpu", dtype=torch.float64)
        routing_weight_sum = torch.zeros(num_experts, device="cpu", dtype=torch.float64)
        max_activations = obs.max_activations.clone()

        for i in range(num_experts):
            active_mask = (selected_experts == i).any(dim=-1)
            if not active_mask.any():
                continue

            active_routing_weights = routing_weights[active_mask, i]
            ean_norms = activations[i, active_mask].norm(dim=-1)

            expert_frequency[i] = active_mask.sum()
            ean_sum[i] = (ean_norms * active_routing_weights).sum().to(dtype=torch.float64)
            routing_weight_sum[i] = active_routing_weights.sum().to(dtype=torch.float64)
            max_act = activations[i, active_mask].max().cpu()
            if max_act > max_activations[i]:
                max_activations[i] = max_act

        obs.expert_frequency += expert_frequency
        obs.ean_sum += ean_sum
        obs.routing_weight_sum += routing_weight_sum
        obs.max_activations = max_activations

    # ── Shared ───────────────────────────────────────────────────────────────

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_observations(self) -> dict[int, LayerObservation]:
        return self.observations

    def save_observations(self, path: str) -> None:
        data = {
            layer_idx: {
                "expert_frequency": obs.expert_frequency.cpu(),
                "ean_sum": obs.ean_sum.cpu(),
                "routing_weight_sum": obs.routing_weight_sum.cpu(),
                "max_activations": obs.max_activations.cpu(),
            }
            for layer_idx, obs in self.observations.items()
        }
        torch.save(data, path)
        logger.info(f"Saved observations to {path}")

    def load_observations(self, path: str) -> None:
        data = torch.load(path, weights_only=True)
        for layer_idx, obs_data in data.items():
            self.observations[int(layer_idx)] = LayerObservation(
                expert_frequency=obs_data["expert_frequency"],
                ean_sum=obs_data["ean_sum"],
                routing_weight_sum=obs_data["routing_weight_sum"],
                max_activations=obs_data["max_activations"],
            )
        logger.info(f"Loaded observations from {path}")
