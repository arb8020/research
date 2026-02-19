"""MoE activation observer using PyTorch hooks.

Collects per-expert statistics during forward passes for REAP scoring.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class MoELayerConfig:
    """Configuration for accessing MoE layer components.

    Different model architectures organize their MoE layers differently.
    This config maps attribute names for a specific architecture.
    """

    moe_block_attr: str  # e.g., "mlp" for Qwen3
    experts_attr: str  # e.g., "experts"
    router_attr: str  # e.g., "gate"
    num_experts: int
    num_experts_per_tok: int


# Registry mapping model class names to their MoE config
MOE_CONFIG_REGISTRY: dict[str, Callable[[Any], MoELayerConfig]] = {}


def register_moe_config(model_class: str) -> Callable:
    """Decorator to register MoE config extractor for a model class."""

    def decorator(fn: Callable[[Any], MoELayerConfig]) -> Callable:
        MOE_CONFIG_REGISTRY[model_class] = fn
        return fn

    return decorator


@register_moe_config("Qwen3MoeForCausalLM")
def qwen3_moe_config(model: nn.Module) -> MoELayerConfig:
    """Extract MoE config from Qwen3 model."""
    config = model.config
    return MoELayerConfig(
        moe_block_attr="mlp",
        experts_attr="experts",
        router_attr="gate",
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
    )


@register_moe_config("MixtralForCausalLM")
def mixtral_moe_config(model: nn.Module) -> MoELayerConfig:
    """Extract MoE config from Mixtral model."""
    config = model.config
    return MoELayerConfig(
        moe_block_attr="block_sparse_moe",
        experts_attr="experts",
        router_attr="gate",
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
    )


@register_moe_config("DeepseekV2ForCausalLM")
def deepseek_moe_config(model: nn.Module) -> MoELayerConfig:
    """Extract MoE config from DeepSeek-V2 model."""
    config = model.config
    return MoELayerConfig(
        moe_block_attr="mlp",
        experts_attr="experts",
        router_attr="gate",
        num_experts=config.n_routed_experts,
        num_experts_per_tok=config.num_experts_per_tok,
    )


@dataclass
class LayerObservation:
    """Collected statistics for one MoE layer."""

    expert_frequency: Tensor  # [num_experts] count of routed tokens
    ean_sum: Tensor  # [num_experts] sum of activation norms
    routing_weight_sum: Tensor  # [num_experts] sum of routing weights
    max_activations: Tensor  # [num_experts] peak activation per expert


class MoEObserver:
    """Observes MoE activations during forward passes.

    Attaches hooks to MoE layers to collect:
    - Expert routing frequency
    - Expert activation norms (EAN)
    - Router weights
    - Max activations (for super-expert detection)
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.hooks: list[Any] = []
        self.observations: dict[int, LayerObservation] = {}

        # Get MoE config for this model architecture
        model_class = model.__class__.__name__
        if model_class not in MOE_CONFIG_REGISTRY:
            raise ValueError(
                f"Unsupported model architecture: {model_class}. "
                f"Supported: {list(MOE_CONFIG_REGISTRY.keys())}"
            )

        self.moe_config = MOE_CONFIG_REGISTRY[model_class](model)
        self._setup_hooks()

    def _get_moe_layers(self) -> list[tuple[int, nn.Module]]:
        """Get all MoE layers from the model."""
        layers = []

        # Navigate to decoder layers
        if hasattr(self.model, "model"):
            base = self.model.model
        else:
            base = self.model

        if hasattr(base, "layers"):
            decoder_layers = base.layers
        elif hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
            decoder_layers = base.decoder.layers
        else:
            raise ValueError("Could not find decoder layers in model")

        for idx, layer in enumerate(decoder_layers):
            moe_block = getattr(layer, self.moe_config.moe_block_attr, None)
            if moe_block is not None and hasattr(moe_block, self.moe_config.experts_attr):
                layers.append((idx, moe_block))

        return layers

    def _setup_hooks(self) -> None:
        """Register forward hooks on all MoE layers."""
        moe_layers = self._get_moe_layers()
        logger.info(f"Found {len(moe_layers)} MoE layers")

        for layer_idx, moe_block in moe_layers:
            # Initialize observation storage for this layer
            self.observations[layer_idx] = LayerObservation(
                expert_frequency=torch.zeros(self.moe_config.num_experts, device=self.device),
                ean_sum=torch.zeros(self.moe_config.num_experts, device=self.device),
                routing_weight_sum=torch.zeros(self.moe_config.num_experts, device=self.device),
                max_activations=torch.zeros(self.moe_config.num_experts, device=self.device),
            )

            # Get the router module
            router = getattr(moe_block, self.moe_config.router_attr)

            # Hook the router to capture routing decisions
            hook = router.register_forward_hook(self._make_router_hook(layer_idx, moe_block))
            self.hooks.append(hook)

    def _make_router_hook(self, layer_idx: int, moe_block: nn.Module) -> Callable:
        """Create a hook function for a specific layer."""

        def hook(
            module: nn.Module,
            inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> None:
            # output is router logits: [num_tokens, num_experts] (already flattened in some models)
            router_logits = output
            hidden_states = inputs[0]  # [batch, seq, hidden_dim] or [num_tokens, hidden_dim]

            with torch.no_grad():
                self._process_routing(layer_idx, moe_block, hidden_states, router_logits)

        return hook

    def _process_routing(
        self,
        layer_idx: int,
        moe_block: nn.Module,
        hidden_states: Tensor,
        router_logits: Tensor,
    ) -> None:
        """Process routing decisions and update observations."""
        obs = self.observations[layer_idx]
        num_experts = self.moe_config.num_experts
        top_k = self.moe_config.num_experts_per_tok

        # hidden_states may be [batch, seq, hidden] or [num_tokens, hidden] depending on model
        assert hidden_states.ndim in (2, 3), (
            f"Unexpected hidden_states shape: {hidden_states.shape}"
        )
        hidden_dim = hidden_states.shape[-1]
        hidden_flat = hidden_states.view(-1, hidden_dim)  # [num_tokens, hidden]
        logits_flat = router_logits.view(-1, num_experts)  # [num_tokens, num_experts]

        # Get routing weights (softmax of logits)
        routing_weights = torch.softmax(logits_flat, dim=-1)

        # Get top-k experts per token
        top_weights, top_indices = torch.topk(routing_weights, top_k, dim=-1)  # [batch*seq, top_k]

        # Re-normalize top-k weights
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Get expert modules
        experts = getattr(moe_block, self.moe_config.experts_attr)

        # For each expert, compute activation norms for tokens routed to it
        for expert_idx in range(num_experts):
            # Mask for tokens routed to this expert
            # top_indices: [num_tokens, top_k]
            routed_mask = (top_indices == expert_idx).any(dim=-1)  # [num_tokens]
            num_routed = routed_mask.sum().item()

            if num_routed == 0:
                continue

            # Get the routing weight for this expert (for routed tokens)
            # Find which position in top_k contains this expert
            expert_position = (top_indices == expert_idx).float()  # [num_tokens, top_k]
            expert_weights = (top_weights * expert_position).sum(dim=-1)  # [num_tokens]

            # Get hidden states for routed tokens
            routed_hidden = hidden_flat[routed_mask]  # [num_routed, hidden]

            # Compute expert activations
            expert = experts[expert_idx]
            expert_output = expert(routed_hidden)  # [num_routed, hidden]

            # Compute activation norms (L2)
            activation_norms = expert_output.norm(dim=-1)  # [num_routed]

            # Get routing weights for routed tokens
            routed_weights = expert_weights[routed_mask]  # [num_routed]

            # Update observations
            obs.expert_frequency[expert_idx] += num_routed
            obs.ean_sum[expert_idx] += activation_norms.sum()
            obs.routing_weight_sum[expert_idx] += routed_weights.sum()
            obs.max_activations[expert_idx] = torch.max(
                obs.max_activations[expert_idx], activation_norms.max()
            )

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_observations(self) -> dict[int, LayerObservation]:
        """Return collected observations."""
        return self.observations

    def save_observations(self, path: str) -> None:
        """Save observations to disk."""
        data = {}
        for layer_idx, obs in self.observations.items():
            data[layer_idx] = {
                "expert_frequency": obs.expert_frequency.cpu(),
                "ean_sum": obs.ean_sum.cpu(),
                "routing_weight_sum": obs.routing_weight_sum.cpu(),
                "max_activations": obs.max_activations.cpu(),
            }
        torch.save(data, path)
        logger.info(f"Saved observations to {path}")

    def load_observations(self, path: str) -> None:
        """Load observations from disk."""
        data = torch.load(path, weights_only=True)
        for layer_idx, obs_data in data.items():
            self.observations[int(layer_idx)] = LayerObservation(
                expert_frequency=obs_data["expert_frequency"].to(self.device),
                ean_sum=obs_data["ean_sum"].to(self.device),
                routing_weight_sum=obs_data["routing_weight_sum"].to(self.device),
                max_activations=obs_data["max_activations"].to(self.device),
            )
        logger.info(f"Loaded observations from {path}")
