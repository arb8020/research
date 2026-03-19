"""Training backends

Available implementations:
- PyTorchTrainingBackend: Standard PyTorch (OOP, stateful) - IMPLEMENTED
- FSDP2TrainingBackend: PyTorch FSDP2 distributed - IMPLEMENTED
- MegatronTrainingBackend: Megatron-Core (TP/PP/EP) - IMPLEMENTED
- NmoeTrainingBackend: reserved name for future native nmoe adapter - STUB / FAIL-LOUD
- TorchFuncTrainingBackend: torch.func + torchopt (functional) - STUB
- JAXTrainingBackend: Raw JAX (pure functional, TPU) - STUB
- TorchaxTrainingBackend: torchax (PyTorch on JAX) - STUB

Implemented backends implement the TrainingBackend protocol. Reserved stubs fail
loud rather than pretending to implement runtime semantics they do not have.
"""

from ...training.backends.fsdp2_backend import FSDP2Config, FSDP2TrainingBackend
from ...training.backends.jax_backend import JAXTrainingBackend
from ...training.backends.megatron_backend import MegatronConfig, MegatronTrainingBackend
from ...training.backends.nmoe_backend import NmoeConfig, NmoeTrainingBackend
from ...training.backends.protocol import TrainingBackend
from ...training.backends.pytorch import PyTorchTrainingBackend
from ...training.backends.pytorch_factory import (
    compute_device_map_single_gpu,
    create_adamw_optimizer,
    create_backend_with_scheduler,
    create_cross_entropy_loss,
    # Tier 2: Convenience
    create_pytorch_backend,
    create_warmup_cosine_scheduler,
    load_hf_model,
    # Tier 1: Granular (export for power users)
    parse_dtype,
)
from ...training.backends.torch_func import TorchFuncTrainingBackend
from ...training.backends.torchax_backend import TorchaxTrainingBackend
from ...training.backends.torchtitan_factory import (
    create_torchtitan_backend,
    ensure_single_rank_torchtitan_dist,
)

__all__ = [
    # Protocol
    "TrainingBackend",
    # Backends
    "PyTorchTrainingBackend",
    "FSDP2TrainingBackend",
    "FSDP2Config",
    "MegatronTrainingBackend",
    "MegatronConfig",
    "NmoeTrainingBackend",
    "NmoeConfig",
    "TorchFuncTrainingBackend",
    "JAXTrainingBackend",
    "TorchaxTrainingBackend",
    # Tier 2: Convenience
    "create_pytorch_backend",
    "create_backend_with_scheduler",
    "create_torchtitan_backend",
    # Tier 1: Granular
    "parse_dtype",
    "compute_device_map_single_gpu",
    "load_hf_model",
    "create_adamw_optimizer",
    "create_cross_entropy_loss",
    "create_warmup_cosine_scheduler",
    "ensure_single_rank_torchtitan_dist",
]
