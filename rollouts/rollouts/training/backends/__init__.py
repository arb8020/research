"""Training backends

Available implementations:
- PyTorchTrainingBackend (D6v1): Standard PyTorch (OOP, stateful) - IMPLEMENTED
- TorchFuncTrainingBackend (D6v2): torch.func + torchopt (functional) - STUB
- JAXTrainingBackend (D6v3): Raw JAX (pure functional, TPU) - STUB
- TorchaxTrainingBackend (D6v4): torchax (PyTorch on JAX) - STUB (experimental)

All backends implement the TrainingBackend protocol.
"""

from ...training.backends.jax_backend import JAXTrainingBackend
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

__all__ = [
    "TrainingBackend",
    "PyTorchTrainingBackend",
    "TorchFuncTrainingBackend",
    "JAXTrainingBackend",
    "TorchaxTrainingBackend",
    # Tier 2
    "create_pytorch_backend",
    "create_backend_with_scheduler",
    # Tier 1
    "parse_dtype",
    "compute_device_map_single_gpu",
    "load_hf_model",
    "create_adamw_optimizer",
    "create_cross_entropy_loss",
    "create_warmup_cosine_scheduler",
]
