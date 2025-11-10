"""Training backends

Available implementations:
- PyTorchTrainingBackend (D6v1): Standard PyTorch (OOP, stateful) - IMPLEMENTED
- TorchFuncTrainingBackend (D6v2): torch.func + torchopt (functional) - STUB
- JAXTrainingBackend (D6v3): Raw JAX (pure functional, TPU) - STUB
- TorchaxTrainingBackend (D6v4): torchax (PyTorch on JAX) - STUB (experimental)

All backends implement the TrainingBackend protocol.
"""

from rollouts.training.backends.protocol import TrainingBackend
from rollouts.training.backends.pytorch import PyTorchTrainingBackend
from rollouts.training.backends.torch_func import TorchFuncTrainingBackend
from rollouts.training.backends.jax_backend import JAXTrainingBackend
from rollouts.training.backends.torchax_backend import TorchaxTrainingBackend

__all__ = [
    "TrainingBackend",
    "PyTorchTrainingBackend",
    "TorchFuncTrainingBackend",
    "JAXTrainingBackend",
    "TorchaxTrainingBackend",
]
