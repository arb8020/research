"""Backend registry for NVFP4 kernel implementations.

Provides a simple registry system for managing multiple kernel backends
(reference, Triton, CuTe, etc.) with consistent interfaces.
"""
from typing import Callable, Protocol
from dataclasses import dataclass
from kernel_utils.task import input_t, output_t


class KernelBackend(Protocol):
    """Protocol for NVFP4 kernel implementations.

    All kernels must accept input_t and return output_t.
    """

    def __call__(self, data: input_t) -> output_t:
        """Execute the kernel.

        Args:
            data: Input tuple (a, b, scale_a, scale_b, scale_a_perm, scale_b_perm, c)

        Returns:
            Output tensor (modified c)
        """
        ...


@dataclass
class BackendInfo:
    """Metadata about a backend implementation."""

    name: str
    kernel_fn: Callable[[input_t], output_t]
    description: str
    language: str  # "pytorch", "triton", "cuda", "cutlass"

    def __call__(self, data: input_t) -> output_t:
        """Execute this backend's kernel."""
        return self.kernel_fn(data)


class BackendRegistry:
    """Registry for managing multiple kernel backends.

    Example usage:
        >>> from kernel_utils.backends import BACKENDS
        >>> BACKENDS.register("my_kernel", my_kernel_fn, "My custom kernel", "triton")
        >>> kernel = BACKENDS["my_kernel"]
        >>> result = kernel(test_input)
    """

    def __init__(self):
        self._backends: dict[str, BackendInfo] = {}

    def register(
        self,
        name: str,
        kernel_fn: Callable[[input_t], output_t],
        description: str,
        language: str,
    ) -> None:
        """Register a new backend.

        Args:
            name: Unique backend name (e.g., "reference", "triton", "cute")
            kernel_fn: Callable that implements the kernel
            description: Human-readable description
            language: Implementation language ("pytorch", "triton", "cuda", "cutlass")

        Raises:
            AssertionError: If backend name already registered
        """
        assert name not in self._backends, f"Backend '{name}' already registered"
        assert callable(kernel_fn), "kernel_fn must be callable"
        assert len(name) > 0, "Backend name cannot be empty"
        assert language in ["pytorch", "triton", "cuda", "cutlass", "other"], \
            f"Invalid language: {language}"

        self._backends[name] = BackendInfo(name, kernel_fn, description, language)

    def get(self, name: str) -> BackendInfo:
        """Get backend by name.

        Args:
            name: Backend name

        Returns:
            BackendInfo for the specified backend

        Raises:
            KeyError: If backend not found
        """
        if name not in self._backends:
            available = list(self._backends.keys())
            raise KeyError(
                f"Backend '{name}' not found. Available backends: {available}"
            )
        return self._backends[name]

    def list(self) -> list[str]:
        """List all registered backend names.

        Returns:
            List of backend names in registration order
        """
        return list(self._backends.keys())

    def items(self):
        """Iterate over (name, backend_info) pairs."""
        return self._backends.items()

    def __contains__(self, name: str) -> bool:
        """Check if backend is registered."""
        return name in self._backends

    def __getitem__(self, name: str) -> BackendInfo:
        """Get backend by name (dict-like access)."""
        return self.get(name)

    def __len__(self) -> int:
        """Number of registered backends."""
        return len(self._backends)

    def __repr__(self) -> str:
        """String representation."""
        backends = ", ".join(self._backends.keys())
        return f"BackendRegistry({len(self)} backends: {backends})"


# Global registry instance - import and use this
BACKENDS = BackendRegistry()


# Auto-registration removed to avoid circular import issues
# Backends are now registered explicitly in smoke_test.py and kernel modules
