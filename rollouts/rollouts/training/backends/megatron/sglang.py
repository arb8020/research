from __future__ import annotations

"""Backend-local SGLang compatibility shim for the Megatron adapter.

This mirrors the role of miles/slime's `megatron_utils/sglang.py`: isolate
version-specific SGLang imports behind one local boundary instead of smearing
them across worker/update-weight code.
"""

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

from sglang.srt.utils import MultiprocessingSerializer

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

__all__ = [
    "FlattenedTensorBucket",
    "MultiprocessingSerializer",
    "monkey_patch_torch_reductions",
]
