"""KV cache implementations.

Two backends:
- PagedKVCache: vLLM-style block allocation with hash-based prefix caching
- RadixKVCache: SGLang-style prefix tree (TODO)
"""

from rollouts.inference.cache.paged import PagedKVCache, Block
from rollouts.inference.cache.radix import RadixKVCache
from rollouts.inference.cache.protocol import KVCacheManager

__all__ = ["PagedKVCache", "RadixKVCache", "Block", "KVCacheManager"]
