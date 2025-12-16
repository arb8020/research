"""KV cache implementations.

Two backends:
- PagedKVCache: vLLM-style block allocation with hash-based prefix caching
- RadixKVCache: SGLang-style prefix tree (TODO)
"""

from rollouts.inference.cache.paged import Block, PagedKVCache
from rollouts.inference.cache.protocol import KVCacheManager
from rollouts.inference.cache.radix import RadixKVCache

__all__ = ["PagedKVCache", "RadixKVCache", "Block", "KVCacheManager"]
