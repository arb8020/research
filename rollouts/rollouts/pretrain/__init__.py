"""Minimal pretraining playground.

TODO: Inference/generation
  - Wire up rollouts/inference/ engine to pretrain weights
  - Need: convert functional weights dict to format inference engine expects
  - KV cache already exists in rollouts/inference/kv_cache.py
  - See nanochat/engine.py for reference

TODO: Tokenizer (for data prep + inference)
  - Use tiktoken with o200k_base or similar
  - Only needed when tokenizing raw text -> .npy shards
  - Training already works with pre-tokenized data

TODO: Fused kernels (Triton)
  - Fused cross-entropy (avoid materializing full logits) - see modded-nanogpt triton_kernels.py
    - ~150 lines Triton, computes softmax + log + CE in one pass
    - Only stores log-sum-exp per row, not full logits
  - Fused RMSNorm (simpler, good first Triton kernel to learn API)
  - Alternative: Liger Kernel (linkedin/liger-kernel) for drop-in fused ops
  - Research: pyncd (mit-zardini-lab/pyncd) - algebraic DL expressions, could potentially
    derive fusion opportunities via algebraic rewrites (associativity, distributivity)
  - Also see: CuTe DSL (nvidia-cutlass-dsl) for CUTLASS-level perf, Gluon for lower-level Triton

TODO: SYNTH data pipeline (for fast pretraining experiments)
  Current problem: streaming from HuggingFace is slow (~50+ hrs for 10B tokens locally)
  because of per-request network overhead, not CPU.

  Solution: two-phase approach
  1. Download phase: `huggingface-cli download` or datasets library to download
     parquet files to local/cloud disk. Full dataset is 236GB, but we only need
     ~30GB for 10M samples (~10B tokens).
  2. Tokenize phase: read from local parquet files, tokenize with tiktoken,
     save as .npy shards. This is CPU-bound and fast.

  Infrastructure (using broker + Prime Intellect):
  - Create persistent disk on PI (dc_roan cheapest @ $0.05/GB/mo)
  - Spin up cheapest GPU instance (we only need the fast datacenter network)
  - Download SYNTH subset to attached disk
  - Tokenize to .npy shards on same disk
  - Terminate instance, disk persists with tokenized data
  - Later: attach disk to 8xH100 for training

  Scripts:
  - scripts/tokenize_synth.py - tokenization (works, tested)
  - scripts/pi_tokenize.py - PI disk management (works, tested)
  - TODO: add download phase, make tokenize read from local parquet not stream

  See also: broker/broker/providers/primeintellect.py for disk operations
"""
