# JAX GPT-2 Implementation

Clean, well-documented GPT-2 implementation in JAX following best practices:
- **Noam Shazeer shape suffix convention** for clear tensor dimensions
- **Tiger Style assertions** for safety and clarity
- **Casey Muratori API design** for simplicity and composability

## Files

```
backends/jax/
├── kvcache.py      # KV cache with functional JAX array updates
├── layers.py       # Layer implementations (attention, MLP, layer norm)
├── model.py        # GPT-2 model and transformer blocks
├── loader.py       # Weight loading from HuggingFace
├── generate.py     # Generation utilities (placeholder)
├── example.py      # Simple usage example
├── test_gpt2.py    # Test suite comparing against HuggingFace
├── test_gpu.py     # GPU test script (runs on remote)
├── run_gpu_test.py # GPU deployment script
└── README.md       # This file
```

## Installation

```bash
# From research repository root
cd ~/research

# Install dependencies with uv
uv sync --extra example-nano-inference-jax

# This installs:
# - JAX (GPU on Linux, CPU on macOS)
# - transformers, huggingface_hub, safetensors
# - numpy and other dependencies
```

## Usage

### Quick Example

```bash
uv run python examples/nano-inference/backends/jax/example.py
```

This will:
1. Download GPT-2 weights from HuggingFace
2. Run a forward pass on "Hello world"
3. Show top-5 predicted next tokens

### Run Tests

```bash
# Test on 5 different inputs
uv run python examples/nano-inference/backends/jax/test_gpt2.py

# Test on more batches
uv run python examples/nano-inference/backends/jax/test_gpt2.py --batches 10
```

Tests compare JAX implementation against HuggingFace reference.

### Use in Your Code

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("examples/nano-inference")))

import jax.numpy as jnp
from backends.jax.model import gpt2_forward
from backends.jax.loader import load_weights
from config import GPT2Config

# Load weights
weights = load_weights("gpt2")
config = GPT2Config()

# Run inference
input_ids_BT = jnp.array([[15496, 995]])  # "Hello world"
logits_BTV = gpt2_forward(input_ids_BT, weights, config)
```

## Shape Suffix Convention

Variables use Noam Shazeer's shape suffix convention:

```python
# Shape suffixes:
#   B: batch_size
#   T: seq_len (time/sequence dimension)
#   D: d_model
#   F: d_ff (feedforward hidden dimension)
#   H: head_dim (dimension per head)
#   N: n_heads (number of query heads)
#   K: n_kv_heads (number of key-value heads)
#   V: vocab_size

# Examples:
input_ids_BT     # (batch_size, seq_len)
logits_BTV       # (batch_size, seq_len, vocab_size)
hidden_BTD       # (batch_size, seq_len, d_model)
attn_weights_BNTT # (batch_size, n_heads, seq_len, seq_len)
```

This makes tensor shapes immediately obvious from variable names!

## Code Style

Follows three key principles:

### 1. Tiger Style Safety
- Comprehensive assertions (2+ per function)
- Split compound assertions for clarity
- Assert both positive and negative spaces

### 2. Shape Suffixes
- Every tensor variable has shape suffix
- Makes dimensions clear without comments

### 3. Casey Muratori API Design
- Simple, transparent functions
- No hidden coupling or state
- Clear granularity

## Testing

### CPU Testing

The test suite compares against HuggingFace GPT-2:

```bash
$ uv run python examples/nano-inference/backends/jax/test_gpt2.py

🧪 Testing JAX GPT-2 vs HuggingFace across multiple batches
======================================================================

📊 Batch 1/5: Hello world
----------------------------------------
Input tokens: [[15496, 995]]
Max absolute difference: 0.000123
✅ PASS

...

📋 SUMMARY REPORT
======================================================================
Total batches tested: 5
Batches passed: 5
Pass rate: 100.0%

🎉 ALL TESTS PASSED!
```

### GPU Testing

Deploy and test on a remote GPU instance:

```bash
# First time: provision new GPU and run test
python examples/nano-inference/backends/jax/run_gpu_test.py

# Keep instance running for fast iteration
# Output will show: "Iterate: python backends/jax/run_gpu_test.py --use-existing jax-gpt2-dev"

# Fast iteration: reuse existing instance (no bootstrap needed)
python examples/nano-inference/backends/jax/run_gpu_test.py --use-existing jax-gpt2-dev

# Or with direct SSH
python examples/nano-inference/backends/jax/run_gpu_test.py --use-existing root@123.45.67.89:22

# Provision and terminate after test
python examples/nano-inference/backends/jax/run_gpu_test.py --terminate
```

**Prerequisites for GPU testing:**
```bash
# Set API keys
export RUNPOD_API_KEY="your-key"
export SSH_KEY_PATH="~/.ssh/id_ed25519"  # optional
```

The `--use-existing` flag makes iteration fast:
- ✅ **First run**: ~5 min (provision + bootstrap + test)
- ✅ **Subsequent runs**: ~30 sec (just sync code + test)

## Dependencies

Uses utilities from `examples/nano-inference/utils/`:
- `utils.weights` - Weight loading (safetensors, pytorch, npz)
- `utils.comparison` - Logits comparison utilities

## Implementation Notes

### JAX Immutability

JAX arrays are immutable. The KV cache pattern:

```python
# Update creates NEW arrays
new_k_LTKD = cache.k_LTKD.at[layer_idx, start_pos:end_pos].set(k_TKD)
new_v_LTKD = cache.v_LTKD.at[layer_idx, start_pos:end_pos].set(v_TKD)

# Return new KVCache dataclass
return KVCache(k_LTKD=new_k_LTKD, v_LTKD=new_v_LTKD)
```

### Weight Format

Weights are loaded with HuggingFace naming convention:
- `transformer.wte.weight` - Token embeddings
- `transformer.wpe.weight` - Position embeddings
- `transformer.h.{i}.attn.c_attn.weight` - QKV projection
- `transformer.h.{i}.mlp.c_fc.weight` - MLP first layer
- etc.

Convenience aliases added:
- `wte` → `transformer.wte.weight`
- `wpe` → `transformer.wpe.weight`

## References

- [Noam Shazeer shape suffixes](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd)
- [Tiger Style (TigerBeetle)](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md)
- [Casey Muratori on Code Reuse](https://www.youtube.com/watch?v=ZQ5_u8Lgvyk)
- [JAX Documentation](https://jax.readthedocs.io/)
