# GLM-4.5 MoE Functional Implementation Notes

## Session: 2024-12-16

### What I'm doing
Converting GLM-4.5 (355B MoE, 92 layers) to functional PyTorch, following the pattern from Qwen2.5 and Qwen3.

### Friction Points & Unblocks

#### 1. **Finding the right weight key names**
- **Friction:** Spent time guessing that `model.norm.weight` was in shard 93, but it was actually in shard 92
- **What I did:** Trial and error with debug prints, downloading wrong shard
- **Unblock:** `curl` the HF index JSON and grep for "norm":
  ```bash
  curl -s "https://huggingface.co/zai-org/GLM-4.5/raw/main/model.safetensors.index.json" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); [print(k,v) for k,v in d['weight_map'].items() if 'norm' in k.lower() and 'layers' not in k]"
  ```
- **Util idea:** `get_weight_location(model_id, pattern)` - fetches index.json and returns shard + key name

#### 2. **GPU OOM from duplicate weights**
- **Friction:** Downloaded weights directly to GPU, then HF model also loaded to GPU = 2x memory = OOM
- **What I did:** Initially loaded with `device="cuda:0"` in safetensors, then `model.to(cuda)`
- **Unblock:** Sequential execution - run HF first, delete model, then run functional
- **Util idea:** The test pattern should always be: load to CPU, run HF (save output to CPU), delete HF, run functional, compare

#### 3. **Didn't use existing tools.py utilities**
- **Friction:** Wrote my own hook-based layer capture in `debug_layer()`, reimplemented `capture_intermediate()`
- **What I should have done:** Read `tools/functional_extractor/tools.py` first - has `capture_intermediate()`, `get_weight_info()`, `list_modules()`
- **Util idea:** When starting a new model, first run:
  ```python
  from tools.functional_extractor.tools import list_modules, get_weight_info
  print(list_modules(model, max_depth=2))  # See structure
  print(get_weight_info(model, "norm"))    # Find norm weights
  ```

#### 4. **Partial weight download for large models**
- **Friction:** GLM-4.5 is 717GB across 93 shards. Can't download all of it for testing 5 layers.
- **Unblock:** Manual shard selection based on layer-to-shard mapping:
  - Shard 1: embed_tokens + layer 0
  - Shard 2: layer 1
  - Shard 3: layer 2
  - Shard 4+: MoE layers (7.87GB each)
- **Util idea:** `download_partial_model(model_id, num_layers)` - reads index.json, figures out which shards needed, downloads only those

#### 5. **MoE architecture specifics**
- **Learning:** GLM-4.5 uses `first_k_dense_replace: 3` meaning layers 0-2 are dense MLP, layers 3+ are MoE
- **Learning:** Router uses sigmoid (not softmax) + group-based top-k selection
- **Learning:** Has both routed experts (160) AND shared expert (1) that always runs
- **Util idea:** Config parser that extracts these MoE-specific params and prints a summary

#### 6. **Disk space on remote GPU**
- **Friction:** First GPU had 50GB disk, ran out downloading shards
- **Unblock:** `broker info <gpu_id>` showed disk at 100%, needed to provision with `container_disk=100`
- **Util idea:** Default disk size should be based on model size estimate, not fixed 50GB

### Current Status
- ✅ HF model runs successfully (sequential memory management works)
- ✅ Functional forward runs successfully (no crashes)
- ❌ **Output mismatch: max_diff=12.8** - need to debug layer-by-layer
  - HF: `[-0.4766, 0.0583, -0.1719, 0.9453, 1.9609]`
  - Func: `[2.1562, 3.5156, 3.9219, 2.4375, 2.7188]`

#### 7. **Debugging output mismatch**
- **Friction:** Remote execution doesn't pass CLI args - had to hardcode defaults
- **Learning:** argparse `action="store_true"` ignores `default=True`, need explicit default
- **Util idea:** `run_on_gpu()` should support passing extra script args

#### 8. **Layer-by-layer results**
```
✅ Embeddings: PASS (max_diff=0.00e+00)
✅ Layer 0 (dense): PASS
✅ Layer 1 (dense): PASS
✅ Layer 2 (dense): PASS
❌ Layer 3 (MoE): FAIL (max_diff=7.75e-03)
```
- Dense layers work perfectly!
- **First divergence at Layer 3 (first MoE layer)**
- The diff is 7.75e-03 (not huge), suggests small numerical issue in MoE routing/expert execution
- Sample values look identical at first 5 elements - need to find where max diff is

### Things that worked well
- `--keep-alive` flag to reuse GPU between iterations
- `--gpu-id` to reconnect to existing instance
- Partial shard download saved ~650GB of bandwidth
- bifrost/broker error messages were clear about disk full

### Architecture notes for GLM-4.5
```
- 92 layers total
- Layers 0-2: Dense MLP (gate_proj, up_proj, down_proj)
- Layers 3-91: MoE (160 routed experts + 1 shared expert)
- Router: sigmoid activation, group-based top-k (8 groups, top-4 per group, 8 experts total)
- Attention: QKNorm (RMSNorm on Q and K after projection)
- RoPE: Standard rotary embeddings
- Vocab: 151552 tokens
- Hidden: 6144
- Heads: 48 (128 dim each)
- KV heads: 8 (GQA ratio 6:1)
```
