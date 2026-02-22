# NCCL Weight Sync Debug Info

## Problem Summary

2-GPU GRPO training with NCCL weight sync fails after Step 1 completes.

**Error:**
```
NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.cpp:94, unhandled cuda error
ncclUnhandledCudaError: Call to CUDA function failed.
Cuda failure 'invalid argument'. The full weights of the ModelRunner are partially updated.
```

**What worked:**
- Step 1 training completed: `reward=0.324 | pg_loss=0.0239`
- NCCL group initialization succeeded
- SGLang launched on GPU 0, trainer on GPU 1

**What failed:**
- Weight broadcast from trainer (GPU 1) to SGLang (GPU 0) after Step 1
- HTTP response: `POST /update_weights_from_distributed HTTP/1.1" 400 Bad Request`

## Architecture

```
GPU 0: SGLang inference (rank 1 in NCCL group)
GPU 1: PyTorch FSDP trainer (rank 0 in NCCL group)

Trainer                          SGLang
   |                                |
   |-- POST /init_weights_update_group (join NCCL) -->
   |                                |
   |<-- dist.init_process_group ----|
   |                                |
   [NCCL group ready, world_size=2] |
   |                                |
   [Step 1 training]                |
   |                                |
   |-- POST /update_weights_from_distributed -->
   |   (names, shapes, dtypes)      |
   |                                |
   |-- dist.broadcast(param) ------>|  <-- FAILS HERE
   |                                |
```

## Code Path

### 1. NCCL Group Init (works)
**File:** `rollouts/training/backends/pytorch.py:815-927`

```python
async def init_nccl_weight_sync(self, inference_endpoints, master_addr, master_port):
    # 1. HTTP to SGLang: POST /init_weights_update_group
    # 2. Trainer: create_stateless_process_group(rank=0)
    # 3. Both join concurrently via trio nursery
```

### 2. Weight Sync (fails)
**File:** `rollouts/training/backends/pytorch.py:929-1016`

```python
async def sync_weights_nccl(self):
    # 1. Get state_dict from model
    # 2. Build param_info list (names, shapes, dtypes)
    # 3. Concurrently:
    #    - POST /update_weights_from_distributed to SGLang
    #    - dist.broadcast(param) for each param
    # 4. dist.barrier()
```

### 3. SGLang Side
SGLang's `/update_weights_from_distributed` endpoint:
- Receives param metadata via HTTP
- Calls `dist.broadcast()` to receive each tensor
- Updates model weights in-place

## Logs

### Training Log (relevant section)
```
[22:21:54] Initializing NCCL weight sync with 1 engine(s)...
[22:21:54] Initializing NCCL weight sync group (world_size=2)
[22:21:55] NCCL weight sync group initialized successfully
[22:21:55] NCCL weight sync initialized
--- Step 1/100 ---
[22:22:13] Step 1: reward=0.324 | pg_loss=0.0239 | entropy=0.64
[22:22:13] Syncing weights via NCCL to sglang...
[22:22:14] Failed to update parameter online: NCCL error in: NCCLUtils.cpp:94
           ncclUnhandledCudaError: Call to CUDA function failed.
           Cuda failure 'invalid argument'.
[22:22:14] POST /update_weights_from_distributed HTTP/1.1" 400 Bad Request
```

### SGLang Server Args
```
base_gpu_id=0
tp_size=1
device='cuda'
```

## Possible Causes

1. **CUDA context mismatch**: Trainer is on GPU 1 but tensors may be created on wrong device
2. **Shape/dtype mismatch**: State dict from FSDP may have different shapes than SGLang expects
3. **Process group mismatch**: SGLang may not be using the same NCCL group as trainer
4. **Tensor not contiguous**: `param.data.contiguous()` may not be enough

## Files to Check

1. **pytorch.py:993-1004** - The broadcast loop:
```python
def _do_broadcast() -> None:
    for name, param in state_dict.items():
        param_data = param.data.contiguous()
        if param_data.device.type != "cuda":
            param_data = param_data.cuda()  # <-- Which GPU?
        dist.broadcast(param_data, src=0, group=self._nccl_process_group)
    dist.barrier(group=self._nccl_process_group)
```

2. **SGLang's update_weights_from_distributed handler** - Check if it expects specific tensor layout

3. **FSDP state_dict format** - May need `StateDictType.FULL_STATE_DICT` options

## Debug Steps

1. Add `NCCL_DEBUG=INFO` to see detailed NCCL errors
2. Print tensor device/shape before broadcast
3. Check if SGLang's model has same param names/shapes
4. Verify CUDA device is set correctly before broadcast

## Config Used

```python
# grpo_2gpu_01.py
hardware = HardwareConfig(
    gpu_type="RTX A5000",
    gpu_count=2,
    provider="runpod",
)

config = GRPOConfig(
    checkpoint=CheckpointConfig(
        weight_sync_mode="nccl",  # GPU-to-GPU broadcast
    ),
    trainer=TrainerConfig(
        cuda_device_ids=(1,),  # GPU 1 for training
    ),
    inference=InferenceConfig(
        cuda_device_ids=(0,),  # GPU 0 for inference
        port=30000,
        mem_fraction=0.9,
    ),
)
```

## Full Log File

See: `results/rl/run_20260221-221732/training.log`
