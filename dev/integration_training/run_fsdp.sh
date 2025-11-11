#!/bin/bash
# Launch FSDP training with torchrun
#
# Usage:
#   bash run_fsdp.sh [config_file] [num_gpus] [gpu_list]
#
# Examples:
#   bash run_fsdp.sh configs/02_debug_sft_fsdp.py 4
#   bash run_fsdp.sh configs/02_debug_sft_fsdp.py 4 "0,1,2,3"
#   bash run_fsdp.sh configs/02_debug_sft_fsdp.py 2 "0,2"  # non-contiguous GPUs

set -e

CONFIG_FILE=${1:-configs/02_debug_sft_fsdp.py}
NUM_GPUS=${2:-4}
GPU_LIST=${3:-""}

echo "==================================="
echo "Launching FSDP Training"
echo "Config: $CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
if [ -n "$GPU_LIST" ]; then
    echo "GPU List: $GPU_LIST"
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
fi
echo "==================================="
echo

# Launch with torchrun (replaces python -m torch.distributed.launch)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train.py "$CONFIG_FILE"
