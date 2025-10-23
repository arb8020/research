#!/bin/bash
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment (uv creates .venv in workspace root)
# Go up two levels: speedrun -> examples -> workspace root
WORKSPACE_ROOT="$(cd ../.. && pwd)"
if [ -f "$WORKSPACE_ROOT/.venv/bin/activate" ]; then
    source "$WORKSPACE_ROOT/.venv/bin/activate"
    echo "Activated virtual environment: $WORKSPACE_ROOT/.venv"
else
    echo "Warning: Virtual environment not found at $WORKSPACE_ROOT/.venv"
fi

# Default values
SCRIPT="single-file.py"
GPU_COUNT=8

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --script)
      SCRIPT="$2"
      shift 2
      ;;
    --gpu-count)
      GPU_COUNT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: run.sh [--script <path>] [--gpu-count <n>]"
      exit 1
      ;;
  esac
done

echo "========================================="
echo "GPT-2 Training Run"
echo "========================================="
echo "Script: $SCRIPT"
echo "GPUs: $GPU_COUNT"
echo "========================================="

# Download data if not present
if [ ! -d "data/fineweb10B" ]; then
    echo "Downloading FineWeb-10B dataset..."
    python data/cached_fineweb10B.py
    echo "Dataset download complete!"
fi

# Run training with torchrun
# Use PYTHONUNBUFFERED=1 to get real-time output
echo "Starting training..."
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=$GPU_COUNT "$SCRIPT"
