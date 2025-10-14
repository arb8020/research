# Research

GPU provisioning and remote execution utilities for ML research.

## Components

- **broker**: GPU cloud provisioning across multiple providers (RunPod, etc.)
- **bifrost**: Remote code deployment and execution
- **shared**: Common utilities and SSH foundation

## Development

This repository uses [git-branchless](https://github.com/arxanas/git-branchless) for workflow management.

## Quick Start

```bash
# Install dependencies
uv sync

# Provision a GPU
python examples/phase1_example.py
```
