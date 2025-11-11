# Research

GPU provisioning and remote execution utilities for ML research.

## Components

### Infrastructure
- **broker**: GPU cloud provisioning across multiple providers (RunPod, etc.)
- **bifrost**: Remote code deployment and execution via SSH
- **kerbal**: Script execution orchestration (dependency setup, tmux, GPU management)
- **shared**: Common utilities and SSH foundation

### Compute
- **miniray**: Lightweight distributed computing (~600 lines, Heinrich's pattern + TCP)
- **rollouts**: LLM evaluation and agentic RL framework

### Documentation
- **docs/**: Repository-wide documentation (code style, research advice, project notes)
- **dev/**: Active research experiments (promoted to top-level when published)

## Quick Start

```bash
# Install all dependencies
uv sync

# Install with specific experiment dependencies
uv sync --extra dev-outlier-features

# Install individual packages
uv pip install -e miniray/
uv pip install -e rollouts/
```

## Workflow

This repository uses [git-branchless](https://github.com/arxanas/git-branchless) for workflow management.
