#!/usr/bin/env python3
"""Convenience script to run the dev loop server.

Usage:
    # From anywhere:
    python ~/research/rollouts/rollouts/frontend/run.py

    # With custom project:
    python ~/research/rollouts/rollouts/frontend/run.py --project ~/my-agent-project
"""
import sys
from pathlib import Path

# Add rollouts to path
rollouts_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rollouts_root))

# Server doesn't need trio - it's standalone
if __name__ == "__main__":
    from rollouts.frontend.server import main
    main()
