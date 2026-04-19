"""
Pokemon RL experiment 01 — PPO self-play vs random opponent, gen9randombattle.

Run locally:
    python -m argus run --config dev/pokemon-rl/experiment_01.py --local

Run on Modal:
    python -m argus run --config dev/pokemon-rl/experiment_01.py --provider modal
"""

from __future__ import annotations

import sys
import os

from dataclasses import dataclass
from rollouts.training.configs import DepsConfig, HardwareConfig

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

hardware = HardwareConfig(
    gpu_type="L4",
    gpu_count=1,
    cpu_count=16.0,
    provider="modal",
    deps=DepsConfig(
        python_version="3.12",
        system_packages=(
            "bash", "curl", "git", "build-essential", "libnuma1", "tmux",
        ),
        pip_packages=(
            "torch>=2.4",
            "numpy",
            "gymnasium",
            "trio",
            "trio-asyncio",
            "poke-env @ git+https://github.com/hsahovic/poke-env.git",
        ),
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
        bootstrap_commands=(
            # Install nvm + Node v20 (better-sqlite3 requires v20, not v24)
            "curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash",
            "bash -c 'source ~/.nvm/nvm.sh && nvm install 20 && nvm use 20'",
            # Symlink node/npm into /usr/local/bin so subprocesses can find them
            "bash -c 'ln -sf $(source ~/.nvm/nvm.sh && nvm which 20) /usr/local/bin/node'",
            "bash -c 'ln -sf $(dirname $(source ~/.nvm/nvm.sh && nvm which 20))/npm /usr/local/bin/npm'",
            # Clone and install Pokemon Showdown
            "git clone --depth 1 https://github.com/smogon/pokemon-showdown.git /opt/pokemon-showdown",
            "bash -c 'source ~/.nvm/nvm.sh && nvm use 20 && npm install --prefix /opt/pokemon-showdown'",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PokemonRLConfig:
    format_id: str = "gen9randombattle"
    n_envs: int = 16
    horizon: int = 64
    total_steps: int = 20_000_000
    lr: float = 3e-4
    hidden_size: int = 256
    lstm_size: int = 256
    checkpoint_dir: str = "checkpoints"


config = PokemonRLConfig()

# ---------------------------------------------------------------------------
# Train entry point
# ---------------------------------------------------------------------------

def train(config: PokemonRLConfig | None = None, **kwargs):
    import glob

    # Wire up Node + Showdown paths for the remote environment
    import shutil
    node_bins = glob.glob("/root/.nvm/versions/node/v20.*/bin/node")
    if node_bins:
        os.environ["NODE_BIN"] = sorted(node_bins)[-1]
    elif shutil.which("node"):
        os.environ["NODE_BIN"] = shutil.which("node")
    # else: leave NODE_BIN as-is (local default in sim_bridge.py)
    if os.path.exists("/opt/pokemon-showdown/pokemon-showdown"):
        os.environ["SHOWDOWN_PATH"] = "/opt/pokemon-showdown/pokemon-showdown"
    import logging as _logging
    _logging.getLogger("experiment").info(
        f"NODE_BIN={os.environ.get('NODE_BIN')} "
        f"SHOWDOWN_PATH={os.environ.get('SHOWDOWN_PATH')}"
    )

    # Add pokemon-rl package to path.
    # On Modal the repo is cloned to /root/research; locally it's the workspace root.
    # Walk up from this file until we find dev/pokemon-rl.
    _here = os.path.abspath(__file__)
    _candidate = _here
    for _ in range(6):
        _candidate = os.path.dirname(_candidate)
        _pkg = os.path.join(_candidate, "dev", "pokemon-rl")
        if os.path.isdir(_pkg):
            if _pkg not in sys.path:
                sys.path.insert(0, _pkg)
            break

    from train import Config, train as ppo_train

    cfg = config or PokemonRLConfig()
    ppo_cfg = Config(
        format_id=cfg.format_id,
        n_envs=cfg.n_envs,
        horizon=cfg.horizon,
        total_steps=cfg.total_steps,
        lr=cfg.lr,
        hidden_size=cfg.hidden_size,
        lstm_size=cfg.lstm_size,
        checkpoint_dir=cfg.checkpoint_dir,
    )
    ppo_train(ppo_cfg)
    return {"metrics_history": []}
