#!/usr/bin/env python3
"""Parallel deployment script for full model sweep.

Launches all models simultaneously for overnight runs.
Each deployment runs in a separate subprocess.

Usage:
    # Dry run (show what would be deployed)
    python deploy_sweep.py --dry-run

    # Launch all MoE models (default)
    python deploy_sweep.py

    # Launch all dense models
    python deploy_sweep.py --config-dir sweep_dense

    # Launch specific models
    python deploy_sweep.py --models 01 03 06

    # Monitor running deployments
    python deploy_sweep.py --status
"""

import subprocess
import time
from pathlib import Path
from typing import List
import argparse
import sys


def discover_configs(config_dir: str = "sweep_configs") -> List[str]:
    """Discover all .py config files in the specified directory."""
    config_path = Path(config_dir)
    if not config_path.exists():
        print(f"Error: Config directory '{config_dir}' does not exist")
        sys.exit(1)

    configs = sorted(config_path.glob("*.py"))
    return [str(c) for c in configs]


def build_model_names(configs: List[str]) -> dict:
    """Build model name mapping from config filenames."""
    names = {}
    for config in configs:
        stem = Path(config).stem
        # Extract model number (first 2 chars)
        model_num = stem[:2]
        # Create human-readable name from filename
        # e.g., "01_qwen3_0.6b" -> "Qwen3-0.6B"
        name_part = stem[3:]  # Skip "01_"
        names[model_num] = name_part.replace("_", "-").upper()
    return names


# Default MoE configs (for backwards compatibility)
DEFAULT_CONFIGS = [
    "sweep_configs/01_olmoe_1b_7b.py",
    "sweep_configs/02_gpt_oss_20b.py",
    "sweep_configs/03_qwen3_30b.py",
    "sweep_configs/04_mixtral_8x7b.py",
    "sweep_configs/05_qwen_next_80b.py",
    "sweep_configs/06_glm_45_air.py",
    "sweep_configs/07_gpt_oss_120b.py",
]

DEFAULT_MODEL_NAMES = {
    "01": "OLMoE-1B-7B (7B)",
    "02": "GPT-OSS-20B (21.5B)",
    "03": "Qwen3-30B (30.5B)",
    "04": "Mixtral-8x7B (46.7B)",
    "05": "Qwen3-Next-80B (80B)",
    "06": "GLM-4.5-Air (106B)",
    "07": "GPT-OSS-120B (117B)",
}


def get_model_number(config_path: str) -> str:
    """Extract model number from config path."""
    return Path(config_path).stem[:2]


def launch_deployment(config_path: Path, model_names: dict, sweep_log_dir: Path, dry_run: bool = False) -> subprocess.Popen | None:
    """Launch a single deployment in background."""
    model_num = get_model_number(str(config_path))
    model_name = model_names.get(model_num, "Unknown")

    # Extract clean model name from config path for log filename
    model_slug = Path(config_path).stem  # e.g., "01_qwen3_0.6b"

    cmd = ["python", "deploy.py", str(config_path)]
    log_file = sweep_log_dir / f"{model_slug}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{model_num}] 🚀 Launching: {model_name}")
    print(f"      Config: {config_path}")
    print(f"      Log: {log_file}")

    if dry_run:
        print(f"      Command: {' '.join(cmd)}")
        print()
        return None

    # Launch in background, redirect output to log file
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent
        )

    print(f"      PID: {proc.pid}")
    print()

    return proc


def check_sweep_status(num_lines: int = 5, sweep_dir: str = None):
    """Check status of most recent sweep by tailing logs.

    Args:
        num_lines: Number of lines to show from each log (default: 5)
        sweep_dir: Optional specific sweep directory to check (e.g., "sweep_20251025_141227_sweep_dense")
    """
    from datetime import datetime
    import glob

    if sweep_dir:
        # Use specified sweep directory
        sweep_path = f"logs/{sweep_dir}" if not sweep_dir.startswith("logs/") else sweep_dir
        if not Path(sweep_path).exists():
            print(f"Sweep directory not found: {sweep_path}")
            return
        latest_sweep = sweep_path
    else:
        # Find most recent sweep log directory
        sweep_dirs = sorted(glob.glob("logs/sweep_*"), reverse=True)
        if not sweep_dirs:
            print("No sweep logs found in logs/")
            return
        latest_sweep = sweep_dirs[0]

    log_files = sorted(glob.glob(f"{latest_sweep}/*.log"))

    if not log_files:
        print(f"No log files found in {latest_sweep}")
        return

    print("=" * 80)
    print(f"SWEEP STATUS: {Path(latest_sweep).name}")
    print("=" * 80)
    print()

    for log_file in log_files:
        model_name = Path(log_file).stem  # e.g., "01_qwen3_0.6b"
        print(f"{'─' * 80}")
        print(f"📊 {model_name}")
        print(f"{'─' * 80}")

        # Read last N lines of log
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                tail_lines = lines[-num_lines:] if len(lines) >= num_lines else lines
                for line in tail_lines:
                    print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"  ❌ Error reading log: {e}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Deploy all models in parallel")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deployed without actually launching"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="sweep_configs",
        help="Config directory to use (default: sweep_configs, use sweep_dense for dense models)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Which models to deploy (e.g., --models 01 03 06, or 'all')"
    )
    parser.add_argument(
        "--status",
        nargs="?",
        const=5,
        type=int,
        metavar="N",
        help="Check status of running deployments (optionally specify number of lines to show, default: 5)"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        help="Specific sweep directory to check status for (e.g., sweep_20251025_141227_sweep_dense)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=30,
        help="Delay in seconds between launches (default: 30)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Status check mode
    if args.status is not None:
        check_sweep_status(num_lines=args.status, sweep_dir=args.sweep)
        return

    # Discover configs from specified directory
    all_configs = discover_configs(args.config_dir)
    model_names = build_model_names(all_configs)

    if not all_configs:
        print(f"Error: No config files found in {args.config_dir}")
        sys.exit(1)

    # Determine which configs to deploy
    if "all" in args.models:
        configs_to_deploy = all_configs
    else:
        configs_to_deploy = [
            cfg for cfg in all_configs
            if get_model_number(cfg) in args.models
        ]

    # Print summary
    print("=" * 80)
    print("OUTLIER FEATURES SWEEP - DEPLOYMENT PLAN")
    print("=" * 80)
    print(f"Config directory: {args.config_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE DEPLOYMENT'}")
    print(f"Models to deploy: {len(configs_to_deploy)}")
    print()

    for cfg in configs_to_deploy:
        model_num = get_model_number(cfg)
        model_name = model_names.get(model_num, "Unknown")
        print(f"  [{model_num}] {model_name}")

    print()
    print(f"Delay between launches: {args.delay}s")
    print(f"Estimated total runtime: 45-70 minutes per model")
    print(f"Total cost estimate: ~$15-25 for all 7 models")
    print("=" * 80)
    print()

    if not args.dry_run and not args.yes:
        response = input("Proceed with deployment? [y/N]: ")
        if response.lower() != 'y':
            print("Deployment cancelled.")
            return
        print()

    # Create sweep log directory: logs/sweep_<timestamp>_<config_dir>/
    # Example: logs/sweep_20251025_135530_dense/
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_dir_name = Path(args.config_dir).name  # "sweep_configs" or "sweep_dense"
    sweep_log_dir = Path(f"logs/sweep_{timestamp}_{config_dir_name}")

    if not args.dry_run:
        sweep_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Sweep logs: {sweep_log_dir}")
        print()

    # Launch all deployments
    processes = []
    for i, config_path in enumerate(configs_to_deploy):
        proc = launch_deployment(Path(config_path), model_names, sweep_log_dir, dry_run=args.dry_run)

        if proc:
            processes.append((get_model_number(config_path), proc))

        # Delay between launches to avoid overwhelming the system
        if i < len(configs_to_deploy) - 1 and not args.dry_run:
            print(f"⏳ Waiting {args.delay}s before next launch...")
            time.sleep(args.delay)
            print()

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN COMPLETE")
        print("=" * 80)
        return

    # Print summary
    print("=" * 80)
    print("ALL DEPLOYMENTS LAUNCHED")
    print("=" * 80)
    print(f"Running processes: {len(processes)}")
    for model_num, proc in processes:
        model_name = model_names.get(model_num, "Unknown")
        print(f"  [{model_num}] {model_name} (PID: {proc.pid})")
    print()
    print("Monitor progress:")
    print(f"  - Check logs: ls -la {sweep_log_dir}/")
    print(f"  - Tail logs: tail -f {sweep_log_dir}/*.log")
    print("  - Check GPUs: broker list")
    print("  - Check instances: python deploy_sweep.py --status")
    print()
    print("Results will be saved to:")
    print("  dev/outlier-features/results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
