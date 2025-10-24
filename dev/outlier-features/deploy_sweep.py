#!/usr/bin/env python3
"""Parallel deployment script for full model sweep.

Launches all 7 models simultaneously for overnight runs.
Each deployment runs in a separate subprocess.

Usage:
    # Dry run (show what would be deployed)
    python deploy_sweep.py --dry-run

    # Launch all 7 models
    python deploy_sweep.py

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

# All 7 model configs in order
ALL_CONFIGS = [
    "sweep_configs/01_olmoe_1b_7b.py",
    "sweep_configs/02_gpt_oss_20b.py",
    "sweep_configs/03_qwen3_30b.py",
    "sweep_configs/04_mixtral_8x7b.py",
    "sweep_configs/05_qwen_next_80b.py",
    "sweep_configs/06_glm_45_air.py",
    "sweep_configs/07_gpt_oss_120b.py",
]

MODEL_NAMES = {
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


def launch_deployment(config_path: Path, dry_run: bool = False) -> subprocess.Popen | None:
    """Launch a single deployment in background."""
    model_num = get_model_number(str(config_path))
    model_name = MODEL_NAMES.get(model_num, "Unknown")

    cmd = ["python", "deploy.py", str(config_path)]
    log_file = Path(f"logs/deploy_{model_num}_{int(time.time())}.log")
    log_file.parent.mkdir(exist_ok=True)

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


def check_broker_status():
    """Check current GPU instances via broker."""
    print("=" * 80)
    print("CURRENT GPU INSTANCES")
    print("=" * 80)
    subprocess.run(["broker", "list"])
    print()


def main():
    parser = argparse.ArgumentParser(description="Deploy all models in parallel")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deployed without actually launching"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["01", "02", "03", "04", "05", "06", "07", "all"],
        default=["all"],
        help="Which models to deploy (e.g., --models 01 03 06)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check status of running deployments"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=30,
        help="Delay in seconds between launches (default: 30)"
    )

    args = parser.parse_args()

    # Status check mode
    if args.status:
        check_broker_status()
        return

    # Determine which configs to deploy
    if "all" in args.models:
        configs_to_deploy = ALL_CONFIGS
    else:
        configs_to_deploy = [
            cfg for cfg in ALL_CONFIGS
            if get_model_number(cfg) in args.models
        ]

    # Print summary
    print("=" * 80)
    print("OUTLIER FEATURES SWEEP - DEPLOYMENT PLAN")
    print("=" * 80)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE DEPLOYMENT'}")
    print(f"Models to deploy: {len(configs_to_deploy)}")
    print()

    for cfg in configs_to_deploy:
        model_num = get_model_number(cfg)
        model_name = MODEL_NAMES.get(model_num, "Unknown")
        print(f"  [{model_num}] {model_name}")

    print()
    print(f"Delay between launches: {args.delay}s")
    print(f"Estimated total runtime: 45-70 minutes per model")
    print(f"Total cost estimate: ~$15-25 for all 7 models")
    print("=" * 80)
    print()

    if not args.dry_run:
        response = input("Proceed with deployment? [y/N]: ")
        if response.lower() != 'y':
            print("Deployment cancelled.")
            return
        print()

    # Launch all deployments
    processes = []
    for i, config_path in enumerate(configs_to_deploy):
        proc = launch_deployment(Path(config_path), dry_run=args.dry_run)

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
        model_name = MODEL_NAMES.get(model_num, "Unknown")
        print(f"  [{model_num}] {model_name} (PID: {proc.pid})")
    print()
    print("Monitor progress:")
    print("  - Check logs: ls -lt logs/deploy_*.log")
    print("  - Check GPUs: broker list")
    print("  - Check instances: python deploy_sweep.py --status")
    print()
    print("Results will be saved to:")
    print("  examples/outlier-features/results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
