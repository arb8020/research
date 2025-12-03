#!/usr/bin/env python3
"""Deploy kernel_utils smoke test to remote GPU via bifrost.

Usage:
    # Simple: Compare new kernel vs reference
    python smoke_deploy.py --ssh root@host:port --new triton

    # Compare against custom reference
    python smoke_deploy.py --ssh root@host:port --reference cute --new triton_v2

    # Test multiple backends
    python smoke_deploy.py --ssh root@host:port --backends reference triton cute

    # Save results
    python smoke_deploy.py --ssh root@host:port --new triton --save

    # Enable profiling with torch.profiler
    python smoke_deploy.py --ssh root@host:port --new triton --profile

    # Enable NCU profiling
    python smoke_deploy.py --ssh root@host:port --new triton --ncu

    # Enable both torch.profiler and NCU
    python smoke_deploy.py --ssh root@host:port --new triton --profile --ncu

    # Save results and all profiles
    python smoke_deploy.py --ssh root@host:port --new triton --save --profile --ncu

    # Test on specific GPU
    python smoke_deploy.py --ssh root@host:port --gpu 1 --new triton
"""
import argparse
import logging
import sys
from pathlib import Path

from bifrost.client import BifrostClient

# Import shared logging
from shared.logging_config import setup_logging

from kerbal import DependencyConfig, setup_script_deps

logger = logging.getLogger(__name__)


def deploy_and_test(
    ssh_target: str,
    ssh_key: str,
    gpu_id: int,
    backends: list[str] | None = None,
    save_results: bool = False,
    enable_profiling: bool = False,
    enable_ncu: bool = False,
) -> tuple[bool, str]:
    """Deploy kernel utils and run smoke test on remote.

    Args:
        ssh_target: SSH connection string (user@host:port)
        ssh_key: Path to SSH private key
        gpu_id: GPU device ID to use (e.g., 0, 1, 2, ...)
        backends: List of backend names to test (None = all registered)
        save_results: Save results to JSON on remote
        enable_profiling: Enable torch.profiler for detailed performance traces
        enable_ncu: Enable NVIDIA Nsight Compute profiling

    Returns:
        (success, message)
    """
    logger.info("🚀 Deploying kernel_utils smoke test")
    logger.info(f"   Target: {ssh_target}")
    logger.info(f"   GPU: {gpu_id}")
    logger.info("=" * 60)

    # Connect
    client = BifrostClient(ssh_target, ssh_key)

    # Deploy code
    logger.info("\n📦 Deploying code...")
    workspace = client.push(workspace_path="~/.bifrost/workspaces/kernel_smoke")
    logger.info(f"   Deployed to: {workspace}")

    # Expand path
    result = client.exec(f"echo {workspace}")
    if result.exit_code != 0:
        return False, "Failed to expand workspace path"
    workspace_expanded = result.stdout.strip()

    project_dir = f"{workspace_expanded}/dev/kernels-gpumode"

    # Check directory exists
    result = client.exec(f"test -d {project_dir} && echo OK || echo MISSING")
    if result.stdout.strip() != "OK":
        return False, f"Project directory not found: {project_dir}"

    logger.info(f"   Project dir: {project_dir}")

    # Setup dependencies
    logger.info("\n📦 Setting up dependencies...")
    deps = DependencyConfig(
        project_name="kernel-smoke",
        dependencies=[
            "torch>=2.4.0",
            "triton",  # For GPU benchmarking
            "nvidia-cutlass @ git+https://github.com/NVIDIA/cutlass.git@main",  # For CUTLASS kernels
        ],
        python_version=">=3.10",
    )

    setup_script_deps(client, project_dir, deps, install_extras=None)

    # Run smoke test
    logger.info(f"\n🔥 Running smoke test on GPU {gpu_id}...")

    # Build command with optional backend selection
    cmd_parts = [
        f"cd {project_dir}",
        f"CUDA_VISIBLE_DEVICES={gpu_id} python -m kernel_utils.smoke_test",
    ]

    # Add backend selection if specified
    if backends:
        backend_args = " ".join(backends)
        cmd_parts[1] += f" {backend_args}"

    # Add --save flag if requested
    if save_results:
        cmd_parts[1] += " --save"

    # Add --profile flag if requested
    if enable_profiling:
        cmd_parts[1] += " --profile"

    # Add --ncu flag if requested
    if enable_ncu:
        cmd_parts[1] += " --ncu"

    cmd = " && ".join(cmd_parts)
    logger.info(f"   Command: {cmd}")

    result = client.exec(cmd)

    logger.info(result.stdout)
    if result.stderr:
        logger.warning(f"stderr: {result.stderr}")

    success = (result.exit_code == 0)

    # Enhanced message with backend info
    if backends:
        backend_list = ", ".join(backends)
        message = f"Tested {len(backends)} backend(s): {backend_list}"
    else:
        message = "Tested all registered backends"

    if success:
        message = f"✅ {message} - All passed"
    else:
        message = f"❌ {message} - Some failed (exit {result.exit_code})"

    # Download results if saved
    if save_results and success:
        logger.info("\n📥 Downloading results...")
        results_remote = f"{project_dir}/results/smoke_test_results.json"
        results_local = Path("results/smoke_test_results_remote.json")

        # Check if results exist
        check_result = client.exec(f"test -f {results_remote} && echo OK || echo MISSING")
        if check_result.stdout.strip() == "OK":
            # Pull results file
            results_local.parent.mkdir(parents=True, exist_ok=True)
            pull_result = client.exec(f"cat {results_remote}")
            if pull_result.exit_code == 0:
                results_local.write_text(pull_result.stdout)
                logger.info(f"   Saved to: {results_local}")
            else:
                logger.warning("   Failed to download results")
        else:
            logger.warning("   Results file not found on remote")

    # Download profile traces if profiling was enabled
    if enable_profiling and success:
        logger.info("\n📥 Downloading torch profile traces...")
        profiles_remote = f"{project_dir}/profiles"
        profiles_local = Path("profiles_remote")

        # Check if profiles directory exists
        check_result = client.exec(f"test -d {profiles_remote} && echo OK || echo MISSING")
        if check_result.stdout.strip() == "OK":
            # List profile files (only .json, not tensorboard files)
            list_result = client.exec(f"find {profiles_remote} -name '*.json' -type f")
            if list_result.exit_code == 0:
                profile_files = [f.strip() for f in list_result.stdout.strip().split('\n') if f.strip()]

                if profile_files:
                    profiles_local.mkdir(parents=True, exist_ok=True)
                    for remote_file in profile_files:
                        filename = Path(remote_file).name
                        local_file = profiles_local / filename

                        pull_result = client.exec(f"cat {remote_file}")
                        if pull_result.exit_code == 0:
                            local_file.write_text(pull_result.stdout)
                            logger.info(f"   Downloaded: {local_file}")
                        else:
                            logger.warning(f"   Failed to download: {filename}")

                    logger.info("\n   📊 View torch profiles:")
                    logger.info(f"      Chrome trace: Open chrome://tracing and load files from {profiles_local}/")
                    logger.info(f"      TensorBoard: tensorboard --logdir={profiles_local}/")
                else:
                    logger.warning("   No profile files found")
            else:
                logger.warning("   Failed to list profile files")
        else:
            logger.warning("   Profiles directory not found on remote")

    # Download NCU reports if NCU profiling was enabled
    if enable_ncu and success:
        logger.info("\n📥 Downloading NCU reports...")
        ncu_remote = f"{project_dir}/ncu_reports"
        ncu_local = Path("ncu_reports_remote")

        # Check if ncu_reports directory exists
        check_result = client.exec(f"test -d {ncu_remote} && echo OK || echo MISSING")
        if check_result.stdout.strip() == "OK":
            # Use bifrost's download_files for efficient binary transfer
            ncu_local.mkdir(parents=True, exist_ok=True)

            logger.info(f"   Downloading from {ncu_remote}...")
            result = client.download_files(
                remote_path=ncu_remote,
                local_path=str(ncu_local),
                recursive=True
            )

            if result.success:
                # List what we downloaded
                ncu_files = list(ncu_local.glob("*.csv"))
                if ncu_files:
                    for local_file in ncu_files:
                        size_kb = local_file.stat().st_size / 1024
                        logger.info(f"   Downloaded: {local_file.name} ({size_kb:.1f} KB)")

                    logger.info("\n   📊 View NCU reports:")
                    logger.info(f"      CSV files in {ncu_local}/")
                else:
                    logger.warning("   No NCU CSV files found")
            else:
                logger.warning(f"   Failed to download NCU reports: {result.error}")
        else:
            logger.warning("   NCU reports directory not found on remote")

    return success, message


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy kernel smoke test to remote GPU",
        epilog="""
Examples:
  # Compare new kernel vs reference (simple API)
  python smoke_deploy.py --ssh root@gpu-server:22 --new triton

  # Compare new kernel vs custom reference
  python smoke_deploy.py --ssh root@gpu-server:22 --reference cute --new triton_v2

  # Test multiple backends (advanced)
  python smoke_deploy.py --ssh root@gpu-server:22 --backends reference triton cute

  # Save and download results
  python smoke_deploy.py --ssh root@gpu-server:22 --new triton --save
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ssh", required=True, help="SSH connection (user@host:port)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (default: 0)")
    parser.add_argument(
        "--backends",
        nargs="*",
        help="Backend names to test (default: all registered backends)",
    )
    parser.add_argument(
        "--reference",
        help="Reference backend to compare against (default: 'reference')",
    )
    parser.add_argument(
        "--new",
        help="New backend to test (shorthand for --backends <ref> <new>)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON and download locally",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch.profiler and download trace files",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Enable NVIDIA Nsight Compute profiling and download reports",
    )
    args = parser.parse_args()

    # Handle --reference and --new shorthand
    if args.new:
        reference = args.reference or "reference"
        args.backends = [reference, args.new]
        if args.reference:
            # Update the reference backend for smoke_test
            logger.info(f"Comparing {args.new} vs {reference}")

    # Setup logging
    setup_logging(
        level="INFO",
        logger_levels={
            "httpx": "WARNING",
            "urllib3": "WARNING",
            "paramiko": "WARNING",
        }
    )

    success, message = deploy_and_test(
        args.ssh,
        args.ssh_key,
        args.gpu,
        backends=args.backends,
        save_results=args.save,
        enable_profiling=args.profile,
        enable_ncu=args.ncu,
    )

    if success:
        logger.info(f"\n{message}")
        return 0
    else:
        logger.error(f"\n{message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
