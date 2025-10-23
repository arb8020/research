#!/usr/bin/env python3
"""Automated GPU deployment for corpus-proximity indexing pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Optional, TypeAlias

from dotenv import load_dotenv

from broker import CloudType, GPUClient, GPUInstance
from bifrost import BifrostClient
from config import Config
from cluster_corpus import get_cache_key
from shared.config import get_prime_key, get_runpod_key
from shared.logging_config import setup_logging


SCRIPT_DIR = Path(__file__).resolve().parent
REMOTE_WORKSPACE_PATH = "~/.bifrost/workspace/examples/corpus-proximity"
TMUX_SESSION = "corpus_proximity_pipeline"

ProvisionError: TypeAlias = Literal["create_failed", "ready_timeout", "ssh_timeout"]
ProvisionResult: TypeAlias = (
    tuple[Literal[True], GPUInstance, None] |
    tuple[Literal[False], None, ProvisionError]
)


def normalize_save_dir(save_dir: Path | str) -> str:
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if save_dir.is_absolute():
        try:
            save_dir = save_dir.relative_to(SCRIPT_DIR)
        except ValueError as exc:
            raise AssertionError(f"save_dir must reside inside project: {save_dir}") from exc
    posix_path = PurePosixPath(save_dir)
    parts = [p for p in posix_path.parts if p not in (".", "..")]
    normalized = "/".join(parts) if parts else ""
    assert normalized, f"save_dir normalized to empty string from: {save_dir}"
    assert not normalized.startswith("/"), f"save_dir should be relative, got: {normalized}"
    return normalized


def load_config_from_file(config_path: str) -> Config:
    spec_path = SCRIPT_DIR / config_path
    if not spec_path.exists():
        raise FileNotFoundError(f"Config not found: {spec_path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp_config", spec_path)
    assert spec and spec.loader, f"Failed to load spec from {spec_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "config", None)
    if not isinstance(config, Config):
        raise TypeError(f"Config file must define Config instance, got {type(config)}")
    return config


def get_credentials(provider_filter: Optional[str] = None) -> dict[str, str]:
    credentials: dict[str, str] = {}
    if runpod_key := get_runpod_key():
        credentials["runpod"] = runpod_key
    if prime_key := get_prime_key():
        credentials["primeintellect"] = prime_key
    if provider_filter:
        if provider_filter not in credentials:
            raise ValueError(f"Provider '{provider_filter}' not found. Set {provider_filter.upper()}_API_KEY")
        return {provider_filter: credentials[provider_filter]}
    assert credentials, "No API keys found - set RUNPOD_API_KEY or PRIME_API_KEY"
    return credentials


def search_cheapest_gpus(gpu_client: GPUClient, max_offers: int = 5):
    offers = gpu_client.search(
        query=gpu_client.cloud_type == CloudType.COMMUNITY,
        sort=lambda x: x.price_per_hour,
        reverse=False,
    )
    assert offers, "No GPU offers found"
    return offers[:max_offers]


def provision_instance(gpu_client: GPUClient, offers, instance_name: str, *, min_vram: int | None = None, min_cpu_ram: int | None = None, max_price: float | None = None) -> ProvisionResult:
    filtered = []
    for offer in offers:
        if min_vram and offer.vram_gb < min_vram:
            continue
        if min_cpu_ram and offer.memory_gb < min_cpu_ram:
            continue
        if max_price and offer.price_per_hour > max_price:
            continue
        filtered.append(offer)
    offers_to_use = filtered or offers

    instance = gpu_client.create(
        offers_to_use,
        image="runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
        name=instance_name,
        n_offers=len(offers_to_use),
    )
    if not instance:
        logging.error("Failed to create instance")
        return (False, None, "create_failed")

    if not instance.wait_until_ready(timeout=300):
        logging.error("Instance failed to become ready, terminating...")
        gpu_client.terminate_instance(instance.id, instance.provider)
        return (False, None, "ready_timeout")

    if not instance.wait_until_ssh_ready(timeout=900):
        logging.error("SSH failed to become ready, terminating...")
        gpu_client.terminate_instance(instance.id, instance.provider)
        return (False, None, "ssh_timeout")

    return (True, instance, None)


def find_instance_by_name_or_id(gpu_client: GPUClient, identifier: str) -> Optional[GPUInstance]:
    instances = gpu_client.list_instances()
    for instance in instances:
        if instance.name == identifier or instance.id == identifier:
            return instance
    return None


def deploy_code(bifrost_client: BifrostClient, *, use_existing: bool) -> str:
    bootstrap_cmd = [
        """if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\"$HOME/.cargo/bin:$PATH\"
fi""",
        "uv sync --extra example-corpus-proximity",
    ]
    workspace_path = bifrost_client.push(bootstrap_cmd=bootstrap_cmd if not use_existing else None)
    return workspace_path or "~/.bifrost/workspace"


def start_remote_pipeline(bifrost_client: BifrostClient, config_arg: str) -> None:
    setup_cmd = f"cd {REMOTE_WORKSPACE_PATH} && rm -f .pipeline_complete .pipeline_failed pipeline.log"
    bifrost_client.exec(setup_cmd)
    tmux_cleanup = f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null || true"
    bifrost_client.exec(f"cd {REMOTE_WORKSPACE_PATH} && {tmux_cleanup}")
    run_cmd = (
        f"cd {REMOTE_WORKSPACE_PATH} && "
        f"tmux new-session -d -s {TMUX_SESSION} "
        f"'uv run python run_full_pipeline.py {config_arg}'"
    )
    result = bifrost_client.exec(run_cmd)
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to start remote pipeline: {result.stderr}")


def wait_for_pipeline_completion(bifrost_client: BifrostClient, timeout: int = 10800) -> bool:
    poll_interval = 30
    max_iterations = max(1, timeout // poll_interval)
    logging.info("⏳ Waiting for pipeline completion (timeout: %ss)", timeout)

    # Define pipeline steps for progress reporting
    steps = [
        ("prepare_data.py", "📥 Downloading data"),
        ("embed_chunks.py", "🧮 Generating embeddings"),
        ("cluster_corpus.py", "🗂️  Clustering corpus"),
        ("name_clusters.py", "🏷️  Naming clusters"),
    ]

    check_cmd = f"""
cd {REMOTE_WORKSPACE_PATH}
if [ -f .pipeline_complete ]; then
  echo COMPLETE
elif [ -f .pipeline_failed ]; then
  echo FAILED
else
  echo RUNNING
fi
"""

    # Command to get current step from log
    progress_cmd = f"""
cd {REMOTE_WORKSPACE_PATH}
if [ -f pipeline.log ]; then
  tail -20 pipeline.log | grep -E '(prepare_data|embed_chunks|cluster_corpus|name_clusters)' | tail -1
else
  echo ""
fi
"""

    for i in range(max_iterations):
        result = bifrost_client.exec(check_cmd)
        status = (result.stdout or "RUNNING").strip().splitlines()[-1]
        if status == "COMPLETE":
            logging.info("✅ Remote pipeline complete")
            return True
        if status == "FAILED":
            logging.error("❌ Remote pipeline reported failure")
            return False

        # Get current step from logs
        progress_result = bifrost_client.exec(progress_cmd)
        current_step = "Unknown"
        step_emoji = "⏳"

        if progress_result.stdout:
            log_line = progress_result.stdout.strip()
            for step_name, step_desc in steps:
                if step_name in log_line:
                    current_step = step_desc
                    step_emoji = step_desc.split()[0]
                    break

        elapsed = (i + 1) * poll_interval
        logging.info("%s Pipeline running: %s (%ss / %ss)", step_emoji, current_step, elapsed, timeout)
        time.sleep(poll_interval)
    logging.error("❌ Pipeline timed out after %ss", timeout)
    return False


def sync_results(bifrost_client: BifrostClient, config: Config, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_save_dir = normalize_save_dir(config.clustering.cache_dir)
    cache_key = get_cache_key(config)
    remote_prefix = f"{REMOTE_WORKSPACE_PATH}/{remote_save_dir}/{cache_key}"

    targets = [
        (f"{remote_prefix}/tree.json", output_dir / "tree.json"),
        (f"{remote_prefix}/stats.json", output_dir / "stats.json"),
        (f"{remote_prefix}/config.json", output_dir / "config.json"),
        (f"{remote_prefix}/chunk_to_cluster.json", output_dir / "chunk_to_cluster.json"),
    ]

    for remote_path, local_path in targets:
        try:
            result = bifrost_client.download_files(remote_path=remote_path, local_path=str(local_path))
            if not (result and result.success and result.files_copied > 0):
                logging.warning("⚠️  Failed to sync %s", remote_path)
            else:
                logging.info("✅ Synced %s", local_path.name)
        except Exception as exc:
            logging.warning("⚠️  Error syncing %s: %s", remote_path, exc)

    try:
        result = bifrost_client.download_files(
            remote_path=f"{REMOTE_WORKSPACE_PATH}/pipeline.log",
            local_path=str(output_dir / "pipeline.log"),
        )
        if result and result.success and result.files_copied > 0:
            logging.info("✅ Synced pipeline.log")
    except Exception as exc:
        logging.warning("⚠️  Could not sync pipeline log: %s", exc)


def cleanup_instance(instance_id: str) -> None:
    import subprocess

    logging.info("🧹 Terminating GPU instance %s", instance_id)
    try:
        result = subprocess.run(["broker", "terminate", instance_id, "--yes"], text=True, capture_output=True)
    except Exception as exc:
        logging.warning("⚠️  Cleanup error: %s", exc)
        logging.info("Manual cleanup: broker terminate %s", instance_id)
        return

    if result.returncode == 0:
        logging.info("✅ Cleanup complete")
    else:
        logging.warning("⚠️  Cleanup may have failed: %s", result.stderr)


def connect_existing_instance(identifier: str, provider: Optional[str], ssh_key_path: str) -> tuple[BifrostClient, Optional[GPUInstance]]:
    credentials = get_credentials(provider_filter=provider) if provider else get_credentials()
    gpu_client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)
    instance = find_instance_by_name_or_id(gpu_client, identifier)
    if not instance:
        raise ValueError(f"Instance not found (name or ID): {identifier}")
    ssh_connection = instance.ssh_connection_string()
    bifrost_client = BifrostClient(ssh_connection=ssh_connection, ssh_key_path=ssh_key_path)
    return bifrost_client, instance


def provision_new_instance(provider: Optional[str], instance_name: str, config: Config) -> tuple[BifrostClient, GPUInstance]:
    credentials = get_credentials(provider_filter=provider)
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")
    gpu_client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)
    offers = search_cheapest_gpus(gpu_client)
    success, instance, error = provision_instance(
        gpu_client,
        offers,
        instance_name,
        min_vram=config.deployment.min_vram,
        min_cpu_ram=config.deployment.min_cpu_ram,
        max_price=config.deployment.max_price,
    )
    if not success or not instance:
        raise RuntimeError(f"Provisioning failed: {error}")
    bifrost_client = BifrostClient(ssh_connection=instance.ssh_connection_string(), ssh_key_path=ssh_key_path)
    return bifrost_client, instance


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy corpus-proximity to GPU")
    parser.add_argument("--provider", type=str, choices=["runpod", "primeintellect"], help="GPU provider")
    parser.add_argument("--use-existing", type=str, metavar="NAME_OR_SSH", help="Existing instance name or SSH")
    parser.add_argument("--name", type=str, help="Custom instance name")
    parser.add_argument("--config", type=str, required=True, help="Path to config file relative to corpus-proximity directory")
    parser.add_argument("--keep-running", action="store_true", help="Keep GPU running after completion")
    args = parser.parse_args()

    setup_logging()
    load_dotenv()

    try:
        config = load_config_from_file(args.config)
    except Exception as exc:
        logging.error("Failed to load config: %s", exc)
        return 1

    if args.keep_running:
        config.deployment.keep_running = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_results = SCRIPT_DIR / "remote_results" / f"clustering_{timestamp}"

    bifrost_client: Optional[BifrostClient] = None
    instance: Optional[GPUInstance] = None
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    try:
        if args.use_existing:
            if "@" in args.use_existing and ":" in args.use_existing:
                bifrost_client = BifrostClient(ssh_connection=args.use_existing, ssh_key_path=ssh_key_path)
            else:
                bifrost_client, instance = connect_existing_instance(args.use_existing, args.provider, ssh_key_path)
        else:
            instance_name = args.name or "corpus-proximity-dev"
            bifrost_client, instance = provision_new_instance(args.provider, instance_name, config)

        workspace_path = deploy_code(bifrost_client, use_existing=bool(args.use_existing))
        logging.info("✅ Code deployed to %s", workspace_path)

        start_remote_pipeline(bifrost_client, args.config)
        success = wait_for_pipeline_completion(bifrost_client)

        logging.info("💾 Syncing results to %s", local_results)
        sync_results(bifrost_client, config, local_results)

        if success:
            logging.info("🎉 Deployment complete")
        else:
            logging.warning("⚠️  Pipeline did not complete successfully; review logs")

        keep_running = config.deployment.keep_running
        if keep_running:
            logging.info("GPU left running per configuration")
            if instance:
                logging.info("Instance: %s", instance.ssh_connection_string())
        elif instance:
            cleanup_instance(instance.id)

        return 0 if success else 1

    except KeyboardInterrupt:
        logging.error("Interrupted")
        if instance and not config.deployment.keep_running:
            cleanup_instance(instance.id)
        return 1
    except Exception as exc:
        logging.error("Deployment failed: %s", exc)
        if instance and not config.deployment.keep_running:
            cleanup_instance(instance.id)
        return 1


if __name__ == "__main__":
    sys.exit(main())
