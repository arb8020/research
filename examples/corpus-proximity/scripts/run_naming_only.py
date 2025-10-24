#!/usr/bin/env python3
"""Run only the naming step on existing GPU instance with cached data."""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from broker import GPUClient
from bifrost import BifrostClient
from shared.config import get_runpod_key, get_prime_key
from cluster_corpus import get_cache_key
import importlib.util

REMOTE_WORKSPACE_PATH = "~/.bifrost/workspace/examples/corpus-proximity"

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_naming_only.py <config_path> [--download]")
        print("Example: python run_naming_only.py configs/clustering_01_tiny.py --download")
        return 1

    config_arg = sys.argv[1]
    download_results = "--download" in sys.argv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Check for OPENAI_API_KEY
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        logging.error("❌ OPENAI_API_KEY not found in environment")
        logging.error("   Set it in .env file or export it before running")
        return 1

    logging.info("✅ OPENAI_API_KEY found")

    # Get credentials
    credentials = {}
    if runpod_key := get_runpod_key():
        credentials["runpod"] = runpod_key
    if prime_key := get_prime_key():
        credentials["primeintellect"] = prime_key

    if not credentials:
        logging.error("No API keys found")
        return 1

    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")
    gpu_client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

    # List instances
    instances = gpu_client.list_instances()

    # Find corpus-proximity instance
    corpus_instance = None
    for instance in instances:
        if instance.name and "corpus" in instance.name.lower():
            corpus_instance = instance
            break

    if not corpus_instance:
        logging.error("❌ No corpus-proximity instance found")
        logging.info("Available instances:")
        for inst in instances:
            logging.info(f"  - {inst.name} ({inst.id})")
        return 1

    logging.info(f"✅ Found instance: {corpus_instance.name} ({corpus_instance.id})")
    logging.info(f"   SSH: {corpus_instance.ssh_connection_string()}")

    # Connect with bifrost
    bifrost_client = BifrostClient(
        ssh_connection=corpus_instance.ssh_connection_string(),
        ssh_key_path=ssh_key_path
    )

    # Check what's cached on the remote instance
    logging.info("\n" + "="*80)
    logging.info("Checking cached data on remote instance...")
    logging.info("="*80)

    check_cmd = f"""cd {REMOTE_WORKSPACE_PATH} && \
echo "Cluster tree:" && ls -lh data/clusters_tiny/*/tree.json 2>/dev/null || echo "  ❌ No cluster tree found" && \
echo "" && \
echo "Embeddings:" && ls -lh data/embeddings_arctic_tiny/*/embeddings.npy 2>/dev/null || echo "  ❌ No embeddings found" && \
echo "" && \
echo "Processed data:" && ls -lh data/processed_tiny/*.jsonl 2>/dev/null || echo "  ❌ No processed data found"
"""
    result = bifrost_client.exec(check_cmd)
    if result.stdout:
        print(result.stdout)

    # Run just the naming step with OPENAI_API_KEY
    logging.info("\n" + "="*80)
    logging.info("🏷️  Running naming step with OPENAI_API_KEY...")
    logging.info("="*80 + "\n")

    # Export the API key and run the naming script
    cmd = f"""cd {REMOTE_WORKSPACE_PATH} && \
export OPENAI_API_KEY='{openai_api_key}' && \
uv run python name_clusters.py {config_arg} --name"""

    # Use exec_stream to get real-time output
    result = bifrost_client.exec_stream(cmd)

    # Stream output
    for line in result:
        print(line, end='')

    logging.info("\n" + "="*80)
    logging.info("✅ Naming step completed")
    logging.info("="*80)

    # Download results if requested
    if download_results:
        logging.info("\n📥 Downloading results...")

        # Load config to get cache key
        config_path = Path(config_arg)
        spec = importlib.util.spec_from_file_location("exp_config", config_path)
        if not (spec and spec.loader):
            logging.error(f"Failed to load config from {config_path}")
            return 1

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = getattr(module, "config")
        cache_key = get_cache_key(config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_results = Path(f"remote_results/naming_only_{timestamp}")
        local_results.mkdir(parents=True, exist_ok=True)

        # Download results
        remote_prefix = f"{REMOTE_WORKSPACE_PATH}/data/clusters_tiny/{cache_key}"

        targets = [
            (f"{remote_prefix}/tree.json", local_results / "tree.json"),
            (f"{remote_prefix}/stats.json", local_results / "stats.json"),
            (f"{remote_prefix}/config.json", local_results / "config.json"),
            (f"{remote_prefix}/chunk_to_cluster.json", local_results / "chunk_to_cluster.json"),
        ]

        for remote_path, local_path in targets:
            try:
                result = bifrost_client.download_files(
                    remote_path=remote_path,
                    local_path=str(local_path)
                )
                if result and result.success and result.files_copied > 0:
                    logging.info(f"✅ Downloaded {local_path.name}")
                else:
                    logging.warning(f"⚠️  Failed to download {remote_path}")
            except Exception as e:
                logging.warning(f"⚠️  Error downloading {remote_path}: {e}")

        logging.info(f"\n✅ Results saved to {local_results}")
    else:
        logging.info("\nℹ️  To download results, run with --download flag")

    return 0

if __name__ == "__main__":
    sys.exit(main())
