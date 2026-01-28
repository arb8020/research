#!/usr/bin/env python3
"""Demo: Provision RunPod GPU + deploy SGLang server.

Usage:
    # Using a recipe file
    python examples/provision_and_serve.py --recipe recipes/examples/qwen3_0.6b_4090.py

    # Override recipe with CLI args
    python examples/provision_and_serve.py --recipe recipes/examples/qwen3_0.6b_4090.py --ssh root@gpu:22

    # Without recipe (manual args)
    python examples/provision_and_serve.py --model Qwen/Qwen3-0.6B --gpu-type RTX4090

    # Use existing instance
    python examples/provision_and_serve.py --node-id runpod:abc123 --model Qwen/Qwen3-0.6B

Returns the server URL that can be used with query_server.py.

Requirements:
    - RUNPOD_API_KEY environment variable
    - HF_TOKEN environment variable (for gated models)
    - SSH key at ~/.ssh/id_ed25519
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recipes.schema import ServingRecipe

# Add repo root to path for local development
# examples/ is one level down from repo root
repo_root = Path(__file__).parent.parent
# Add subpackages to path (each has nested structure like rollouts/rollouts/)
for subpkg in ["rollouts", "bifrost", "broker", "shared"]:
    pkg_path = repo_root / subpkg
    if pkg_path.exists():
        sys.path.insert(0, str(pkg_path))

# Add repo root for recipes module
sys.path.insert(0, str(repo_root))

# Load .env file if present
env_file = repo_root / ".env"
if env_file.exists():
    import os

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import trio


def load_recipe(recipe_path: str) -> ServingRecipe:
    """Load recipe from .py or .json file."""
    import importlib.util

    from recipes.schema import ServingRecipe

    path = Path(recipe_path)
    if not path.exists():
        # Try relative to repo root
        path = repo_root / recipe_path
    assert path.exists(), f"Recipe not found: {recipe_path}"

    if path.suffix == ".json":
        return ServingRecipe.from_json(path)
    elif path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("recipe_module", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "recipe"), "Recipe file must define 'recipe' variable"
        return module.recipe
    else:
        raise ValueError(f"Recipe must be .py or .json, got: {path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision GPU and deploy SGLang server")

    # Recipe file (optional - provides defaults)
    parser.add_argument(
        "--recipe",
        type=str,
        help="Path to recipe file (.py or .json)",
    )

    # Node acquisition (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ssh",
        type=str,
        help="Static SSH connection string (user@host:port)",
    )
    group.add_argument(
        "--node-id",
        type=str,
        help="Existing broker instance ID (provider:instance_id)",
    )
    # Default: provision new instance

    # GPU configuration (overrides recipe)
    parser.add_argument(
        "--gpu-type",
        type=str,
        help="GPU type to provision (default from recipe or A100)",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        help="Number of GPUs (default from recipe or 1)",
    )

    # Model configuration (overrides recipe)
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model ID (default from recipe)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (default from recipe or 30000)",
    )

    # Cloud type
    parser.add_argument(
        "--community",
        action="store_true",
        help="Use community cloud (cheaper but less reliable)",
    )

    # Account selection
    parser.add_argument(
        "--wafer",
        action="store_true",
        help="Use wafer RunPod account (WAFER_RUNPOD_API_KEY)",
    )

    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    # Import here to get better error messages if deps missing
    try:
        from bifrost import GPUQuery, acquire_node
        from rollouts.deploy import ServerConfig, deploy_sglang_server
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install -e rollouts[deploy] bifrost broker")
        return 1

    # Load recipe if provided
    recipe = None
    if args.recipe:
        print(f"Loading recipe: {args.recipe}")
        recipe = load_recipe(args.recipe)
        print(f"  Name: {recipe.name}")
        print(f"  Model: {recipe.model.model}")
        print(f"  Target: {recipe.target.gpu_count}x {recipe.target.gpu_type}")
        print()

    # Resolve config values (CLI overrides recipe)
    model = args.model or (recipe.model.model if recipe else "Qwen/Qwen2.5-7B-Instruct")
    gpu_type = args.gpu_type or (recipe.target.gpu_type if recipe else "A100")
    gpu_count = args.gpu_count or (recipe.target.gpu_count if recipe else 1)
    port = args.port or (recipe.engine.port if recipe else 30000)

    # Step 1: Acquire node
    print("=" * 60)
    print("Step 1: Acquiring GPU node")
    print("=" * 60)

    if args.ssh:
        print(f"Using static SSH: {args.ssh}")
        client, instance = acquire_node(ssh=args.ssh)
        ssh_connection = args.ssh
    elif args.node_id:
        print(f"Using existing instance: {args.node_id}")
        client, instance = acquire_node(node_id=args.node_id)
        assert instance is not None
        ssh_connection = instance.ssh_connection_string()
    else:
        import os

        cloud_type = "community" if args.community else "secure"

        # Use wafer account if requested
        credentials = {}
        if args.wafer:
            wafer_key = os.environ.get("WAFER_RUNPOD_API_KEY")
            if not wafer_key:
                print("Error: --wafer flag requires WAFER_RUNPOD_API_KEY in .env")
                return 1
            credentials["runpod"] = wafer_key
            print("Using wafer RunPod account")

        print(f"Provisioning new instance: {gpu_count}x {gpu_type} ({cloud_type})")
        client, instance = acquire_node(
            provision=GPUQuery(
                type=gpu_type,
                count=gpu_count,
                cloud_type=cloud_type,
                credentials=credentials,
            )
        )
        assert instance is not None
        ssh_connection = instance.ssh_connection_string()
        print(f"Instance ID: {instance.provider}:{instance.id}")

    print(f"SSH connection: {ssh_connection}")
    print()

    # Step 2: Deploy SGLang server
    print("=" * 60)
    print("Step 2: Deploying SGLang server")
    print("=" * 60)

    # Build ServerConfig from recipe + overrides
    server_config_kwargs: dict = {
        "model": model,
        "port": port,
        "ssh_connection": ssh_connection,
        "gpu_ranks": list(range(gpu_count)),
        "tensor_parallel_size": gpu_count,
    }

    # Apply recipe settings if available
    if recipe:
        if recipe.model.quantization:
            server_config_kwargs["quantization"] = recipe.model.quantization
        if recipe.model.max_model_len:
            server_config_kwargs["max_model_len"] = recipe.model.max_model_len
        if recipe.engine.gpu_memory_utilization != 0.9:
            server_config_kwargs["gpu_memory_utilization"] = recipe.engine.gpu_memory_utilization
        if recipe.engine.enable_prefix_caching:
            server_config_kwargs["enable_prefix_caching"] = True
        if recipe.engine.attention_backend:
            server_config_kwargs["attention_backend"] = recipe.engine.attention_backend

    config = ServerConfig(**server_config_kwargs)

    print(f"Model: {config.model}")
    print(f"Port: {config.port}")
    print(f"GPUs: {config.gpu_ranks}")
    print()

    server_info, error = await deploy_sglang_server(config)

    if error:
        print(f"Deployment failed: {error}")
        return 1

    assert server_info is not None

    # Step 3: Print results
    print()
    print("=" * 60)
    print("Deployment successful!")
    print("=" * 60)
    print()
    print(f"Server URL: {server_info.url}")
    print(f"API Base:   {server_info.get_api_base()}")
    print(f"Model:      {server_info.model}")
    print()
    print("To query the server:")
    print(f"  python examples/query_server.py --url {server_info.get_api_base()}")
    print()
    print("To view server logs:")
    print(f"  ssh {ssh_connection} 'tmux attach -t {server_info.tmux_session}'")
    print()
    if instance:
        print("To terminate instance:")
        print(f"  broker terminate {instance.provider}:{instance.id}")

    return 0


if __name__ == "__main__":
    sys.exit(trio.run(main))
