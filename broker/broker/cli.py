"""Broker CLI - GPU provisioning and management"""

import builtins
import json
import logging
import os
from pathlib import Path

import trio
import typer
from infra_utils.config import (
    discover_ssh_keys,
    get_ssh_key_path,
)
from infra_utils.logging_config import setup_logging
from rich.console import Console
from rich.table import Table

from broker.client import GPUClient
from broker.credentials import (
    CREDENTIALS_FILE,
    KNOWN_PROVIDERS,
    key_preview,
    load_profiles,
    set_active_profile,
    set_profile_key,
)
from broker.types import ProviderCredentials

console = Console()
app = typer.Typer(help="GPU broker - provision cloud GPUs")
auth_app = typer.Typer(help="Manage provider credentials (~/.broker/credentials.toml)")
volumes_app = typer.Typer(
    help="Manage RunPod network volumes", invoke_without_command=True, no_args_is_help=False
)
app.add_typer(auth_app, name="auth")
app.add_typer(volumes_app, name="volumes")

# Logger will be configured in callback
logger = logging.getLogger("broker")


@app.callback()
def main(
    ctx: typer.Context,
    credentials: str | None = typer.Option(
        None,
        "--credentials",
        help="Credentials file or inline 'runpod:key,primeintellect:key'",
    ),
    ssh_key: str | None = typer.Option(None, "--ssh-key", help="Path to SSH private key"),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Show only warnings and errors"),
    json_output: bool = typer.Option(
        True, "--json/--rich", help="Output format (JSON default, --rich for tables)"
    ),
    debug: bool = typer.Option(False, "--debug", help="Show debug logs"),
) -> None:
    """Configure logging and store global options"""

    # Setup logging based on flags
    if json_output:
        # Suppress all logging when outputting JSON
        setup_logging(level="CRITICAL", use_rich=False, use_json=False)
    elif debug:
        setup_logging(level="DEBUG", use_rich=True, rich_tracebacks=True)
    elif quiet:
        setup_logging(level="WARNING", use_rich=True)
    else:
        # Default: INFO level
        setup_logging(level="INFO", use_rich=True)

    # Store options in context
    ctx.obj = {"credentials": credentials, "ssh_key": ssh_key, "json": json_output}


def resolve_credentials(ctx) -> ProviderCredentials:
    """Resolve credentials from CLI flag → env vars → ~/.broker/credentials.toml → error.

    Priority:
    1. --credentials flag (file path or inline format)
    2. Environment variables (RUNPOD_API_KEY, etc.) + ~/.broker/credentials.toml active profile
    3. Error with helpful message
    """
    from broker.credentials import get_credentials

    creds_arg = ctx.obj.get("credentials")

    # Priority 1: CLI flag
    if creds_arg:
        if Path(creds_arg).exists():
            with open(creds_arg) as f:
                creds_dict = json.load(f)
            return ProviderCredentials.from_dict(creds_dict)
        else:
            # Inline format: "runpod:key,primeintellect:key"
            parts = creds_arg.split(",")
            creds_dict = {}
            for part in parts:
                if ":" not in part:
                    logger.error(f"Invalid credentials format: {part}")
                    logger.info("Expected: runpod:key,primeintellect:key")
                    raise typer.Exit(1)
                provider, key = part.split(":", 1)
                creds_dict[provider.strip()] = key.strip()
            return ProviderCredentials.from_dict(creds_dict)

    # Priority 2: env vars + credentials.toml
    creds = get_credentials()
    if creds:
        return ProviderCredentials.from_dict(creds)

    # Priority 3: Error
    logger.error("No credentials found")
    logger.info("")
    logger.info("Run: broker auth login <provider>")
    logger.info("  e.g. broker auth login runpod")
    logger.info("")
    logger.info("Or set environment variables:")
    logger.info("  export RUNPOD_API_KEY=...")
    raise typer.Exit(1)


def resolve_ssh_key(ctx) -> str:
    """Resolve SSH key: CLI → env → discover → error

    Priority:
    1. --ssh-key flag
    2. SSH_KEY_PATH environment variable
    3. Discovery with helpful error showing found keys
    """
    ssh_key_arg = ctx.obj.get("ssh_key")

    # Priority 1: CLI flag
    if ssh_key_arg:
        return ssh_key_arg

    # Priority 2: Environment
    if key_path := get_ssh_key_path():
        return key_path

    # Priority 3: Discovery with helpful error
    found_keys = discover_ssh_keys()

    logger.error("✗ No SSH key specified")
    logger.info("")
    if found_keys:
        logger.info("found keys at:")
        for key in found_keys:
            logger.info(f"  {key}")
        logger.info("")
        logger.info(f"use: --ssh-key {found_keys[0]}")
        logger.info("or: export SSH_KEY_PATH=~/.ssh/id_ed25519")
    else:
        logger.info("no ssh keys found in ~/.ssh/")
        logger.info("generate one: ssh-keygen -t ed25519")

    raise typer.Exit(1)


def parse_instance_id(instance_id: str) -> tuple[str, str | None]:
    """Parse instance ID, supporting optional provider prefix.

    Accepts:
        "abc123" -> ("abc123", None)
        "runpod:abc123" -> ("abc123", "runpod")

    Returns:
        (instance_id, provider) tuple
    """
    if ":" in instance_id:
        provider, id_part = instance_id.split(":", 1)
        return (id_part, provider)
    return (instance_id, None)


@app.command()
def search(  # noqa: PLR0913 - CLI search has many filter options
    ctx: typer.Context,
    gpu_type: str | None = typer.Option(
        None, "--gpu-type", help="Filter by GPU type (e.g., 'A100')"
    ),
    gpu_count: int = typer.Option(
        1, "--gpu-count", help="Number of GPUs (affects pricing, default: 1)"
    ),
    max_price_per_gpu: float | None = typer.Option(
        None, "--max-price-per-gpu", help="Maximum price per GPU per hour"
    ),
    max_total_price: float | None = typer.Option(
        None, "--max-total-price", help="Maximum total price per hour"
    ),
    max_price: float | None = typer.Option(
        None, "--max-price", help="(Deprecated) Use --max-price-per-gpu instead"
    ),
    min_vram: int | None = typer.Option(None, "--min-vram", help="Minimum VRAM in GB"),
    provider: str | None = typer.Option(
        None, "--provider", help="Filter by provider (runpod|primeintellect)"
    ),
    cloud_type: str | None = typer.Option(
        "secure",
        "--cloud-type",
        help="Cloud type: secure (default, guaranteed) or community (spot, cheaper but can be interrupted)",
    ),
    underlying_provider: str | None = typer.Option(
        None,
        "--underlying-provider",
        help="Filter by underlying provider (e.g., massedcompute, hyperstack) for aggregators like PrimeIntellect",
    ),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results"),
) -> None:
    """Search for available GPU offers

    By default searches all configured providers and merges results.
    Use --provider to filter to specific provider.
    """

    async def _search_async() -> None:
        creds = resolve_credentials(ctx)

        # Create client (SSH key not needed for search)
        client = GPUClient(credentials=creds, ssh_key_path=None)

        # Handle pricing flags
        effective_max_price = None
        if max_price_per_gpu is not None:
            effective_max_price = max_price_per_gpu
        elif max_price is not None:
            logger.warning("--max-price is deprecated, use --max-price-per-gpu instead")
            effective_max_price = max_price

        if max_total_price is not None:
            max_per_gpu_from_total = max_total_price / gpu_count
            if effective_max_price is not None:
                # Use the more restrictive constraint
                effective_max_price = min(effective_max_price, max_per_gpu_from_total)
            else:
                effective_max_price = max_per_gpu_from_total

        # Build query
        query = None
        if gpu_type:
            query = client.gpu_type.contains(gpu_type)
        if effective_max_price:
            price_filter = client.price_per_hour <= effective_max_price
            query = price_filter if query is None else query & price_filter
        if min_vram:
            vram_filter = client.vram_gb >= min_vram
            query = vram_filter if query is None else query & vram_filter
        if provider:
            provider_filter = client.provider == provider
            query = provider_filter if query is None else query & provider_filter
        if cloud_type:
            from broker.types import CloudType

            if cloud_type.lower() == "secure":
                cloud_filter = client.cloud_type == CloudType.SECURE
            elif cloud_type.lower() == "community":
                cloud_filter = client.cloud_type == CloudType.COMMUNITY
            else:
                logger.error(f"Invalid cloud type: {cloud_type}. Use 'secure' or 'community'")
                raise typer.Exit(1)
            query = cloud_filter if query is None else query & cloud_filter
        if underlying_provider:
            underlying_filter = client.underlying_provider == underlying_provider
            query = underlying_filter if query is None else query & underlying_filter

        # Search (queries all providers by default, merges results)
        if gpu_count > 1:
            logger.info(f"searching for gpu offers (gpu_count={gpu_count})...")
        else:
            logger.info("searching for gpu offers...")

        # Import api to call search with gpu_count
        from broker import api

        offers = await api.search(
            query, gpu_count=gpu_count, sort=lambda x: x.price_per_hour, credentials=creds.to_dict()
        )

        # Limit results
        offers = offers[:limit]

        # Output
        if ctx.obj["json"]:
            print(
                json.dumps(
                    [
                        {
                            "provider": o.provider,
                            "gpu_type": o.gpu_type,
                            "vram_gb": o.vram_gb,
                            "price_per_hour": o.price_per_hour,
                            "total_price_per_hour": o.price_per_hour * gpu_count,
                            "memory_gb": o.memory_gb,
                            "gpu_count": gpu_count,
                        }
                        for o in offers
                    ],
                    indent=2,
                )
            )
        else:
            # Rich table output
            if gpu_count > 1:
                table = Table(title=f"GPU Node Offers - {gpu_count}x GPUs ({len(offers)} found)")
                table.add_column("Provider", style="cyan")
                table.add_column("GPU Type")
                table.add_column("Cloud", justify="center")
                table.add_column("VRAM", justify="right")
                table.add_column("Node Price/hr", justify="right", style="bold")
                table.add_column("Per GPU", justify="right", style="dim")

                for offer in offers:
                    from broker.types import CloudType

                    # Use offer.gpu_count (which reflects the actual search result) rather than parameter gpu_count
                    total_price = offer.price_per_hour * offer.gpu_count

                    # Format cloud type display
                    if offer.cloud_type == CloudType.SECURE:
                        cloud_display = "[green]Secure[/green]"
                    elif offer.cloud_type == CloudType.COMMUNITY:
                        cloud_display = "[yellow]Community[/yellow]"
                    else:
                        cloud_display = "Unknown"

                    table.add_row(
                        offer.provider,
                        offer.gpu_type,
                        cloud_display,
                        f"{offer.vram_gb}GB",
                        f"${total_price:.2f}",
                        f"${offer.price_per_hour:.2f}",
                    )
            else:
                table = Table(title=f"GPU Offers ({len(offers)} found)")
                table.add_column("Provider", style="cyan")
                table.add_column("GPU Type")
                table.add_column("VRAM", justify="right")
                table.add_column("RAM", justify="right")
                table.add_column("Price/hr", justify="right")

                for offer in offers:
                    table.add_row(
                        offer.provider,
                        offer.gpu_type,
                        f"{offer.vram_gb}GB",
                        f"{offer.memory_gb}GB",
                        f"${offer.price_per_hour:.2f}",
                    )

            console.print(table)
            if gpu_count > 1:
                logger.info(f"showing {len(offers)} cheapest offers for {gpu_count}x gpus")
            else:
                logger.info(f"showing {len(offers)} cheapest offers")

    trio.run(_search_async)


@app.command()
def create(  # noqa: PLR0913 - CLI create has many configuration options
    ctx: typer.Context,
    gpu_type: str | None = typer.Option(None, "--gpu-type", help="GPU type filter"),
    gpu_count: int = typer.Option(1, "--gpu-count", help="Number of GPUs (default: 1)"),
    max_price_per_gpu: float | None = typer.Option(
        None, "--max-price-per-gpu", help="Maximum price per GPU per hour"
    ),
    max_total_price: float | None = typer.Option(
        None, "--max-total-price", help="Maximum total price per hour"
    ),
    max_price: float | None = typer.Option(
        None, "--max-price", help="(Deprecated) Use --max-price-per-gpu instead"
    ),
    cloud_type: str | None = typer.Option(
        "secure",
        "--cloud-type",
        help="Cloud type: secure (default, guaranteed) or community (spot, cheaper but can be interrupted)",
    ),
    image: str = typer.Option(
        "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "--image",
        help="Docker image to use",
    ),
    min_cuda_version: str | None = typer.Option(
        None,
        "--min-cuda-version",
        help="Minimum CUDA version required (e.g., '12.1', '12.8'). Ensures node has compatible driver.",
    ),
    network_volume_id: str | None = typer.Option(
        None,
        "--network-volume-id",
        help="RunPod network volume ID to attach. Volume must be in datacenter specified by --datacenter-id.",
    ),
    datacenter_id: str | None = typer.Option(
        None,
        "--datacenter-id",
        help="RunPod datacenter ID to provision in (required when --network-volume-id is set). Use 'broker volumes' to list volumes and their datacenters.",
    ),
    name: str | None = typer.Option(None, "--name", help="Instance name"),
    wait_ssh: bool = typer.Option(
        False, "--wait-ssh", help="Wait for SSH to be ready before returning"
    ),
    output: str = typer.Option("summary", "--output", help="Output format: summary|ssh|json"),
) -> None:
    """Provision a new GPU instance

    By default returns immediately after provisioning starts.
    Use --wait-ssh to block until SSH is ready.

    Output formats:
      summary: Human-readable status (default)
      ssh: Just SSH connection string (for piping to bifrost)
      json: Full instance details as JSON
    """

    async def _create_async() -> None:
        creds = resolve_credentials(ctx)
        ssh_key = resolve_ssh_key(ctx)

        # Create client
        client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

        # Handle pricing flags
        effective_max_price = None
        if max_price_per_gpu is not None:
            effective_max_price = max_price_per_gpu
        elif max_price is not None:
            logger.warning("--max-price is deprecated, use --max-price-per-gpu instead")
            effective_max_price = max_price

        if max_total_price is not None:
            max_per_gpu_from_total = max_total_price / gpu_count
            if effective_max_price is not None:
                # Use the more restrictive constraint
                effective_max_price = min(effective_max_price, max_per_gpu_from_total)
            else:
                effective_max_price = max_per_gpu_from_total

        # Build query if filters provided
        query = None
        if gpu_type:
            query = client.gpu_type.contains(gpu_type)
        if effective_max_price:
            price_filter = client.price_per_hour <= effective_max_price
            query = price_filter if query is None else query & price_filter
        if cloud_type:
            from broker.types import CloudType

            if cloud_type.lower() == "secure":
                cloud_filter = client.cloud_type == CloudType.SECURE
            elif cloud_type.lower() == "community":
                cloud_filter = client.cloud_type == CloudType.COMMUNITY
            else:
                logger.error(f"Invalid cloud type: {cloud_type}. Use 'secure' or 'community'")
                raise typer.Exit(1)
            query = cloud_filter if query is None else query & cloud_filter

        # Create instance
        if gpu_count > 1:
            cloud_msg = f" ({cloud_type} cloud)" if cloud_type else ""
            logger.info(f"provisioning {gpu_count}x gpu instance{cloud_msg}...")
        else:
            cloud_msg = f" ({cloud_type} cloud)" if cloud_type else ""
            logger.info(f"provisioning instance{cloud_msg}...")
        instance = await client.create(
            query,
            image=image,
            name=name,
            gpu_count=gpu_count,
            min_cuda_version=min_cuda_version,
            network_volume_id=network_volume_id,
            datacenter_id=datacenter_id,
        )

        if not instance:
            logger.error("✗ Failed to provision instance")
            raise typer.Exit(1)

        # Wait for SSH if requested
        if wait_ssh:
            logger.debug("waiting for ssh to be ready...")
            if not await instance.wait_until_ssh_ready(timeout=300):
                logger.error("✗ SSH failed to become ready")
                raise typer.Exit(1)

        # Output based on format
        if output == "json":
            print(
                json.dumps(
                    {
                        "id": instance.id,
                        "provider": instance.provider,
                        "gpu_type": instance.gpu_type,
                        "gpu_count": instance.gpu_count,
                        "status": instance.status.value,
                        "ssh": instance.ssh_connection_string() if wait_ssh else None,
                        "price_per_hour": instance.price_per_hour,
                        "total_price_per_hour": instance.price_per_hour * instance.gpu_count,
                    },
                    indent=2,
                )
            )
        elif output == "ssh":
            if wait_ssh:
                print(instance.ssh_connection_string())
            else:
                logger.error("✗ Cannot output SSH - use --wait-ssh")
                raise typer.Exit(1)
        else:  # summary
            if wait_ssh:
                gpu_info = (
                    f"{instance.gpu_count}x {instance.gpu_type}"
                    if instance.gpu_count > 1
                    else instance.gpu_type
                )
                logger.info(f"instance {instance.id} ready: {instance.ssh_connection_string()}")
                logger.info(f"  gpu: {gpu_info}")
                if instance.gpu_count > 1:
                    logger.info(
                        f"  price: ${instance.price_per_hour:.2f}/gpu/hr (${instance.price_per_hour * instance.gpu_count:.2f}/hr total)"
                    )
                else:
                    logger.info(f"  price: ${instance.price_per_hour:.2f}/hr")
            else:
                gpu_info = (
                    f"{instance.gpu_count}x {instance.gpu_type}"
                    if instance.gpu_count > 1
                    else instance.gpu_type
                )
                logger.info(f"instance {instance.id} provisioning started")
                logger.info(f"  gpu: {gpu_info}")
                if instance.gpu_count > 1:
                    logger.info(
                        f"  price: ${instance.price_per_hour:.2f}/gpu/hr (${instance.price_per_hour * instance.gpu_count:.2f}/hr total)"
                    )
                else:
                    logger.info(f"  price: ${instance.price_per_hour:.2f}/hr")
                logger.info(f"use 'broker status {instance.id}' to check progress")

    trio.run(_create_async)


@app.command(name="list")
def list_instances(ctx: typer.Context) -> None:
    """List all your GPU instances"""

    async def _list_async() -> None:
        creds = resolve_credentials(ctx)
        ssh_key = resolve_ssh_key(ctx)

        client = GPUClient(credentials=creds, ssh_key_path=ssh_key)
        instances = await client.list_instances()

        if ctx.obj["json"]:
            print(
                json.dumps(
                    [
                        {
                            "id": i.id,
                            "gpu_type": i.gpu_type,
                            "price_per_hour": i.price_per_hour,
                            "status": i.status.value,
                            "provider": i.provider,
                        }
                        for i in instances
                    ],
                    indent=2,
                )
            )
        else:
            if not instances:
                logger.info("No instances found")
                return

            table = Table(title=f"GPU Instances ({len(instances)})")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name")
            table.add_column("Provider")
            table.add_column("GPUs")
            table.add_column("Status")
            table.add_column("Node Price/hr", justify="right")

            for instance in instances:
                # Show GPU count and type
                if instance.gpu_count > 1:
                    gpu_display = f"{instance.gpu_count}x {instance.gpu_type}"
                else:
                    gpu_display = instance.gpu_type

                # Calculate node price (per-GPU price × count)
                node_price = instance.price_per_hour * instance.gpu_count

                table.add_row(
                    instance.id,
                    instance.name or "-",
                    instance.provider,
                    gpu_display,
                    instance.status.value,
                    f"${node_price:.2f}",
                )

            console.print(table)

    trio.run(_list_async)


@app.command()
def status(
    ctx: typer.Context,
    instance_id_arg: str = typer.Argument(..., metavar="INSTANCE_ID", help="Instance ID"),
    provider_arg: str | None = typer.Argument(
        None, help="Provider (runpod|primeintellect). Auto-detect if omitted."
    ),
) -> None:
    """Get instance status

    Instance ID can include provider prefix: runpod:abc123 or just abc123.
    Provider can be omitted for convenience. If omitted, will search all
    providers. Errors if instance ID exists in multiple providers.
    """

    async def _status_async() -> None:
        creds = resolve_credentials(ctx)
        ssh_key = resolve_ssh_key(ctx)

        client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

        # Parse provider:id format if present
        instance_id, parsed_provider = parse_instance_id(instance_id_arg)
        provider = provider_arg or parsed_provider

        # Auto-detect provider if not specified
        if provider is None:
            instances = await client.list_instances()
            matches = [i for i in instances if i.id == instance_id]

            if len(matches) == 0:
                logger.error(f"✗ Instance {instance_id} not found in any provider")
                raise typer.Exit(1)
            if len(matches) > 1:
                logger.error(f"✗ Instance {instance_id} found in multiple providers:")
                for m in matches:
                    logger.error(f"  - {m.provider}")
                logger.info(f"specify provider: broker status {instance_id} <provider>")
                raise typer.Exit(1)

            instance = matches[0]
        else:
            instance = await client.get_instance(instance_id, provider)

            if not instance:
                logger.error(f"✗ Instance {instance_id} not found in {provider}")
                raise typer.Exit(1)

        if ctx.obj["json"]:
            print(
                json.dumps(
                    {
                        "id": instance.id,
                        "provider": instance.provider,
                        "status": instance.status.value,
                        "gpu_type": instance.gpu_type,
                        "ssh": (instance.ssh_connection_string() if instance.public_ip else None),
                    },
                    indent=2,
                )
            )
        else:
            logger.info(f"instance: {instance.id}")
            logger.info(f"provider: {instance.provider}")
            logger.info(f"status: {instance.status.value}")
            logger.info(f"gpu: {instance.gpu_type}")
            if instance.public_ip:
                logger.info(f"ssh: {instance.ssh_connection_string()}")

    trio.run(_status_async)


@app.command()
def ssh(
    ctx: typer.Context,
    instance_id_arg: str = typer.Argument(..., metavar="INSTANCE_ID"),
    provider_arg: str | None = typer.Argument(
        None, help="Provider (runpod|primeintellect). Auto-detect if omitted."
    ),
) -> None:
    """Get SSH connection string for instance

    Instance ID can include provider prefix: runpod:abc123 or just abc123.
    Provider can be omitted for convenience. If omitted, will search all
    providers. Errors if instance ID exists in multiple providers.
    """

    async def _ssh_async() -> None:
        creds = resolve_credentials(ctx)
        ssh_key = resolve_ssh_key(ctx)

        client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

        # Parse provider:id format if present
        instance_id, parsed_provider = parse_instance_id(instance_id_arg)
        provider = provider_arg or parsed_provider

        # Auto-detect provider if not specified (same logic as status)
        if provider is None:
            instances = await client.list_instances()
            matches = [i for i in instances if i.id == instance_id]

            if len(matches) == 0:
                logger.error(f"✗ Instance {instance_id} not found in any provider")
                raise typer.Exit(1)
            if len(matches) > 1:
                logger.error(f"✗ Instance {instance_id} found in multiple providers:")
                for m in matches:
                    logger.error(f"  - {m.provider}")
                logger.info(f"specify provider: broker ssh {instance_id} <provider>")
                raise typer.Exit(1)

            instance = matches[0]
        else:
            instance = await client.get_instance(instance_id, provider)

        if not instance or not instance.public_ip:
            logger.error("✗ Instance not ready")
            raise typer.Exit(1)

        # Output full SSH command with key path (copy-pastable)
        print(instance._instance.ssh_connection_string(ssh_key_path=ssh_key, full_command=True))

    trio.run(_ssh_async)


@app.command()
def exec(  # noqa: A001 - typer command name, not shadowing builtin
    ctx: typer.Context,
    instance_id_arg: str = typer.Argument(..., metavar="INSTANCE_ID", help="Instance ID"),
    command: list[str] = typer.Argument(..., help="Command to execute"),
    provider_arg: str | None = typer.Option(
        None, "--provider", "-p", help="Provider (auto-detect if omitted)"
    ),
) -> None:
    """Execute a command on GPU instance via SSH

    Instance ID can include provider prefix: runpod:abc123 or just abc123.

    Example:
        broker exec abc123 -- hostname
        broker exec runpod:abc123 -- nvidia-smi
        broker exec abc123 -- tail -f /root/train.log
    """

    async def _exec_async() -> None:
        creds = resolve_credentials(ctx)
        ssh_key = resolve_ssh_key(ctx)

        client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

        # Parse provider:id format if present
        instance_id, parsed_provider = parse_instance_id(instance_id_arg)
        provider = provider_arg or parsed_provider

        # Auto-detect provider if not specified
        if provider is None:
            instances = await client.list_instances()
            matches = [i for i in instances if i.id == instance_id]

            if len(matches) == 0:
                logger.error(f"✗ Instance {instance_id} not found in any provider")
                raise typer.Exit(1)
            if len(matches) > 1:
                logger.error(f"✗ Instance {instance_id} found in multiple providers:")
                for m in matches:
                    logger.error(f"  - {m.provider}")
                logger.info(
                    f"specify provider: broker exec {instance_id} --provider <provider> -- <cmd>"
                )
                raise typer.Exit(1)

            instance = matches[0]
        else:
            instance = await client.get_instance(instance_id, provider)

        if not instance or not instance._instance.public_ip:
            logger.error("✗ Instance not ready (no public IP)")
            raise typer.Exit(1)

        # Build and execute SSH command via subprocess (simpler than async SSH)
        import subprocess

        ssh_cmd = instance._instance.ssh_connection_string(ssh_key_path=ssh_key, full_command=True)
        cmd_str = " ".join(command)
        full_cmd = f'{ssh_cmd} "{cmd_str}"'

        result = subprocess.run(full_cmd, shell=True)  # noqa: ASYNC221 - intentional blocking for interactive SSH
        raise typer.Exit(result.returncode)

    trio.run(_exec_async)


@app.command()
def info(
    ctx: typer.Context,
    instance_id_arg: str = typer.Argument(..., metavar="INSTANCE_ID", help="Instance ID"),
    provider_arg: str | None = typer.Argument(
        None, help="Provider (runpod|primeintellect). Auto-detect if omitted."
    ),
) -> None:
    """Get detailed system information from GPU instance

    Instance ID can include provider prefix: runpod:abc123 or just abc123.
    Collects GPU utilization, VRAM usage, CPU usage, memory usage, and disk usage
    via SSH connection to the instance.
    """

    async def _info_async() -> None:
        creds = resolve_credentials(ctx)
        ssh_key = resolve_ssh_key(ctx)

        client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

        # Parse provider:id format if present
        instance_id, parsed_provider = parse_instance_id(instance_id_arg)
        provider = provider_arg or parsed_provider

        # Auto-detect provider if not specified
        if provider is None:
            instances = await client.list_instances()
            matches = [i for i in instances if i.id == instance_id]

            if len(matches) == 0:
                logger.error(f"✗ Instance {instance_id} not found in any provider")
                raise typer.Exit(1)
            if len(matches) > 1:
                logger.error(f"✗ Instance {instance_id} found in multiple providers:")
                for m in matches:
                    logger.error(f"  - {m.provider}")
                logger.info(f"specify provider: broker info {instance_id} <provider>")
                raise typer.Exit(1)

            instance = matches[0]
        else:
            instance = await client.get_instance(instance_id, provider)

            if not instance:
                logger.error(f"✗ Instance {instance_id} not found in {provider}")
                raise typer.Exit(1)

        # Check if instance is ready
        if not instance._instance.public_ip:
            logger.error("✗ Instance not ready (no public IP)")
            raise typer.Exit(1)

        # Collect system info
        if not ctx.obj["json"]:
            logger.info("collecting system information...")

        # Temporarily suppress SSH connection logs
        ssh_logger = logging.getLogger("shared.ssh_foundation")
        paramiko_logger = logging.getLogger("paramiko")
        original_ssh_level = ssh_logger.level
        original_paramiko_level = paramiko_logger.level
        ssh_logger.setLevel(logging.WARNING)
        paramiko_logger.setLevel(logging.WARNING)

        try:
            # Get GPU info
            gpu_cmd = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
            gpu_result = await instance._instance.aexec(gpu_cmd)

            # Get CPU info
            cpu_cmd = "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'"
            cpu_result = await instance._instance.aexec(cpu_cmd)

            # Get memory info
            mem_cmd = "free -m | awk 'NR==2{printf \"%s,%s,%s\", $3,$2,$3*100/$2 }'"
            mem_result = await instance._instance.aexec(mem_cmd)

            # Get disk info
            disk_cmd = "df -h / | awk 'NR==2{printf \"%s,%s,%s\", $3,$2,$5}'"
            disk_result = await instance._instance.aexec(disk_cmd)

        except Exception as e:
            logger.exception(f"✗ Failed to collect system info: {e}")
            raise typer.Exit(1) from None
        finally:
            # Restore original log levels
            ssh_logger.setLevel(original_ssh_level)
            paramiko_logger.setLevel(original_paramiko_level)

        # Parse and display results
        if ctx.obj["json"]:
            # Parse GPU info
            gpus = []
            for line in gpu_result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "utilization_percent": float(parts[2]),
                            "memory_used_mb": float(parts[3]),
                            "memory_total_mb": float(parts[4]),
                            "memory_percent": (float(parts[3]) / float(parts[4])) * 100,
                        })

            # Parse CPU info
            cpu_util = float(cpu_result.stdout.strip()) if cpu_result.stdout.strip() else 0.0

            # Parse memory info
            mem_parts = mem_result.stdout.strip().split(",")
            mem_used_mb = int(mem_parts[0]) if len(mem_parts) > 0 else 0
            mem_total_mb = int(mem_parts[1]) if len(mem_parts) > 1 else 0
            mem_percent = float(mem_parts[2]) if len(mem_parts) > 2 else 0.0

            # Parse disk info
            disk_parts = disk_result.stdout.strip().split(",")
            disk_used = disk_parts[0] if len(disk_parts) > 0 else "0"
            disk_total = disk_parts[1] if len(disk_parts) > 1 else "0"
            disk_percent = disk_parts[2] if len(disk_parts) > 2 else "0%"

            print(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "provider": instance.provider,
                        "gpus": gpus,
                        "cpu_utilization_percent": cpu_util,
                        "memory_used_mb": mem_used_mb,
                        "memory_total_mb": mem_total_mb,
                        "memory_percent": mem_percent,
                        "disk_used": disk_used,
                        "disk_total": disk_total,
                        "disk_percent": disk_percent,
                    },
                    indent=2,
                )
            )
        else:
            # Display with Rich tables
            console.print(f"\n[bold]Instance: {instance_id}[/bold] ({instance.provider})\n")

            # GPU Table
            gpu_table = Table(title="GPU Utilization", show_header=True)
            gpu_table.add_column("GPU", style="cyan", justify="center")
            gpu_table.add_column("Name", style="white")
            gpu_table.add_column("GPU Util", justify="right")
            gpu_table.add_column("VRAM Used", justify="right")
            gpu_table.add_column("VRAM Total", justify="right")
            gpu_table.add_column("VRAM %", justify="right")

            for line in gpu_result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpu_idx = parts[0]
                        gpu_name = parts[1]
                        gpu_util = float(parts[2])
                        vram_used = float(parts[3])
                        vram_total = float(parts[4])
                        vram_percent = (vram_used / vram_total) * 100

                        # Color coding for utilization
                        gpu_util_str = f"{gpu_util:.1f}%"
                        if gpu_util > 80:
                            gpu_util_str = f"[red]{gpu_util_str}[/red]"
                        elif gpu_util > 50:
                            gpu_util_str = f"[yellow]{gpu_util_str}[/yellow]"
                        else:
                            gpu_util_str = f"[green]{gpu_util_str}[/green]"

                        vram_percent_str = f"{vram_percent:.1f}%"
                        if vram_percent > 80:
                            vram_percent_str = f"[red]{vram_percent_str}[/red]"
                        elif vram_percent > 50:
                            vram_percent_str = f"[yellow]{vram_percent_str}[/yellow]"
                        else:
                            vram_percent_str = f"[green]{vram_percent_str}[/green]"

                        gpu_table.add_row(
                            gpu_idx,
                            gpu_name,
                            gpu_util_str,
                            f"{vram_used:.0f} MB",
                            f"{vram_total:.0f} MB",
                            vram_percent_str,
                        )

            console.print(gpu_table)

            # System Resources Table
            sys_table = Table(title="System Resources", show_header=True)
            sys_table.add_column("Resource", style="cyan")
            sys_table.add_column("Used", justify="right")
            sys_table.add_column("Total", justify="right")
            sys_table.add_column("Utilization", justify="right")

            # CPU row
            cpu_util_str = cpu_result.stdout.strip()
            if cpu_util_str:
                cpu_util = float(cpu_util_str)
                cpu_display = f"{cpu_util:.1f}%"
                if cpu_util > 80:
                    cpu_display = f"[red]{cpu_display}[/red]"
                elif cpu_util > 50:
                    cpu_display = f"[yellow]{cpu_display}[/yellow]"
                else:
                    cpu_display = f"[green]{cpu_display}[/green]"
                sys_table.add_row("CPU", "-", "-", cpu_display)

            # Memory row
            mem_parts = mem_result.stdout.strip().split(",")
            if len(mem_parts) >= 3:
                mem_used = int(mem_parts[0])
                mem_total = int(mem_parts[1])
                mem_percent = float(mem_parts[2])

                mem_percent_str = f"{mem_percent:.1f}%"
                if mem_percent > 80:
                    mem_percent_str = f"[red]{mem_percent_str}[/red]"
                elif mem_percent > 50:
                    mem_percent_str = f"[yellow]{mem_percent_str}[/yellow]"
                else:
                    mem_percent_str = f"[green]{mem_percent_str}[/green]"

                sys_table.add_row(
                    "Memory",
                    f"{mem_used} MB",
                    f"{mem_total} MB",
                    mem_percent_str,
                )

            # Disk row
            disk_parts = disk_result.stdout.strip().split(",")
            if len(disk_parts) >= 3:
                disk_used = disk_parts[0]
                disk_total = disk_parts[1]
                disk_percent_raw = disk_parts[2].rstrip("%")

                try:
                    disk_pct = float(disk_percent_raw)
                    disk_display = f"{disk_pct:.0f}%"
                    if disk_pct > 80:
                        disk_display = f"[red]{disk_display}[/red]"
                    elif disk_pct > 50:
                        disk_display = f"[yellow]{disk_display}[/yellow]"
                    else:
                        disk_display = f"[green]{disk_display}[/green]"
                except ValueError:
                    disk_display = disk_parts[2]

                sys_table.add_row(
                    "Disk (/)",
                    disk_used,
                    disk_total,
                    disk_display,
                )

            console.print(sys_table)
            console.print()

    trio.run(_info_async)


@app.command()
def terminate(
    ctx: typer.Context,
    instance_id_arg: str = typer.Argument(..., metavar="INSTANCE_ID"),
    provider_arg: str | None = typer.Argument(
        None, help="Provider (runpod|primeintellect). Auto-detect if omitted."
    ),
    yes: bool = typer.Option(True, "-y", "--yes", help="Skip confirmation (default: yes)"),
) -> None:
    """Terminate GPU instance

    Instance ID can include provider prefix: runpod:abc123 or just abc123.
    Provider can be omitted for convenience. If omitted, will search all
    providers. Errors if instance ID exists in multiple providers.
    """

    async def _terminate_async() -> None:
        creds = resolve_credentials(ctx)
        ssh_key = resolve_ssh_key(ctx)

        client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

        # Parse provider:id format if present
        instance_id, parsed_provider = parse_instance_id(instance_id_arg)
        provider = provider_arg or parsed_provider

        # Auto-detect provider if not specified (same logic as status)
        if provider is None:
            instances = await client.list_instances()
            matches = [i for i in instances if i.id == instance_id]

            if len(matches) == 0:
                logger.error(f"✗ Instance {instance_id} not found in any provider")
                raise typer.Exit(1)
            if len(matches) > 1:
                logger.error(f"✗ Instance {instance_id} found in multiple providers:")
                for m in matches:
                    logger.error(f"  - {m.provider}")
                logger.info(f"specify provider: broker terminate {instance_id} <provider>")
                raise typer.Exit(1)

            instance = matches[0]
            provider_to_use = instance.provider
        else:
            instance = await client.get_instance(instance_id, provider)
            if not instance:
                logger.error(f"✗ Instance {instance_id} not found in {provider}")
                raise typer.Exit(1)
            provider_to_use = provider

        # Confirmation prompt
        if not yes:
            confirm = typer.confirm(f"Terminate instance {instance_id} on {provider_to_use}?")
            if not confirm:
                logger.info("cancelled")
                raise typer.Exit(0)

        success = await client.terminate_instance(instance_id, provider_to_use)

        if success:
            logger.info(f"instance {instance_id} terminated")
        else:
            logger.error("✗ Failed to terminate instance")
            raise typer.Exit(1)

    trio.run(_terminate_async)


@app.command()
def cleanup(
    ctx: typer.Context,
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation"),
    provider: str | None = typer.Option(
        None, "--provider", help="Only terminate instances from this provider"
    ),
    exclude: builtins.list[str] | None = typer.Option(
        None,
        "--exclude",
        help="Instance IDs to exclude from cleanup (can be specified multiple times)",
    ),
) -> None:
    """Terminate all running GPU instances

    This command lists all your instances across all configured providers
    and terminates them. Use with caution!

    Examples:
        broker cleanup --exclude abc123 --exclude def456
        broker cleanup --exclude abc123 def456 ghi789
    """

    async def _cleanup_async() -> None:
        creds = resolve_credentials(ctx)
        ssh_key = resolve_ssh_key(ctx)

        client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

        # Get all instances
        logger.info("fetching all instances...")
        instances = await client.list_instances()

        # Filter by provider if specified
        if provider:
            instances = [i for i in instances if i.provider == provider]

        # Exclude specified instances
        if exclude:
            exclude_set = set(exclude)
            instances = [i for i in instances if i.id not in exclude_set]
            if exclude_set:
                logger.info(f"excluding {len(exclude_set)} instance(s) from cleanup")

        if not instances:
            if provider:
                logger.info(f"no instances found in {provider}")
            else:
                logger.info("no instances found")
            return

        # Display instances to be terminated
        if not ctx.obj["json"]:
            table = Table(title=f"Instances to Terminate ({len(instances)})")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name")
            table.add_column("Provider")
            table.add_column("GPUs")
            table.add_column("Status")
            table.add_column("Price/hr", justify="right")

            for instance in instances:
                if instance.gpu_count > 1:
                    gpu_display = f"{instance.gpu_count}x {instance.gpu_type}"
                else:
                    gpu_display = instance.gpu_type

                node_price = instance.price_per_hour * instance.gpu_count

                table.add_row(
                    instance.id,
                    instance.name or "-",
                    instance.provider,
                    gpu_display,
                    instance.status.value,
                    f"${node_price:.2f}",
                )

            console.print(table)

        # Confirmation prompt
        if not yes:
            if provider:
                confirm = typer.confirm(
                    f"Terminate all {len(instances)} instance(s) in {provider}?"
                )
            else:
                confirm = typer.confirm(
                    f"Terminate all {len(instances)} instance(s) across all providers?"
                )
            if not confirm:
                logger.info("cancelled")
                raise typer.Exit(0)

        # Terminate all instances
        failed = []
        succeeded = []

        for instance in instances:
            logger.info(f"terminating {instance.id} ({instance.provider})...")
            success = await client.terminate_instance(instance.id, instance.provider)

            if success:
                succeeded.append(instance)
            else:
                failed.append(instance)

        # Report results
        if succeeded:
            logger.info(f"successfully terminated {len(succeeded)} instance(s)")

        if failed:
            logger.error(f"✗ Failed to terminate {len(failed)} instance(s):")
            for instance in failed:
                logger.error(f"  - {instance.id} ({instance.provider})")
            raise typer.Exit(1)

    trio.run(_cleanup_async)


@app.command()
def logs(
    ctx: typer.Context,
    pod_id: str = typer.Argument(..., help="RunPod pod ID"),
) -> None:
    """Fetch system logs from a RunPod pod.

    This uses a separate Chrome debug instance to authenticate with RunPod's
    internal API. On first run, it will launch Chrome - log into RunPod console
    once, then logs will work automatically.

    Example:
        broker logs 2ymw46y21mcz4f
    """
    from broker.runpod_logs import fetch_pod_logs, is_chrome_debug_running, launch_chrome_debug

    if not is_chrome_debug_running():
        logger.info("launching Chrome debug instance...")
        launch_chrome_debug()
        console.print(
            "[yellow]Chrome debug instance launched.[/yellow]\n"
            "Please log into RunPod console: https://console.runpod.io\n"
            "Then run this command again."
        )
        raise typer.Exit(0)

    result = fetch_pod_logs(pod_id)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)

    if ctx.obj["json"]:
        print(json.dumps(result, indent=2))
    else:
        # Pretty print logs
        if isinstance(result, list):
            for entry in result:
                console.print(entry)
        else:
            console.print(result)


@volumes_app.callback(invoke_without_command=True)
def volumes_root(ctx: typer.Context) -> None:
    """Manage RunPod network volumes."""
    if ctx.invoked_subcommand is None:
        volumes_list(ctx)


@volumes_app.command("list")
def volumes_list(ctx: typer.Context) -> None:
    """List RunPod network volumes and their datacenters.

    Network volumes are persistent storage that survive pod termination.
    Each volume is locked to a specific datacenter — you must provision
    pods in the same datacenter to attach the volume.

    Use the volume ID and datacenter ID with:
        broker create --network-volume-id <id> --datacenter-id <dc-id> ...
    """

    async def _volumes_async() -> None:
        creds = resolve_credentials(ctx)
        runpod_key = creds.runpod
        if not runpod_key:
            logger.error("RunPod API key required. Run: broker auth login runpod")
            raise typer.Exit(1)

        from broker.providers.runpod import list_network_volumes

        vols = await list_network_volumes(api_key=runpod_key)

        if not vols:
            console.print("No network volumes found.")
            console.print(
                "Create one with: broker volumes create --name <name> --datacenter-id <dc-id> --size-gb <gb>"
            )
            return

        console.print(f"Found {len(vols)} network volume(s):\n")
        for vol in vols:
            vol_id = vol.get("id", "unknown")
            vol_name = vol.get("name", "unnamed")
            dc_id = vol.get("dataCenterId", "unknown")
            size_gb = vol.get("size", "?")
            console.print(f"  {vol_name}")
            console.print(f"    id:           {vol_id}")
            console.print(f"    datacenter:   {dc_id}")
            console.print(f"    size:         {size_gb} GB")
            console.print(f"    attach with:  --network-volume-id {vol_id} --datacenter-id {dc_id}")
            console.print("")

    trio.run(_volumes_async)


@volumes_app.command("create")
def volumes_create(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="Network volume name"),
    datacenter_id: str = typer.Option(..., "--datacenter-id", help="RunPod datacenter ID"),
    size_gb: int = typer.Option(..., "--size-gb", min=1, help="Volume size in GB"),
    if_missing: bool = typer.Option(
        True,
        "--if-missing/--no-if-missing",
        help="Reuse existing volume with same name+datacenter instead of failing",
    ),
) -> None:
    """Create a RunPod network volume via REST API."""

    async def _create_volume_async() -> None:
        creds = resolve_credentials(ctx)
        runpod_key = creds.runpod
        if not runpod_key:
            logger.error("RunPod API key required. Run: broker auth login runpod")
            raise typer.Exit(1)

        from broker.providers.runpod import create_network_volume, list_network_volumes

        existing = await list_network_volumes(api_key=runpod_key)
        matching = [
            vol
            for vol in existing
            if vol.get("name") == name and vol.get("dataCenterId") == datacenter_id
        ]
        if matching:
            existing_vol = matching[0]
            existing_size = existing_vol.get("size")
            if existing_size is not None and str(existing_size) != str(size_gb):
                logger.warning(
                    f"Existing volume has size {existing_size} GB (requested {size_gb} GB). Reusing existing."
                )
            if if_missing:
                if ctx.obj["json"]:
                    print(json.dumps(existing_vol, indent=2))
                else:
                    console.print(f"Volume already exists: {name}")
                    console.print(f"  id:         {existing_vol.get('id', 'unknown')}")
                    console.print(f"  datacenter: {datacenter_id}")
                    console.print(f"  size:       {existing_size} GB")
                return

            logger.error(f"Volume already exists: name={name}, datacenter={datacenter_id}")
            raise typer.Exit(1)

        created = await create_network_volume(
            name=name,
            datacenter_id=datacenter_id,
            size_gb=size_gb,
            api_key=runpod_key,
        )
        if ctx.obj["json"]:
            print(json.dumps(created, indent=2))
        else:
            console.print(f"Created volume: {created.get('name', name)}")
            console.print(f"  id:         {created.get('id', 'unknown')}")
            console.print(f"  datacenter: {created.get('dataCenterId', datacenter_id)}")
            console.print(f"  size:       {created.get('size', size_gb)} GB")
            console.print(
                f"  attach with: --network-volume-id {created.get('id', 'unknown')} --datacenter-id {created.get('dataCenterId', datacenter_id)}"
            )

    trio.run(_create_volume_async)


@volumes_app.command("delete")
def volumes_delete(
    ctx: typer.Context,
    volume_id: str | None = typer.Option(None, "--id", help="Network volume ID to delete"),
    name: str | None = typer.Option(None, "--name", help="Network volume name to resolve"),
    datacenter_id: str | None = typer.Option(
        None, "--datacenter-id", help="RunPod datacenter ID (required with --name)"
    ),
    if_missing: bool = typer.Option(
        True,
        "--if-missing/--no-if-missing",
        help="Treat missing volume as success",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip delete confirmation prompt"),
) -> None:
    """Delete a RunPod network volume via REST API."""

    async def _delete_volume_async() -> None:
        if volume_id is None and name is None:
            logger.error("Provide either --id or --name with --datacenter-id")
            raise typer.Exit(1)
        if volume_id is not None and name is not None:
            logger.error("Use either --id or --name (not both)")
            raise typer.Exit(1)
        if name is not None and datacenter_id is None:
            logger.error("--datacenter-id is required with --name")
            raise typer.Exit(1)
        if datacenter_id is not None and name is None:
            logger.error("--datacenter-id can only be used with --name")
            raise typer.Exit(1)

        creds = resolve_credentials(ctx)
        runpod_key = creds.runpod
        if not runpod_key:
            logger.error("RunPod API key required. Run: broker auth login runpod")
            raise typer.Exit(1)

        from broker.providers.runpod import delete_network_volume, list_network_volumes

        resolved_id = volume_id
        resolved_name = name or "unknown"
        resolved_datacenter = datacenter_id or "unknown"
        resolved_size = "?"

        if resolved_id is None:
            volumes = await list_network_volumes(api_key=runpod_key)
            matches = [
                vol
                for vol in volumes
                if vol.get("name") == name and vol.get("dataCenterId") == datacenter_id
            ]
            if not matches:
                if if_missing:
                    if ctx.obj["json"]:
                        print(json.dumps({"deleted": False, "reason": "not_found"}, indent=2))
                    else:
                        console.print(
                            f"Volume not found for name={name}, datacenter={datacenter_id}; nothing to delete."
                        )
                    return
                logger.error(f"Volume not found for name={name}, datacenter={datacenter_id}")
                raise typer.Exit(1)
            selected = matches[0]
            resolved_id = selected.get("id")
            resolved_name = selected.get("name", resolved_name)
            resolved_datacenter = selected.get("dataCenterId", resolved_datacenter)
            resolved_size = selected.get("size", resolved_size)

        assert resolved_id is not None
        if not yes:
            answer = typer.confirm(
                f"Delete volume {resolved_name} (id={resolved_id}, dc={resolved_datacenter}, size={resolved_size} GB)? [y/N]: "
            )
            if not answer:
                console.print("Aborted.")
                raise typer.Exit(1)

        await delete_network_volume(volume_id=resolved_id, api_key=runpod_key)
        if ctx.obj["json"]:
            print(
                json.dumps(
                    {
                        "deleted": True,
                        "id": resolved_id,
                        "name": resolved_name,
                        "dataCenterId": resolved_datacenter,
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"Deleted volume: {resolved_name}")
            console.print(f"  id:         {resolved_id}")
            console.print(f"  datacenter: {resolved_datacenter}")

    trio.run(_delete_volume_async)


@auth_app.command("login")
def auth_login(
    provider: str = typer.Argument(
        ..., help="Provider name (runpod, vast, lambdalabs, primeintellect)"
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", "-k", help="API key (prompted if omitted)"
    ),
    profile: str = typer.Option("default", "--profile", "-p", help="Profile name"),
) -> None:
    """Save a provider API key to ~/.broker/credentials.toml."""
    if provider not in KNOWN_PROVIDERS:
        logger.error(f"Unknown provider: {provider}")
        logger.info(f"Known providers: {', '.join(sorted(KNOWN_PROVIDERS))}")
        raise typer.Exit(1)

    if api_key is None:
        api_key = typer.prompt(f"{provider} API key", hide_input=True)

    assert api_key, "API key cannot be empty"

    set_profile_key(profile, provider, api_key)
    logger.info(f"Saved {provider} key to profile '{profile}' ({key_preview(api_key)})")
    logger.info(f"Config: {CREDENTIALS_FILE}")


@auth_app.command("status")
def auth_status() -> None:
    """Show configured credentials and active profile."""
    from broker.credentials import ENV_VAR_MAP, get_credentials

    profiles = load_profiles()

    if not profiles:
        logger.info("No credentials configured. Run: broker auth login <provider>")
        logger.info(f"Config file: {CREDENTIALS_FILE}")
        return

    # Show resolved credentials — what will actually be used
    resolved = get_credentials()
    if resolved:
        console.print("[bold]Active credentials[/bold]")
        # Build reverse lookup: provider -> source
        _, active_profile_creds = next(
            ((n, p) for n, p in profiles.items() if isinstance(p, dict) and p.get("active")),
            (None, {}),
        )
        for provider, key in resolved.items():
            if isinstance(active_profile_creds, dict) and active_profile_creds.get(provider) == key:
                source = "profile"
            else:
                source = "env"
            console.print(f"  {provider}: {key_preview(key)} [dim]({source})[/dim]")
    else:
        console.print("[bold]No active credentials[/bold]")
        console.print("  Run: broker auth login <provider>")

    # Show all profiles
    console.print(f"\n[bold]Profiles[/bold] [dim]({CREDENTIALS_FILE})[/dim]")
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        is_active = profile.get("active", False)
        marker = " *" if is_active else ""
        allowed = profile.get("providers")
        providers_str = f" [dim](providers: {', '.join(allowed)})[/dim]" if allowed else ""
        console.print(f"  {name}{marker}{providers_str}")

    # Show ignored env vars — env vars that exist but are overridden by profile
    ignored_env = []
    for env_var, provider in ENV_VAR_MAP.items():
        env_val = os.getenv(env_var)
        if not env_val:
            continue
        profile_val = (
            active_profile_creds.get(provider) if isinstance(active_profile_creds, dict) else None
        )
        if profile_val and profile_val != env_val:
            ignored_env.append((provider, env_var, key_preview(env_val)))

    if ignored_env:
        console.print("\n[dim]Ignored env vars (profile key takes precedence)[/dim]")
        for provider, env_var, preview in ignored_env:
            console.print(f"  [dim]{provider}: {preview} ({env_var})[/dim]")


@auth_app.command("switch")
def auth_switch(
    profile: str = typer.Argument(..., help="Profile name to activate"),
) -> None:
    """Switch the active credentials profile."""
    try:
        set_active_profile(profile)
        logger.info(f"Switched to profile '{profile}'")
    except ValueError as e:
        logger.exception(str(e))
        profiles = load_profiles()
        if profiles:
            logger.info(f"Available profiles: {', '.join(profiles.keys())}")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
