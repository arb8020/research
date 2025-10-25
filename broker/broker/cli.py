"""Broker CLI - GPU provisioning and management"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from broker.client import GPUClient
from broker.types import ProviderCredentials
from shared.config import (
    create_env_template,
    discover_ssh_keys,
    get_lambda_key,
    get_prime_key,
    get_runpod_key,
    get_ssh_key_path,
)
from shared.logging_config import setup_logging

console = Console()
app = typer.Typer(help="GPU broker - provision cloud GPUs")

# Logger will be configured in callback
logger = logging.getLogger("broker")


@app.callback()
def main(
    ctx: typer.Context,
    credentials: Optional[str] = typer.Option(
        None,
        "--credentials",
        help="Credentials file or inline 'runpod:key,primeintellect:key'",
    ),
    ssh_key: Optional[str] = typer.Option(
        None, "--ssh-key", help="Path to SSH private key"
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Show only warnings and errors"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    debug: bool = typer.Option(False, "--debug", help="Show debug logs"),
):
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


@app.command()
def init():
    """Create .env template for credentials

    Creates a .env file in the current directory with template
    for RunPod/Prime Intellect API keys and SSH key path.
    """
    try:
        create_env_template("broker")
        logger.info("✓ Created .env with credential template")
        logger.info("")
        logger.info("Edit .env with your API keys:")
        logger.info("  RUNPOD_API_KEY=your_key_here")
        logger.info("  PRIME_API_KEY=your_key_here")
        logger.info("  LAMBDA_API_KEY=your_key_here")
        logger.info("  SSH_KEY_PATH=~/.ssh/id_ed25519")
        logger.info("")
        logger.info("Then run: broker search")
    except FileExistsError:
        logger.error("✗ .env already exists")
        logger.info("")
        logger.info("To edit manually:")
        logger.info(f"  Open: {Path('.env').absolute()}")
        logger.info("  Ensure it contains:")
        logger.info("    RUNPOD_API_KEY=your_key_here")
        logger.info("    PRIME_API_KEY=your_key_here")
        logger.info("    LAMBDA_API_KEY=your_key_here")
        logger.info("    SSH_KEY_PATH=~/.ssh/id_ed25519")
        logger.info("")
        logger.info("Or delete .env and run 'broker init' again")
        raise typer.Exit(1)


def resolve_credentials(ctx) -> ProviderCredentials:
    """Resolve credentials from CLI → env → .env → error

    Priority:
    1. --credentials flag (file path or inline format)
    2. Environment variables (RUNPOD_API_KEY, PRIME_API_KEY)
    3. .env file (loaded by python-dotenv)
    4. Error with helpful message
    """
    creds_arg = ctx.obj.get("credentials")

    # Priority 1: CLI flag
    if creds_arg:
        if Path(creds_arg).exists():
            # File path
            with open(creds_arg) as f:
                creds_dict = json.load(f)
            return ProviderCredentials.from_dict(creds_dict)
        else:
            # Inline format: "runpod:key,primeintellect:key"
            parts = creds_arg.split(",")
            creds_dict = {}
            for part in parts:
                if ":" not in part:
                    logger.error(f"✗ Invalid credentials format: {part}")
                    logger.info("Expected: runpod:key,primeintellect:key")
                    raise typer.Exit(1)
                provider, key = part.split(":", 1)
                creds_dict[provider.strip()] = key.strip()
            return ProviderCredentials.from_dict(creds_dict)

    # Priority 2+3: Environment variables (includes .env via load_dotenv)
    runpod_key = get_runpod_key()
    prime_key = get_prime_key()
    lambda_key = get_lambda_key()

    if runpod_key or prime_key or lambda_key:
        return ProviderCredentials(
            runpod=runpod_key or "",
            primeintellect=prime_key or "",
            lambdalabs=lambda_key or ""
        )

    # Priority 4: Error with helpful message
    logger.error("✗ No credentials found")
    logger.info("")
    logger.info("Try: broker init")
    logger.info("Or: export RUNPOD_API_KEY=... PRIME_API_KEY=... LAMBDA_API_KEY=...")
    logger.info("Or: --credentials <file|runpod:key,primeintellect:key,lambdalabs:key>")
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
        logger.info("Found keys at:")
        for key in found_keys:
            logger.info(f"  {key}")
        logger.info("")
        logger.info("Set SSH_KEY_PATH in .env (run: broker init)")
        logger.info(f"Or use: --ssh-key {found_keys[0]}")
    else:
        logger.info("No SSH keys found in ~/.ssh/")
        logger.info("Generate one: ssh-keygen -t ed25519")
        logger.info("")
        logger.info("Then set SSH_KEY_PATH in .env (run: broker init)")

    raise typer.Exit(1)


@app.command()
def search(
    ctx: typer.Context,
    gpu_type: Optional[str] = typer.Option(
        None, "--gpu-type", help="Filter by GPU type (e.g., 'A100')"
    ),
    gpu_count: int = typer.Option(
        1, "--gpu-count", help="Number of GPUs (affects pricing, default: 1)"
    ),
    max_price_per_gpu: Optional[float] = typer.Option(
        None, "--max-price-per-gpu", help="Maximum price per GPU per hour"
    ),
    max_total_price: Optional[float] = typer.Option(
        None, "--max-total-price", help="Maximum total price per hour"
    ),
    max_price: Optional[float] = typer.Option(
        None, "--max-price", help="(Deprecated) Use --max-price-per-gpu instead"
    ),
    min_vram: Optional[int] = typer.Option(None, "--min-vram", help="Minimum VRAM in GB"),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Filter by provider (runpod|primeintellect)"
    ),
    cloud_type: Optional[str] = typer.Option(
        "secure", "--cloud-type", help="Cloud type: secure (default, guaranteed) or community (spot, cheaper but can be interrupted)"
    ),
    underlying_provider: Optional[str] = typer.Option(
        None, "--underlying-provider", help="Filter by underlying provider (e.g., massedcompute, hyperstack) for aggregators like PrimeIntellect"
    ),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results"),
):
    """Search for available GPU offers

    By default searches all configured providers and merges results.
    Use --provider to filter to specific provider.
    """
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
        logger.info(f"Searching for GPU offers (gpu_count={gpu_count})...")
    else:
        logger.info("Searching for GPU offers...")

    # Import api to call search with gpu_count
    from broker import api
    offers = api.search(query, gpu_count=gpu_count, sort=lambda x: x.price_per_hour, credentials=creds.to_dict())

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
            logger.info(f"Showing {len(offers)} cheapest offers for {gpu_count}x GPUs")
        else:
            logger.info(f"Showing {len(offers)} cheapest offers")


@app.command()
def create(
    ctx: typer.Context,
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type", help="GPU type filter"),
    gpu_count: int = typer.Option(1, "--gpu-count", help="Number of GPUs (default: 1)"),
    max_price_per_gpu: Optional[float] = typer.Option(
        None, "--max-price-per-gpu", help="Maximum price per GPU per hour"
    ),
    max_total_price: Optional[float] = typer.Option(
        None, "--max-total-price", help="Maximum total price per hour"
    ),
    max_price: Optional[float] = typer.Option(
        None, "--max-price", help="(Deprecated) Use --max-price-per-gpu instead"
    ),
    cloud_type: Optional[str] = typer.Option(
        "secure", "--cloud-type", help="Cloud type: secure (default, guaranteed) or community (spot, cheaper but can be interrupted)"
    ),
    image: str = typer.Option(
        "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204", "--image", help="Docker image to use"
    ),
    name: Optional[str] = typer.Option(None, "--name", help="Instance name"),
    wait_ssh: bool = typer.Option(
        False, "--wait-ssh", help="Wait for SSH to be ready before returning"
    ),
    output: str = typer.Option(
        "summary", "--output", help="Output format: summary|ssh|json"
    ),
):
    """Provision a new GPU instance

    By default returns immediately after provisioning starts.
    Use --wait-ssh to block until SSH is ready.

    Output formats:
      summary: Human-readable status (default)
      ssh: Just SSH connection string (for piping to bifrost)
      json: Full instance details as JSON
    """
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
        logger.info(f"Provisioning {gpu_count}x GPU instance{cloud_msg}...")
    else:
        cloud_msg = f" ({cloud_type} cloud)" if cloud_type else ""
        logger.info(f"Provisioning instance{cloud_msg}...")
    instance = client.create(query, image=image, name=name, gpu_count=gpu_count)

    if not instance:
        logger.error("✗ Failed to provision instance")
        raise typer.Exit(1)

    # Wait for SSH if requested
    if wait_ssh:
        logger.info("Waiting for SSH to be ready...")
        if not instance.wait_until_ssh_ready(timeout=300):
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
            gpu_info = f"{instance.gpu_count}x {instance.gpu_type}" if instance.gpu_count > 1 else instance.gpu_type
            logger.info(f"✓ Instance {instance.id} ready: {instance.ssh_connection_string()}")
            logger.info(f"  GPU: {gpu_info}")
            if instance.gpu_count > 1:
                logger.info(f"  Price: ${instance.price_per_hour:.2f}/GPU/hr (${instance.price_per_hour * instance.gpu_count:.2f}/hr total)")
            else:
                logger.info(f"  Price: ${instance.price_per_hour:.2f}/hr")
        else:
            gpu_info = f"{instance.gpu_count}x {instance.gpu_type}" if instance.gpu_count > 1 else instance.gpu_type
            logger.info(f"✓ Instance {instance.id} provisioning started")
            logger.info(f"  GPU: {gpu_info}")
            if instance.gpu_count > 1:
                logger.info(f"  Price: ${instance.price_per_hour:.2f}/GPU/hr (${instance.price_per_hour * instance.gpu_count:.2f}/hr total)")
            else:
                logger.info(f"  Price: ${instance.price_per_hour:.2f}/hr")
            logger.info(f"Use 'broker status {instance.id}' to check progress")


@app.command()
def list(ctx: typer.Context):
    """List all your GPU instances"""
    creds = resolve_credentials(ctx)
    ssh_key = resolve_ssh_key(ctx)

    client = GPUClient(credentials=creds, ssh_key_path=ssh_key)
    instances = client.list_instances()

    if ctx.obj["json"]:
        print(
            json.dumps(
                [
                    {
                        "id": i.id,
                        "provider": i.provider,
                        "gpu_type": i.gpu_type,
                        "status": i.status.value,
                        "price_per_hour": i.price_per_hour,
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


@app.command()
def status(
    ctx: typer.Context,
    instance_id: str = typer.Argument(..., help="Instance ID"),
    provider: Optional[str] = typer.Argument(
        None, help="Provider (runpod|primeintellect). Auto-detect if omitted."
    ),
):
    """Get instance status

    Provider can be omitted for convenience. If omitted, will search all
    providers. Errors if instance ID exists in multiple providers.
    """
    creds = resolve_credentials(ctx)
    ssh_key = resolve_ssh_key(ctx)

    client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

    # Auto-detect provider if not specified
    if provider is None:
        instances = client.list_instances()
        matches = [i for i in instances if i.id == instance_id]

        if len(matches) == 0:
            logger.error(f"✗ Instance {instance_id} not found in any provider")
            raise typer.Exit(1)
        elif len(matches) > 1:
            logger.error(f"✗ Instance {instance_id} found in multiple providers:")
            for m in matches:
                logger.error(f"  - {m.provider}")
            logger.info(f"Specify provider: broker status {instance_id} <provider>")
            raise typer.Exit(1)

        instance = matches[0]
    else:
        instance = client.get_instance(instance_id, provider)

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
                    "ssh": (
                        instance.ssh_connection_string() if instance.public_ip else None
                    ),
                },
                indent=2,
            )
        )
    else:
        logger.info(f"Instance: {instance.id}")
        logger.info(f"Provider: {instance.provider}")
        logger.info(f"Status: {instance.status.value}")
        logger.info(f"GPU: {instance.gpu_type}")
        if instance.public_ip:
            logger.info(f"SSH: {instance.ssh_connection_string()}")


@app.command()
def ssh(
    ctx: typer.Context,
    instance_id: str = typer.Argument(...),
    provider: Optional[str] = typer.Argument(
        None, help="Provider (runpod|primeintellect). Auto-detect if omitted."
    ),
):
    """Get SSH connection string for instance

    Provider can be omitted for convenience. If omitted, will search all
    providers. Errors if instance ID exists in multiple providers.
    """
    creds = resolve_credentials(ctx)
    ssh_key = resolve_ssh_key(ctx)

    client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

    # Auto-detect provider if not specified (same logic as status)
    if provider is None:
        instances = client.list_instances()
        matches = [i for i in instances if i.id == instance_id]

        if len(matches) == 0:
            logger.error(f"✗ Instance {instance_id} not found in any provider")
            raise typer.Exit(1)
        elif len(matches) > 1:
            logger.error(f"✗ Instance {instance_id} found in multiple providers:")
            for m in matches:
                logger.error(f"  - {m.provider}")
            logger.info(f"Specify provider: broker ssh {instance_id} <provider>")
            raise typer.Exit(1)

        instance = matches[0]
    else:
        instance = client.get_instance(instance_id, provider)

    if not instance or not instance.public_ip:
        logger.error("✗ Instance not ready")
        raise typer.Exit(1)

    # Output full SSH command with key path (copy-pastable)
    print(instance._instance.ssh_connection_string(ssh_key_path=ssh_key, full_command=True))


@app.command()
def info(
    ctx: typer.Context,
    instance_id: str = typer.Argument(..., help="Instance ID"),
    provider: Optional[str] = typer.Argument(
        None, help="Provider (runpod|primeintellect). Auto-detect if omitted."
    ),
):
    """Get detailed system information from GPU instance

    Collects GPU utilization, VRAM usage, CPU usage, memory usage, and disk usage
    via SSH connection to the instance.
    """
    creds = resolve_credentials(ctx)
    ssh_key = resolve_ssh_key(ctx)

    client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

    # Auto-detect provider if not specified
    if provider is None:
        instances = client.list_instances()
        matches = [i for i in instances if i.id == instance_id]

        if len(matches) == 0:
            logger.error(f"✗ Instance {instance_id} not found in any provider")
            raise typer.Exit(1)
        elif len(matches) > 1:
            logger.error(f"✗ Instance {instance_id} found in multiple providers:")
            for m in matches:
                logger.error(f"  - {m.provider}")
            logger.info(f"Specify provider: broker info {instance_id} <provider>")
            raise typer.Exit(1)

        instance = matches[0]
    else:
        instance = client.get_instance(instance_id, provider)

        if not instance:
            logger.error(f"✗ Instance {instance_id} not found in {provider}")
            raise typer.Exit(1)

    # Check if instance is ready
    if not instance._instance.public_ip:
        logger.error("✗ Instance not ready (no public IP)")
        raise typer.Exit(1)

    # Collect system info
    if not ctx.obj["json"]:
        logger.info("Collecting system information...")

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
        gpu_result = instance._instance.exec(gpu_cmd)

        # Get CPU info
        cpu_cmd = "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'"
        cpu_result = instance._instance.exec(cpu_cmd)

        # Get memory info
        mem_cmd = "free -m | awk 'NR==2{printf \"%s,%s,%s\", $3,$2,$3*100/$2 }'"
        mem_result = instance._instance.exec(mem_cmd)

        # Get disk info
        disk_cmd = "df -h / | awk 'NR==2{printf \"%s,%s,%s\", $3,$2,$5}'"
        disk_result = instance._instance.exec(disk_cmd)

    except Exception as e:
        logger.error(f"✗ Failed to collect system info: {e}")
        raise typer.Exit(1)
    finally:
        # Restore original log levels
        ssh_logger.setLevel(original_ssh_level)
        paramiko_logger.setLevel(original_paramiko_level)

    # Parse and display results
    if ctx.obj["json"]:
        # Parse GPU info
        gpus = []
        for line in gpu_result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
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
        mem_parts = mem_result.stdout.strip().split(',')
        mem_used_mb = int(mem_parts[0]) if len(mem_parts) > 0 else 0
        mem_total_mb = int(mem_parts[1]) if len(mem_parts) > 1 else 0
        mem_percent = float(mem_parts[2]) if len(mem_parts) > 2 else 0.0

        # Parse disk info
        disk_parts = disk_result.stdout.strip().split(',')
        disk_used = disk_parts[0] if len(disk_parts) > 0 else "0"
        disk_total = disk_parts[1] if len(disk_parts) > 1 else "0"
        disk_percent = disk_parts[2] if len(disk_parts) > 2 else "0%"

        print(json.dumps({
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
        }, indent=2))
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

        for line in gpu_result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
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
        mem_parts = mem_result.stdout.strip().split(',')
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
        disk_parts = disk_result.stdout.strip().split(',')
        if len(disk_parts) >= 3:
            disk_used = disk_parts[0]
            disk_total = disk_parts[1]
            disk_percent_raw = disk_parts[2].rstrip('%')

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


@app.command()
def terminate(
    ctx: typer.Context,
    instance_id: str = typer.Argument(...),
    provider: Optional[str] = typer.Argument(
        None, help="Provider (runpod|primeintellect). Auto-detect if omitted."
    ),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation"),
):
    """Terminate GPU instance

    Provider can be omitted for convenience. If omitted, will search all
    providers. Errors if instance ID exists in multiple providers.
    """
    creds = resolve_credentials(ctx)
    ssh_key = resolve_ssh_key(ctx)

    client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

    # Auto-detect provider if not specified (same logic as status)
    if provider is None:
        instances = client.list_instances()
        matches = [i for i in instances if i.id == instance_id]

        if len(matches) == 0:
            logger.error(f"✗ Instance {instance_id} not found in any provider")
            raise typer.Exit(1)
        elif len(matches) > 1:
            logger.error(f"✗ Instance {instance_id} found in multiple providers:")
            for m in matches:
                logger.error(f"  - {m.provider}")
            logger.info(f"Specify provider: broker terminate {instance_id} <provider>")
            raise typer.Exit(1)

        instance = matches[0]
        provider = instance.provider
    else:
        instance = client.get_instance(instance_id, provider)
        if not instance:
            logger.error(f"✗ Instance {instance_id} not found in {provider}")
            raise typer.Exit(1)

    # Confirmation prompt
    if not yes:
        confirm = typer.confirm(f"Terminate instance {instance_id} on {provider}?")
        if not confirm:
            logger.info("Cancelled")
            raise typer.Exit(0)

    success = client.terminate_instance(instance_id, provider)

    if success:
        logger.info(f"✓ Instance {instance_id} terminated")
    else:
        logger.error("✗ Failed to terminate instance")
        raise typer.Exit(1)


@app.command()
def cleanup(
    ctx: typer.Context,
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation"),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Only terminate instances from this provider"
    ),
    exclude: Optional[List[str]] = typer.Option(
        None, "--exclude", help="Instance IDs to exclude from cleanup (can be specified multiple times)"
    ),
):
    """Terminate all running GPU instances

    This command lists all your instances across all configured providers
    and terminates them. Use with caution!

    Examples:
        broker cleanup --exclude abc123 --exclude def456
        broker cleanup --exclude abc123 def456 ghi789
    """
    creds = resolve_credentials(ctx)
    ssh_key = resolve_ssh_key(ctx)

    client = GPUClient(credentials=creds, ssh_key_path=ssh_key)

    # Get all instances
    logger.info("Fetching all instances...")
    instances = client.list_instances()

    # Filter by provider if specified
    if provider:
        instances = [i for i in instances if i.provider == provider]

    # Exclude specified instances
    if exclude:
        exclude_set = set(exclude)
        instances = [i for i in instances if i.id not in exclude_set]
        if exclude_set:
            logger.info(f"Excluding {len(exclude_set)} instance(s) from cleanup")

    if not instances:
        if provider:
            logger.info(f"No instances found in {provider}")
        else:
            logger.info("No instances found")
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
            logger.info("Cancelled")
            raise typer.Exit(0)

    # Terminate all instances
    failed = []
    succeeded = []

    for instance in instances:
        logger.info(f"Terminating {instance.id} ({instance.provider})...")
        success = client.terminate_instance(instance.id, instance.provider)

        if success:
            succeeded.append(instance)
        else:
            failed.append(instance)

    # Report results
    if succeeded:
        logger.info(f"✓ Successfully terminated {len(succeeded)} instance(s)")

    if failed:
        logger.error(f"✗ Failed to terminate {len(failed)} instance(s):")
        for instance in failed:
            logger.error(f"  - {instance.id} ({instance.provider})")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
