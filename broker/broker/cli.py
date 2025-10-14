"""Broker CLI - GPU provisioning and management"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from broker.client import GPUClient
from broker.types import ProviderCredentials
from shared.config import (
    create_env_template,
    discover_ssh_keys,
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

    if runpod_key or prime_key:
        return ProviderCredentials(runpod=runpod_key or "", primeintellect=prime_key or "")

    # Priority 4: Error with helpful message
    logger.error("✗ No credentials found")
    logger.info("")
    logger.info("Try: broker init")
    logger.info("Or: export RUNPOD_API_KEY=... PRIME_API_KEY=...")
    logger.info("Or: --credentials <file|runpod:key,primeintellect:key>")
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
    max_price: Optional[float] = typer.Option(
        None, "--max-price", help="Maximum price per hour"
    ),
    min_vram: Optional[int] = typer.Option(None, "--min-vram", help="Minimum VRAM in GB"),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Filter by provider (runpod|primeintellect)"
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

    # Build query
    query = None
    if gpu_type:
        query = client.gpu_type.contains(gpu_type)
    if max_price:
        price_filter = client.price_per_hour <= max_price
        query = price_filter if query is None else query & price_filter
    if min_vram:
        vram_filter = client.vram_gb >= min_vram
        query = vram_filter if query is None else query & vram_filter
    if provider:
        provider_filter = client.provider == provider
        query = provider_filter if query is None else query & provider_filter

    # Search (queries all providers by default, merges results)
    logger.info("Searching for GPU offers...")
    offers = client.search(query, sort=lambda x: x.price_per_hour)

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
                        "memory_gb": o.memory_gb,
                    }
                    for o in offers
                ],
                indent=2,
            )
        )
    else:
        # Rich table output
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
        logger.info(f"Showing {len(offers)} cheapest offers")


@app.command()
def create(
    ctx: typer.Context,
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type", help="GPU type filter"),
    max_price: Optional[float] = typer.Option(
        None, "--max-price", help="Maximum price per hour"
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

    # Build query if filters provided
    query = None
    if gpu_type:
        query = client.gpu_type.contains(gpu_type)
    if max_price:
        price_filter = client.price_per_hour <= max_price
        query = price_filter if query is None else query & price_filter

    # Create instance
    logger.info("Provisioning instance...")
    instance = client.create(query, image=image, name=name)

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
                    "status": instance.status.value,
                    "ssh": instance.ssh_connection_string() if wait_ssh else None,
                    "price_per_hour": instance.price_per_hour,
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
            logger.info(
                f"✓ Instance {instance.id} ready: {instance.ssh_connection_string()}"
            )
        else:
            logger.info(f"✓ Instance {instance.id} provisioning started")
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
        table.add_column("ID", style="cyan")
        table.add_column("Provider")
        table.add_column("GPU")
        table.add_column("Status")
        table.add_column("Price/hr", justify="right")

        for instance in instances:
            table.add_row(
                instance.id,
                instance.provider,
                instance.gpu_type,
                instance.status.value,
                f"${instance.price_per_hour:.2f}",
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

    # Always output just the connection string (for piping)
    print(instance.ssh_connection_string())


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


if __name__ == "__main__":
    app()
