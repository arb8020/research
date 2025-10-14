# CLI Implementation Plan

**Date**: 2025-10-13
**Status**: Ready for Implementation
**Philosophy**: Minimal, explicit, .env-first configuration following Tiger Style principles

---

## Design Decisions Summary

### Configuration
- **Approach**: `.env` file in current directory (no hidden config directories)
- **Precedence**: CLI flags → Environment variables → .env file → Error
- **Commands**: `broker init` / `bifrost init` create .env templates

### Key Principles
1. **Explicit over implicit**: User controls when to wait (`--wait-ssh`), which credentials to use
2. **Minimal by default**: Return immediately unless asked to wait, query all providers unless filtered
3. **Logging via Rich**: Use logging module with RichHandler for all output
4. **Thin wrapper**: CLI is a clean layer over existing SDKs (zero SDK changes needed)

### Confirmed Behaviors
- **Default image**: Keep current SDK default (`runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204`)
- **Wait behavior**: Return immediately by default, explicit `--wait-ssh` to block
- **Bootstrap**: Multiple `--bootstrap` flags joined with `&&` (note: could be improved)
- **Job IDs**: Use SDK's `generate_job_id()`, optional `--name` for human-readable
- **SSH formats**: Accept all variants (user@host:port, ssh -p port user@host, etc.)
- **Provider search**: Query all configured providers, merge results
- **SSH key discovery**: Show helpful error with found keys, require explicit configuration

---

## Part 1: Configuration Management

**File: `shared/config.py`**

Manages .env-based configuration without hidden config directories.

```python
"""
Configuration using .env files - no hidden config directories.

Precedence:
1. CLI flags (highest priority)
2. Environment variables
3. .env file in current directory (via python-dotenv)
4. Error with helpful message
"""

from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
import os

# Load .env on import
load_dotenv()

def get_runpod_key() -> Optional[str]:
    """Get RunPod API key from environment"""
    return os.getenv("RUNPOD_API_KEY")

def get_vast_key() -> Optional[str]:
    """Get Vast API key from environment"""
    return os.getenv("VAST_API_KEY")

def get_ssh_key_path() -> Optional[str]:
    """Get SSH key path from environment"""
    return os.getenv("SSH_KEY_PATH")

def discover_ssh_keys() -> List[str]:
    """Find SSH keys in common locations"""
    common_paths = [
        Path.home() / ".ssh" / "id_ed25519",
        Path.home() / ".ssh" / "id_rsa",
        Path.home() / ".ssh" / "id_ecdsa",
    ]
    return [str(p) for p in common_paths if p.exists()]

def create_env_template(tool: str):
    """Create .env template for broker or bifrost

    Args:
        tool: "broker" or "bifrost"

    Raises:
        FileExistsError: If .env already exists
    """
    if Path(".env").exists():
        raise FileExistsError(".env already exists")

    if tool == "broker":
        template = """# GPU Broker Credentials
RUNPOD_API_KEY=
VAST_API_KEY=
SSH_KEY_PATH=~/.ssh/id_ed25519
"""
    else:  # bifrost
        template = """# Bifrost SSH Configuration
SSH_KEY_PATH=~/.ssh/id_ed25519
"""

    Path(".env").write_text(template)
    Path(".env").chmod(0o600)  # Secure permissions
```

---

## Part 2: Broker CLI

**File: `broker/broker/cli.py`**

Thin CLI wrapper over broker SDK.

### Commands Structure

```
broker
├── init           # Create .env template
├── search         # Search GPU offers
├── create         # Provision instance
├── list           # List instances
├── status         # Get instance status
├── ssh            # Get SSH connection string
└── terminate      # Terminate instance
```

### Implementation

**1. Setup logging with Rich handler**

```python
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
import typer
from typing import Optional, List
from pathlib import Path
import sys

from shared.config import (
    get_runpod_key, get_vast_key, get_ssh_key_path,
    discover_ssh_keys, create_env_template
)
from broker.client import GPUClient
from broker.types import ProviderCredentials

console = Console()
app = typer.Typer(help="GPU broker - provision cloud GPUs")

# Logger will be configured in callback
logger = logging.getLogger("broker")
```

**2. Main callback (global options)**

```python
@app.callback()
def main(
    ctx: typer.Context,
    credentials: Optional[str] = typer.Option(None, "--credentials",
        help="Credentials file or inline 'runpod:key,vast:key'"),
    ssh_key: Optional[str] = typer.Option(None, "--ssh-key",
        help="Path to SSH private key"),
    quiet: bool = typer.Option(False, "-q", "--quiet",
        help="Show only warnings and errors"),
    json_output: bool = typer.Option(False, "--json",
        help="Output results as JSON"),
    debug: bool = typer.Option(False, "--debug",
        help="Show debug logs"),
):
    """Configure logging and store global options"""

    # Setup logging based on flags
    if json_output:
        # Suppress all logging when outputting JSON
        logging.basicConfig(level=logging.CRITICAL)
    elif debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            handlers=[RichHandler(console=console, rich_tracebacks=True)]
        )
    elif quiet:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(message)s",
            handlers=[RichHandler(console=console)]
        )
    else:
        # Default: INFO level
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(console=console)]
        )

    # Store options in context
    ctx.obj = {
        "credentials": credentials,
        "ssh_key": ssh_key,
        "json": json_output
    }
```

**3. Init command**

```python
@app.command()
def init():
    """Create .env template for credentials

    Creates a .env file in the current directory with template
    for RunPod/Vast API keys and SSH key path.
    """
    try:
        create_env_template("broker")
        logger.info("✓ Created .env with credential template")
        logger.info("")
        logger.info("Edit .env with your API keys:")
        logger.info("  RUNPOD_API_KEY=your_key_here")
        logger.info("  VAST_API_KEY=your_key_here")
        logger.info("  SSH_KEY_PATH=~/.ssh/id_ed25519")
        logger.info("")
        logger.info("Then run: broker search")
    except FileExistsError:
        logger.error("✗ .env already exists")
        logger.info("Edit manually or delete to recreate")
        raise typer.Exit(1)
```

**4. Credential resolution helper**

```python
def resolve_credentials(ctx) -> ProviderCredentials:
    """Resolve credentials from CLI → env → .env → error

    Priority:
    1. --credentials flag (file path or inline format)
    2. Environment variables (RUNPOD_API_KEY, VAST_API_KEY)
    3. .env file (loaded by python-dotenv)
    4. Error with helpful message
    """
    creds_arg = ctx.obj.get("credentials")

    # Priority 1: CLI flag
    if creds_arg:
        if Path(creds_arg).exists():
            # File path
            import json
            with open(creds_arg) as f:
                creds_dict = json.load(f)
            return ProviderCredentials.from_dict(creds_dict)
        else:
            # Inline format: "runpod:key,vast:key"
            parts = creds_arg.split(',')
            creds_dict = {}
            for part in parts:
                if ':' not in part:
                    logger.error(f"✗ Invalid credentials format: {part}")
                    logger.info("Expected: runpod:key,vast:key")
                    raise typer.Exit(1)
                provider, key = part.split(':', 1)
                creds_dict[provider.strip()] = key.strip()
            return ProviderCredentials.from_dict(creds_dict)

    # Priority 2+3: Environment variables (includes .env via load_dotenv)
    runpod_key = get_runpod_key()
    vast_key = get_vast_key()

    if runpod_key or vast_key:
        return ProviderCredentials(runpod=runpod_key, vast=vast_key)

    # Priority 4: Error with helpful message
    logger.error("✗ No credentials found")
    logger.info("")
    logger.info("Try: broker init")
    logger.info("Or: export RUNPOD_API_KEY=... VAST_API_KEY=...")
    logger.info("Or: --credentials <file|runpod:key,vast:key>")
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
```

**5. Search command**

```python
@app.command()
def search(
    ctx: typer.Context,
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type",
        help="Filter by GPU type (e.g., 'A100')"),
    max_price: Optional[float] = typer.Option(None, "--max-price",
        help="Maximum price per hour"),
    min_vram: Optional[int] = typer.Option(None, "--min-vram",
        help="Minimum VRAM in GB"),
    provider: Optional[str] = typer.Option(None, "--provider",
        help="Filter by provider (runpod|vast)"),
    limit: int = typer.Option(10, "--limit",
        help="Maximum number of results"),
):
    """Search for available GPU offers

    By default searches all configured providers and merges results.
    Use --provider to filter to specific provider.
    """
    creds = resolve_credentials(ctx)

    # Create client (SSH key not needed for search - pass None)
    # NOTE: This requires GPUClient to make ssh_key_path optional
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
        import json
        print(json.dumps([{
            "provider": o.provider,
            "gpu_type": o.gpu_type,
            "vram_gb": o.vram_gb,
            "price_per_hour": o.price_per_hour,
            "memory_gb": o.memory_gb,
        } for o in offers], indent=2))
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
                f"${offer.price_per_hour:.2f}"
            )

        console.print(table)
        logger.info(f"Showing {len(offers)} cheapest offers")
```

**6. Create command**

```python
@app.command()
def create(
    ctx: typer.Context,
    gpu_type: Optional[str] = typer.Option(None, "--gpu-type",
        help="GPU type filter"),
    max_price: Optional[float] = typer.Option(None, "--max-price",
        help="Maximum price per hour"),
    image: str = typer.Option("runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204", "--image",
        help="Docker image to use"),
    name: Optional[str] = typer.Option(None, "--name",
        help="Instance name"),
    wait_ssh: bool = typer.Option(False, "--wait-ssh",
        help="Wait for SSH to be ready before returning"),
    output: str = typer.Option("summary", "--output",
        help="Output format: summary|ssh|json"),
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
        import json
        print(json.dumps({
            "id": instance.id,
            "provider": instance.provider,
            "gpu_type": instance.gpu_type,
            "status": instance.status.value,
            "ssh": instance.ssh_connection_string() if wait_ssh else None,
            "price_per_hour": instance.price_per_hour
        }, indent=2))
    elif output == "ssh":
        if wait_ssh:
            print(instance.ssh_connection_string())
        else:
            logger.error("✗ Cannot output SSH - use --wait-ssh")
            raise typer.Exit(1)
    else:  # summary
        if wait_ssh:
            logger.info(f"✓ Instance {instance.id} ready: {instance.ssh_connection_string()}")
        else:
            logger.info(f"✓ Instance {instance.id} provisioning started")
            logger.info(f"Use 'broker status {instance.id}' to check progress")
```

**7. List, Status, SSH, Terminate commands**

```python
@app.command()
def list(ctx: typer.Context):
    """List all your GPU instances"""
    creds = resolve_credentials(ctx)
    ssh_key = resolve_ssh_key(ctx)

    client = GPUClient(credentials=creds, ssh_key_path=ssh_key)
    instances = client.list_instances()

    if ctx.obj["json"]:
        import json
        print(json.dumps([{
            "id": i.id,
            "provider": i.provider,
            "gpu_type": i.gpu_type,
            "status": i.status.value,
            "price_per_hour": i.price_per_hour
        } for i in instances], indent=2))
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
                instance.id[:12],
                instance.provider,
                instance.gpu_type,
                instance.status.value,
                f"${instance.price_per_hour:.2f}"
            )

        console.print(table)

@app.command()
def status(
    ctx: typer.Context,
    instance_id: str = typer.Argument(..., help="Instance ID"),
    provider: Optional[str] = typer.Argument(None, help="Provider (runpod|vast). Auto-detect if omitted."),
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
        import json
        print(json.dumps({
            "id": instance.id,
            "provider": instance.provider,
            "status": instance.status.value,
            "gpu_type": instance.gpu_type,
            "ssh": instance.ssh_connection_string() if instance.public_ip else None,
        }, indent=2))
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
    provider: Optional[str] = typer.Argument(None, help="Provider (runpod|vast). Auto-detect if omitted."),
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
        logger.error(f"✗ Instance not ready")
        raise typer.Exit(1)

    # Always output just the connection string (for piping)
    print(instance.ssh_connection_string())

@app.command()
def terminate(
    ctx: typer.Context,
    instance_id: str = typer.Argument(...),
    provider: Optional[str] = typer.Argument(None, help="Provider (runpod|vast). Auto-detect if omitted."),
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
        logger.error(f"✗ Failed to terminate instance")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
```

---

## Part 3: Bifrost CLI

**File: `bifrost/bifrost/cli.py`**

Thin CLI wrapper over bifrost SDK.

### Commands Structure

```
bifrost
├── init           # Create .env template
├── push           # Deploy code
├── exec           # Execute command
├── deploy         # Push + exec (convenience)
├── run            # Detached execution
├── jobs           # List jobs
├── logs           # Show/follow logs
├── download       # Download files
└── upload         # Upload files
```

### Implementation

**1. Setup and helpers**

```python
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
import typer
from typing import Optional, List, Dict
from pathlib import Path
import sys

from shared.config import get_ssh_key_path, discover_ssh_keys, create_env_template
from bifrost.client import BifrostClient
from bifrost.types import JobStatus

console = Console()
app = typer.Typer(help="Bifrost - remote GPU execution")
logger = logging.getLogger("bifrost")

def parse_ssh_connection(conn_str: str) -> tuple[str, str, int]:
    """Parse SSH connection string

    Accepts formats:
      - user@host:port
      - user@host (default port 22)
      - ssh -p port user@host
      - ssh user@host
    """
    import re

    # Remove 'ssh' prefix if present
    conn_str = conn_str.strip()
    if conn_str.startswith('ssh '):
        # Extract from ssh command
        # ssh -p PORT user@host or ssh user@host
        match = re.match(r'ssh\s+(?:-p\s+(\d+)\s+)?([^@]+)@(\S+)', conn_str)
        if match:
            port, user, host = match.groups()
            return user, host, int(port) if port else 22

    # Standard format: user@host:port or user@host
    match = re.match(r'^([^@]+)@([^:]+)(?::(\d+))?$', conn_str)
    if match:
        user, host, port = match.groups()
        return user, host, int(port) if port else 22

    raise ValueError(
        f"Invalid SSH format: {conn_str}\n"
        f"Accepted formats:\n"
        f"  user@host:port\n"
        f"  user@host (default port 22)\n"
        f"  ssh -p port user@host\n"
        f"  ssh user@host"
    )

def resolve_ssh_key(ctx) -> str:
    """Resolve SSH key: CLI → env → discover → error"""
    ssh_key_arg = ctx.obj.get("ssh_key")

    if ssh_key_arg:
        return ssh_key_arg

    if key_path := get_ssh_key_path():
        return key_path

    # Discovery with helpful error
    found_keys = discover_ssh_keys()

    logger.error("✗ No SSH key specified")
    logger.info("")
    if found_keys:
        logger.info("Found keys at:")
        for key in found_keys:
            logger.info(f"  {key}")
        logger.info("")
        logger.info(f"Or use: --ssh-key {found_keys[0]}")
    else:
        logger.info("No SSH keys found in ~/.ssh/")
        logger.info("Generate one: ssh-keygen -t ed25519")
        logger.info("")
    logger.info("Set SSH_KEY_PATH in .env (run: bifrost init)")

    raise typer.Exit(1)

def parse_env_vars(env_list: List[str]) -> Dict[str, str]:
    """Parse environment variables from KEY=VALUE format

    Raises:
        typer.Exit: If format is invalid (caught by CLI commands)
    """
    env_dict = {}
    for item in env_list:
        if '=' not in item:
            logger.error(f"✗ Invalid env format: {item}")
            logger.info("Expected format: KEY=VALUE")
            logger.info("Example: --env API_KEY=abc123 --env DEBUG=true")
            raise typer.Exit(1)
        key, value = item.split('=', 1)
        env_dict[key] = value
    return env_dict
```

**2. Main callback**

```python
@app.callback()
def main(
    ctx: typer.Context,
    ssh_key: Optional[str] = typer.Option(None, "--ssh-key",
        help="Path to SSH private key"),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
    json_output: bool = typer.Option(False, "--json"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Configure logging and store global options"""

    # Setup logging
    if json_output:
        logging.basicConfig(level=logging.CRITICAL)
    elif debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            handlers=[RichHandler(console=console, rich_tracebacks=True)]
        )
    elif quiet:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(message)s",
            handlers=[RichHandler(console=console)]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(console=console)]
        )

    ctx.obj = {
        "ssh_key": ssh_key,
        "json": json_output
    }
```

**3. Init command**

```python
@app.command()
def init():
    """Create .env template for SSH configuration"""
    try:
        create_env_template("bifrost")
        logger.info("✓ Created .env with SSH configuration template")
        logger.info("")
        logger.info("Edit .env with your SSH key path:")
        logger.info("  SSH_KEY_PATH=~/.ssh/id_ed25519")
        logger.info("")
        logger.info("Then run: bifrost push <ssh-connection>")
    except FileExistsError:
        logger.error("✗ .env already exists")
        logger.info("Edit manually or delete to recreate")
        raise typer.Exit(1)
```

**4. Push command**

```python
@app.command()
def push(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(..., help="SSH connection (user@host:port)"),
    bootstrap: Optional[List[str]] = typer.Option(None, "--bootstrap",
        help="Bootstrap command (can specify multiple, joined with &&)"),
    bootstrap_script: Optional[str] = typer.Option(None, "--bootstrap-script",
        help="Path to bootstrap script to upload and execute"),
):
    """Deploy code to remote instance

    Bootstrap options (choose one):
    1. Inline command: --bootstrap "cmd1 && cmd2"
    2. Multiple commands: --bootstrap "cmd1" --bootstrap "cmd2" (joined with &&)
    3. Script file: --bootstrap-script script.sh (uploaded and executed)

    Examples:
      bifrost push user@host:22 --bootstrap "pip install uv && uv sync"
      bifrost push user@host:22 --bootstrap "pip install uv" --bootstrap "uv sync"
      bifrost push user@host:22 --bootstrap-script setup.sh
    """
    ssh_key = resolve_ssh_key(ctx)

    # Validate bootstrap options
    if bootstrap and bootstrap_script:
        logger.error("✗ Cannot use both --bootstrap and --bootstrap-script")
        raise typer.Exit(1)

    # Parse SSH connection
    try:
        user, host, port = parse_ssh_connection(ssh_connection)
        ssh_conn_str = f"{user}@{host}:{port}"
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(1)

    # Prepare bootstrap command
    bootstrap_cmd = None
    if bootstrap_script:
        # Upload script and execute it
        script_path = Path(bootstrap_script)
        if not script_path.exists():
            logger.error(f"✗ Bootstrap script not found: {bootstrap_script}")
            raise typer.Exit(1)

        # Read script content
        bootstrap_cmd = script_path.read_text()
        logger.debug(f"Bootstrap script: {bootstrap_script}")
    elif bootstrap:
        # Join multiple bootstrap commands with &&
        bootstrap_cmd = " && ".join(bootstrap)
        logger.debug(f"Bootstrap: {bootstrap_cmd}")

    # Create client and push
    client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)

    logger.info("Deploying code...")
    workspace_path = client.push(bootstrap_cmd=bootstrap_cmd)

    logger.info(f"✓ Code deployed to {workspace_path}")
```

**5. Exec command**

```python
@app.command()
def exec(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    command: str = typer.Argument(..., help="Command to execute"),
    env: Optional[List[str]] = typer.Option(None, "--env", help="KEY=VALUE"),
):
    """Execute command on remote instance

    To run in a specific directory, use: bifrost exec conn "cd /path && cmd"
    """
    ssh_key = resolve_ssh_key(ctx)

    try:
        user, host, port = parse_ssh_connection(ssh_connection)
        ssh_conn_str = f"{user}@{host}:{port}"
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(1)

    # Parse env vars
    env_dict = parse_env_vars(env) if env else None

    # Create client and execute
    client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)

    logger.info(f"Executing: {command}")
    result = client.exec(command, env=env_dict)

    # Output
    if ctx.obj["json"]:
        import json
        print(json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code
        }, indent=2))
    else:
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.exit_code != 0:
            logger.error(result.stderr)

        if result.exit_code != 0:
            logger.error(f"✗ Command failed with exit code {result.exit_code}")
            raise typer.Exit(result.exit_code)
        else:
            logger.info("✓ Command completed successfully")
```

**6. Deploy command (convenience)**

```python
@app.command()
def deploy(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    command: str = typer.Argument(...),
    bootstrap: Optional[List[str]] = typer.Option(None, "--bootstrap"),
    bootstrap_script: Optional[str] = typer.Option(None, "--bootstrap-script"),
    env: Optional[List[str]] = typer.Option(None, "--env"),
):
    """Deploy code and execute command (convenience: push + exec)"""
    ssh_key = resolve_ssh_key(ctx)

    # Validate bootstrap options
    if bootstrap and bootstrap_script:
        logger.error("✗ Cannot use both --bootstrap and --bootstrap-script")
        raise typer.Exit(1)

    try:
        user, host, port = parse_ssh_connection(ssh_connection)
        ssh_conn_str = f"{user}@{host}:{port}"
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(1)

    # Parse inputs
    bootstrap_cmd = None
    if bootstrap_script:
        script_path = Path(bootstrap_script)
        if not script_path.exists():
            logger.error(f"✗ Bootstrap script not found: {bootstrap_script}")
            raise typer.Exit(1)
        bootstrap_cmd = script_path.read_text()
    elif bootstrap:
        bootstrap_cmd = " && ".join(bootstrap)

    env_dict = parse_env_vars(env) if env else None

    # Create client and deploy
    client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)

    logger.info("Deploying code and executing command...")
    result = client.deploy(command, bootstrap_cmd=bootstrap_cmd, env=env_dict)

    # Output
    if ctx.obj["json"]:
        import json
        print(json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code
        }, indent=2))
    else:
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.exit_code != 0:
            logger.error(result.stderr)

        if result.exit_code != 0:
            logger.error(f"✗ Command failed")
            raise typer.Exit(result.exit_code)
        else:
            logger.info("✓ Deploy and execution completed successfully")
```

**7. Run command (detached)**

```python
@app.command()
def run(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    command: str = typer.Argument(...),
    bootstrap: Optional[List[str]] = typer.Option(None, "--bootstrap"),
    bootstrap_script: Optional[str] = typer.Option(None, "--bootstrap-script"),
    env: Optional[List[str]] = typer.Option(None, "--env"),
    name: Optional[str] = typer.Option(None, "--name",
        help="Human-readable job name"),
):
    """Run command in background (detached mode)

    Job ID generation:
      - Without --name: random ID (abc123def456)
      - With --name: name + random (my-job-abc123)

    The job will continue running even if SSH disconnects.
    Use 'bifrost jobs' to monitor and 'bifrost logs' to view output.
    """
    ssh_key = resolve_ssh_key(ctx)

    # Validate bootstrap options
    if bootstrap and bootstrap_script:
        logger.error("✗ Cannot use both --bootstrap and --bootstrap-script")
        raise typer.Exit(1)

    try:
        user, host, port = parse_ssh_connection(ssh_connection)
        ssh_conn_str = f"{user}@{host}:{port}"
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(1)

    # Parse inputs
    bootstrap_cmd = None
    if bootstrap_script:
        script_path = Path(bootstrap_script)
        if not script_path.exists():
            logger.error(f"✗ Bootstrap script not found: {bootstrap_script}")
            raise typer.Exit(1)
        bootstrap_cmd = script_path.read_text()
    elif bootstrap:
        bootstrap_cmd = " && ".join(bootstrap)

    env_dict = parse_env_vars(env) if env else None

    # Create client and run detached
    client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)

    logger.info("Starting detached job...")
    job_info = client.run_detached(
        command=command,
        bootstrap_cmd=bootstrap_cmd,
        env=env_dict,
        session_name=name
    )

    if ctx.obj["json"]:
        import json
        print(json.dumps({
            "job_id": job_info.job_id,
            "status": job_info.status.value,
            "command": job_info.command
        }, indent=2))
    else:
        logger.info(f"✓ Job {job_info.job_id} started")
        logger.info(f"Monitor: bifrost logs {ssh_connection} {job_info.job_id} --follow")
```

**8. Jobs and Logs commands**

```python
@app.command()
def jobs(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
):
    """List all jobs on remote instance"""
    ssh_key = resolve_ssh_key(ctx)

    try:
        user, host, port = parse_ssh_connection(ssh_connection)
        ssh_conn_str = f"{user}@{host}:{port}"
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(1)

    client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)
    jobs = client.get_all_jobs()

    if ctx.obj["json"]:
        import json
        print(json.dumps([{
            "job_id": j.job_id,
            "status": j.status.value,
            "command": j.command,
            "start_time": j.start_time.isoformat() if j.start_time else None,
        } for j in jobs], indent=2))
    else:
        if not jobs:
            logger.info("No jobs found")
            return

        table = Table(title=f"Jobs on {ssh_connection}")
        table.add_column("Job ID", style="cyan")
        table.add_column("Status")
        table.add_column("Command", max_width=40)
        table.add_column("Runtime")

        for job in jobs:
            status_style = "green" if job.status == JobStatus.COMPLETED else "yellow"
            runtime = f"{int(job.runtime_seconds)}s" if job.runtime_seconds else "N/A"

            table.add_row(
                job.job_id,
                f"[{status_style}]{job.status.value}[/{status_style}]",
                job.command[:40] + "..." if len(job.command) > 40 else job.command,
                runtime
            )

        console.print(table)

@app.command()
def logs(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    job_id: str = typer.Argument(...),
    follow: bool = typer.Option(False, "-f", "--follow",
        help="Follow logs in real-time (like tail -f)"),
    lines: int = typer.Option(100, "-n", help="Number of lines to show"),
):
    """Show job logs"""
    ssh_key = resolve_ssh_key(ctx)

    try:
        user, host, port = parse_ssh_connection(ssh_connection)
        ssh_conn_str = f"{user}@{host}:{port}"
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(1)

    client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)

    if follow:
        logger.info(f"Following logs for {job_id} (Ctrl+C to exit)...")
        try:
            for line in client.follow_job_logs(job_id):
                print(line)
        except KeyboardInterrupt:
            logger.info("\nStopped following logs")
    else:
        logs = client.get_logs(job_id, lines=lines)
        print(logs)
```

**9. Download/Upload commands**

```python
@app.command()
def download(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    remote_path: str = typer.Argument(...),
    local_path: str = typer.Argument(...),
    recursive: bool = typer.Option(False, "-r", "--recursive"),
):
    """Download files from remote to local"""
    ssh_key = resolve_ssh_key(ctx)

    try:
        user, host, port = parse_ssh_connection(ssh_connection)
        ssh_conn_str = f"{user}@{host}:{port}"
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(1)

    client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)

    logger.info(f"Downloading {remote_path}...")
    result = client.download_files(remote_path, local_path, recursive=recursive)

    if result.success:
        logger.info(f"✓ Downloaded {result.files_copied} files ({result.total_bytes} bytes)")
    else:
        logger.error(f"✗ Download failed: {result.error_message}")
        raise typer.Exit(1)

@app.command()
def upload(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    local_path: str = typer.Argument(...),
    remote_path: str = typer.Argument(...),
    recursive: bool = typer.Option(False, "-r", "--recursive"),
):
    """Upload files from local to remote"""
    ssh_key = resolve_ssh_key(ctx)

    try:
        user, host, port = parse_ssh_connection(ssh_connection)
        ssh_conn_str = f"{user}@{host}:{port}"
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(1)

    client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)

    logger.info(f"Uploading {local_path}...")
    result = client.upload_files(local_path, remote_path, recursive=recursive)

    if result.success:
        logger.info(f"✓ Uploaded {result.files_copied} files ({result.total_bytes} bytes)")
    else:
        logger.error(f"✗ Upload failed: {result.error_message}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
```

---

## Part 4: Dependencies

Add to both `broker/pyproject.toml` and `bifrost/pyproject.toml`:

```toml
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "python-dotenv>=1.0.0",
    # ... existing dependencies
]
```

---

## Part 5: Implementation Order

1. **Phase 1: Shared infrastructure**
   - Create `shared/config.py`
   - Create `shared/cli_utils.py` (if needed for shared helpers)
   - Add `python-dotenv` to dependencies

2. **Phase 2: Broker CLI**
   - Implement `broker/broker/cli.py`
   - Test: `broker init`, `broker search`, `broker create`
   - Test: `broker list`, `broker status`, `broker ssh`, `broker terminate`

3. **Phase 3: Bifrost CLI**
   - Implement `bifrost/bifrost/cli.py`
   - Test: `bifrost init`, `bifrost push`, `bifrost exec`, `bifrost deploy`
   - Test: `bifrost run`, `bifrost jobs`, `bifrost logs`
   - Test: `bifrost download`, `bifrost upload`

4. **Phase 4: Integration testing**
   - End-to-end: `broker create` → `bifrost deploy`
   - Test all output formats (summary, json, ssh)
   - Test all logging levels (normal, quiet, debug)
   - Update examples to use new CLI

---

## Part 6: Testing Strategy

### Manual Integration Tests

```bash
# Test broker
broker init
# Edit .env with credentials
broker search --gpu-type A100
broker create --gpu-type A100 --wait-ssh --output ssh > ssh.txt
broker list

# Test with auto-detection (convenient)
broker status <instance-id>
broker ssh <instance-id>

# Test with explicit provider (for scripts)
broker status <instance-id> runpod
broker ssh <instance-id> runpod

# Test bifrost
bifrost init
bifrost deploy $(cat ssh.txt) "python -c 'print(\"hello\")'"
bifrost run $(cat ssh.txt) "sleep 60" --name test-job
bifrost jobs $(cat ssh.txt)
bifrost logs $(cat ssh.txt) <job-id> --follow

# Test bootstrap variants
bifrost push $(cat ssh.txt) --bootstrap "pip install uv && uv sync"
bifrost push $(cat ssh.txt) --bootstrap "pip install uv" --bootstrap "uv sync"
bifrost push $(cat ssh.txt) --bootstrap-script setup.sh

# Cleanup
broker terminate <instance-id>  # Auto-detect
broker terminate <instance-id> runpod  # Explicit
```

### Edge Cases

- Missing credentials → helpful error with suggestions
- Invalid SSH connection format → parse error with accepted formats
- Ambiguous instance ID → error listing providers, suggest explicit syntax
- Instance not ready → wait timeout with status info
- Job not found → clear error message
- Invalid env var format → helpful error with examples
- Network failures → appropriate error messages
- Bootstrap script not found → clear error with path

---

## Part 7: Notes

### Required SDK Changes

1. **GPUClient ssh_key_path parameter**: Make `ssh_key_path` optional (default to `None`). Operations like `search()` don't require SSH access and shouldn't be coupled to SSH key resolution. Only operations that actually provision or connect to instances should require SSH keys.

   **Rationale** (Casey Muratori - decoupling): Searching for offers and connecting to instances are separate concerns. Don't couple operations to resources they don't actually need.

### Design Decisions

1. **Provider argument is optional** for `status`, `ssh`, and `terminate` commands. Auto-detection is performed by querying all providers. If ambiguous (same ID in multiple providers), fails with clear error message.

   **Rationale** (Casey Muratori - redundancy for convenience): Provide both convenient path (auto-detect) and explicit path (specify provider). Stateless design - no retained mappings, just query on-demand. Tiger Style - fail fast with helpful error messages.

2. **Bootstrap supports three approaches**:
   - Single inline command: `--bootstrap "cmd1 && cmd2"`
   - Multiple flags: `--bootstrap "cmd1" --bootstrap "cmd2"` (joined with `&&`)
   - Script file: `--bootstrap-script script.sh`

   **Rationale** (Casey Muratori - granular tiers): Different users need different levels of control. Simple cases use inline, complex setups use scripts. All three are valid, no hidden transformations.

3. **Working directory via explicit cd syntax**: The CLI does not provide a `--dir` working directory flag for `exec`. Users should use `"cd /path && cmd"` syntax explicitly.

   **Rationale** (Tiger Style - transparent, explicit): Avoid leaky abstractions where CLI secretly manipulates command strings. Keep the interface honest about what it does.

4. **Error handling uses helpful messages**: All error paths provide context and suggest solutions (e.g., show found SSH keys, suggest which command to run).

   **Rationale** (Tiger Style - fail fast, helpful errors): Errors should guide users toward solutions, not just report problems.

### Known Limitations

1. **SSH key discovery**: Currently tries common paths but doesn't handle multiple keys intelligently. Shows all found keys in error message.

2. **Provider credentials**: Inline format (`runpod:key,vast:key`) is basic. Could support more formats in future.

### Future Improvements

1. **Config profiles**: Support multiple .env files or config profiles for different projects/accounts
2. **Interactive mode**: Add `broker search --interactive` for guided provisioning
3. **Bulk operations**: Add `broker terminate --all` or `broker list --terminate-idle`
4. **Better SSH handling**: Auto-detect SSH key type, handle passphrases

---

## Summary

This implementation provides minimal, explicit CLIs that are thin wrappers over the existing SDKs. The design follows Tiger Style and Casey Muratori's component API principles:

**Tiger Style Principles Applied:**
- **Explicit over implicit**: User controls when to wait, which credentials to use
- **Transparent state**: All config in visible .env file (no hidden config directories)
- **Fail fast with helpful errors**: Clear messages with suggested solutions, explicit validation
- **Simple control flow**: Thin wrapper, minimal abstractions

**Casey Muratori Principles Applied:**
- **Granular tiers**: Multiple bootstrap options (inline, flags, script) for different use cases
- **Redundancy for convenience**: Optional provider auto-detection with explicit fallback
- **Decoupling**: Search doesn't require SSH keys; operations only depend on what they need
- **Stateless over retained mode**: No local state/mappings; queries on-demand

**Sean Goedecke Principles Applied:**
- **Minimize state**: No persistent mappings, query providers directly
- **Boring is good**: Standard .env files, no clever tricks
- **Hot paths stay simple**: Common operations (status, SSH) are convenient by default

**Key Features:**
- Provider auto-detection with clear ambiguity handling
- Multiple bootstrap approaches (command, flags, script)
- Explicit error messages that guide users to solutions
- JSON output mode for scripting/automation
- Composable design supports piping between tools

**Implementation Requirements:**
- **One SDK change**: Make `GPUClient.ssh_key_path` optional
- **One new dependency**: `python-dotenv` for .env file support
- **New CLI files**: `broker/broker/cli.py`, `bifrost/bifrost/cli.py`, `shared/config.py`
