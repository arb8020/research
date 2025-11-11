# Bifrost SSH Key Handling - Code Snippets and Locations

## 1. Global Callback (Where --ssh-key is Defined)

**File:** `/Users/chiraagbalu/research/bifrost/bifrost/cli.py`  
**Lines:** 106-129

```python
@app.callback()
def main(
    ctx: typer.Context,
    ssh_key: Optional[str] = typer.Option(
        None, "--ssh-key", help="Path to SSH private key"
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet"),
    json_output: bool = typer.Option(False, "--json"),
    debug: bool = typer.Option(False, "--debug"),
):
    """Configure logging and store global options"""

    # Setup logging
    if json_output:
        setup_logging(level="CRITICAL", use_rich=False, use_json=False)
    elif debug:
        setup_logging(level="DEBUG", use_rich=True, rich_tracebacks=True)
    elif quiet:
        setup_logging(level="WARNING", use_rich=True)
    else:
        setup_logging(level="INFO", use_rich=True)

    ctx.obj = {"ssh_key": ssh_key, "json": json_output}
```

**Key Points:**
- `--ssh-key` is a global option (in `@app.callback()`)
- Stored in context object `ctx.obj` for access by subcommands
- Must be placed BEFORE the subcommand name in CLI

## 2. SSH Key Resolution Function

**File:** `/Users/chiraagbalu/research/bifrost/bifrost/cli.py`  
**Lines:** 58-85

```python
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
        logger.info(f"Or use: --ssh-key {found_keys[0]}")  # <-- BUG: Misleading!
    else:
        logger.info("No SSH keys found in ~/.ssh/")
        logger.info("Generate one: ssh-keygen -t ed25519")
        logger.info("")
    logger.info("Set SSH_KEY_PATH in .env (run: bifrost init)")

    raise typer.Exit(1)
```

**Precedence Chain:**
1. CLI argument `--ssh-key` (from context)
2. Environment variable `SSH_KEY_PATH` (via `get_ssh_key_path()`)
3. Auto-discovery from `~/.ssh/` (via `discover_ssh_keys()`)
4. Error with suggestions

**The Bug (Line 78):**
```python
logger.info(f"Or use: --ssh-key {found_keys[0]}")
```
Should be:
```python
logger.info(f"Or use: bifrost --ssh-key {found_keys[0]} <command> ...")
```

## 3. Exec Command Implementation

**File:** `/Users/chiraagbalu/research/bifrost/bifrost/cli.py`  
**Lines:** 214-265

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
    ssh_key = resolve_ssh_key(ctx)  # <-- Calls the resolution function

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
        print(
            json.dumps(
                {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                },
                indent=2,
            )
        )
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

**Key Points:**
- Line 225: Calls `resolve_ssh_key(ctx)` to get SSH key
- Line 238: Passes `ssh_key_path=ssh_key` to BifrostClient

## 4. Helper Functions in Shared Config

**File:** `/Users/chiraagbalu/research/shared/shared/config.py`  
**Lines:** 85-97

```python
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
```

**Supports:**
- Ed25519 keys (modern, recommended)
- RSA keys (legacy)
- ECDSA keys (legacy)

## 5. Init Command (Creates .env Template)

**File:** `/Users/chiraagbalu/research/bifrost/bifrost/cli.py`  
**Lines:** 131-145

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

**Template Generation:** `/Users/chiraagbalu/research/shared/shared/config.py` lines 100-126

```python
def create_env_template(tool: str):
    """Create .env template for broker or bifrost"""
    if Path(".env").exists():
        raise FileExistsError(".env already exists")

    if tool == "broker":
        template = """# GPU Broker Credentials
RUNPOD_API_KEY=
PRIME_API_KEY=
LAMBDA_API_KEY=
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

## 6. All Commands Using resolve_ssh_key()

**File:** `/Users/chiraagbalu/research/bifrost/bifrost/cli.py`

| Command | Line | Purpose |
|---------|------|---------|
| `push` | 173 | Deploy code |
| `exec` | 225 | Execute command |
| `deploy` | 278 | Deploy + execute |
| `run` | 355 | Background job |
| `jobs` | 414 | List jobs |
| `logs` | 485 | View logs |
| `download` | 517 | Download files |
| `upload` | 549 | Upload files |

All these commands follow the same pattern:
```python
ssh_key = resolve_ssh_key(ctx)
# ... validation ...
client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)
```

## 7. Client Integration

**File:** `/Users/chiraagbalu/research/bifrost/bifrost/client.py`

The resolved SSH key is passed to BifrostClient:
```python
client = BifrostClient(ssh_conn_str, ssh_key_path=ssh_key)
```

The client uses this to establish SSH connections for:
- Pushing code (git-based deployment)
- Executing commands
- Managing jobs (tmux sessions)
- Transferring files (SFTP)

## Execution Flow Diagram

```
User Input:
  bifrost --ssh-key ~/.ssh/id_ed25519 exec user@host "ls"
         ^^^^^^ 1. Global callback processes this first
  
                                    ^^^^^^ 2. Routes to exec subcommand
                                         ^^^^^^^^^^^^^^^^^ 3. Command arguments

Flow:

[main() callback]  ← Processes --ssh-key, -q, --json, --debug
        |
        v
[ctx.obj = {"ssh_key": "~/.ssh/id_ed25519", ...}]
        |
        v
[exec() command]
        |
        v
[resolve_ssh_key(ctx)]
        |
        +─→ Check ctx.obj.get("ssh_key") ─→ Found? Return it
        |
        +─→ Check get_ssh_key_path() ─→ Found? Return it
        |
        +─→ Check discover_ssh_keys() ─→ Found? Display options
        |
        +─→ Raise error with suggestions
        
[BifrostClient(ssh_connection, ssh_key_path=ssh_key)]
        |
        v
[Execute command via SSH]
```

## Testing

**File:** `/Users/chiraagbalu/research/bifrost/tests/smoke_exec_stream.py`

Shows how BifrostClient accepts ssh_key_path:
```python
client = BifrostClient("user@example.com:22", ssh_key_path=str(key_path))
```

The CLI correctly extracts and passes this from the user's configuration.
