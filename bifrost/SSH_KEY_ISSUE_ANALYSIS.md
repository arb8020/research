# Bifrost CLI SSH Key Configuration Analysis

## Summary

You've discovered a **mismatch between the error message and the actual CLI implementation**. The error message on line 78 of `/Users/chiraagbalu/research/bifrost/bifrost/cli.py` suggests using `--ssh-key` flag, but this flag is defined in the `@app.callback()` decorator (global callback), not in individual commands like `exec`. This creates a confusing UX where:

1. Users are told to use `--ssh-key` when SSH key is missing
2. But `bifrost exec --ssh-key` fails with "No such option: --ssh-key" because the flag must come **before** the command name

## Root Cause: Flag Placement in Typer

The issue stems from how Typer handles global options. In Typer (based on Click), flags defined in the `@app.callback()` are "global" and must be placed **before the subcommand name**, not after.

### Correct vs Incorrect Usage

**CORRECT:**
```bash
bifrost --ssh-key ~/.ssh/id_ed25519 exec user@host "ls"
```

**INCORRECT (fails with "No such option: --ssh-key"):**
```bash
bifrost exec --ssh-key ~/.ssh/id_ed25519 user@host "ls"
```

## How SSH Key Resolution Works

The SSH key resolution follows a precedence chain in the `resolve_ssh_key()` function (lines 58-85):

```
1. CLI flag: --ssh-key (via ctx.obj["ssh_key"])
2. Environment variable: SSH_KEY_PATH (via get_ssh_key_path())
3. Discovery: Search ~/.ssh/ for common keys (id_ed25519, id_rsa, id_ecdsa)
4. Error with helpful suggestions
```

### Code Flow

1. **CLI Entry Point** (`bifrost/cli.py`, lines 106-129):
   - Global callback stores `--ssh-key` argument in `ctx.obj`
   - Available to all subcommands via the context

2. **Key Resolution** (`bifrost/cli.py`, lines 58-85):
   - Called by each command (exec, push, deploy, run, etc.)
   - Tries CLI flag first (highest priority)
   - Falls back to SSH_KEY_PATH environment variable
   - Attempts discovery of common SSH keys
   - Provides helpful error message if nothing found

3. **Config Loading** (`shared/shared/config.py`, lines 85-97):
   - `get_ssh_key_path()` reads SSH_KEY_PATH from environment/`.env`
   - `discover_ssh_keys()` finds common SSH keys in `~/.ssh/`

## The Two Recommended Configuration Methods

### Method 1: Set SSH_KEY_PATH in .env (RECOMMENDED)

This is the intended design pattern:

```bash
# Initialize .env template
bifrost init

# Edit .env
echo "SSH_KEY_PATH=~/.ssh/id_ed25519" >> .env

# Now use without --ssh-key flag
bifrost exec user@host "python train.py"
```

The `init` command (lines 132-145) creates a `.env` template with helpful instructions.

### Method 2: Use Global --ssh-key Flag (ALTERNATIVE)

Place the flag **before** the subcommand:

```bash
bifrost --ssh-key ~/.ssh/id_ed25519 exec user@host "python train.py"
```

This works because Typer processes the `@app.callback()` options before routing to the subcommand.

## Affected Commands

All commands that execute SSH operations use `resolve_ssh_key()`:
- `push` (line 173)
- `exec` (line 225)
- `deploy` (line 278)
- `run` (line 355)
- `jobs` (line 414)
- `logs` (line 485)
- `download` (line 517)
- `upload` (line 549)

## The Misleading Error Message

Lines 71-83 show the problematic error message:

```python
logger.error("✗ No SSH key specified")
logger.info("")
if found_keys:
    logger.info("Found keys at:")
    for key in found_keys:
        logger.info(f"  {key}")
    logger.info("")
    logger.info(f"Or use: --ssh-key {found_keys[0]}")  # <-- MISLEADING
else:
    logger.info("No SSH keys found in ~/.ssh/")
    logger.info("Generate one: ssh-keygen -t ed25519")
    logger.info("")
logger.info("Set SSH_KEY_PATH in .env (run: bifrost init)")
```

**The Problem:** It suggests `--ssh-key` without mentioning it must come **before** the command name. This is confusing because users naturally place flags after the command.

## Recommended Fix

Update the error message to be accurate:

```python
# Option 1: Show correct syntax
logger.info(f"Or use: bifrost --ssh-key {found_keys[0]} <command> ...")

# Option 2: Recommend the simpler approach
logger.info("Or set SSH_KEY_PATH in .env:")
logger.info("  1. Run: bifrost init")
logger.info("  2. Edit .env with your key path")
```

## Implementation Details

### Main Callback (lines 106-129)

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
    # ...setup logging...
    ctx.obj = {"ssh_key": ssh_key, "json": json_output}
```

The context object is passed to all subcommands.

### SSH Key Path Resolution (lines 58-85)

```python
def resolve_ssh_key(ctx) -> str:
    """Resolve SSH key: CLI → env → discover → error"""
    ssh_key_arg = ctx.obj.get("ssh_key")
    
    if ssh_key_arg:
        return ssh_key_arg
    
    if key_path := get_ssh_key_path():
        return key_path
    
    # Discovery...
```

Uses walrus operator to cleanly check environment variable.

### Shared Config (shared/shared/config.py, lines 85-97)

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

Uses standard `~/.ssh/` locations. Supports modern ed25519 keys and legacy RSA/ECDSA.

## Example Usage Scenarios

### Scenario 1: First Time Setup
```bash
$ bifrost exec user@gpu.com "ls"
✗ No SSH key specified

Found keys at:
  /Users/user/.ssh/id_ed25519

Or use: --ssh-key /Users/user/.ssh/id_ed25519
Set SSH_KEY_PATH in .env (run: bifrost init)

# Solution 1: Use init (RECOMMENDED)
$ bifrost init
# Edit .env with SSH_KEY_PATH=~/.ssh/id_ed25519
$ bifrost exec user@gpu.com "ls"

# Solution 2: Use global flag
$ bifrost --ssh-key ~/.ssh/id_ed25519 exec user@gpu.com "ls"
```

### Scenario 2: Multiple SSH Keys
If you have multiple keys, the recommended approach is to:
1. Set your default in `.env` via `bifrost init`
2. Override temporarily with `bifrost --ssh-key ~/.ssh/other_key exec ...`

### Scenario 3: With Environment Variables
```bash
export SSH_KEY_PATH=~/.ssh/id_ed25519
bifrost exec user@gpu.com "python train.py"
```

## File Locations

- **CLI Implementation**: `/Users/chiraagbalu/research/bifrost/bifrost/cli.py`
- **SSH Key Helper Functions**: `/Users/chiraagbalu/research/shared/shared/config.py`
- **Client Implementation**: `/Users/chiraagbalu/research/bifrost/bifrost/client.py`
- **Tests**: `/Users/chiraagbalu/research/bifrost/tests/smoke_exec_stream.py`

## Testing

The smoke test (`smoke_exec_stream.py`) shows that BifrostClient accepts `ssh_key_path` as a constructor argument:

```python
client = BifrostClient("user@example.com:22", ssh_key_path=str(key_path))
```

The CLI handles parameter extraction and passes it correctly to the client.

## Key Takeaways

1. **Configuration Priority**: CLI flag > Environment variable > Auto-discovery > Error
2. **Best Practice**: Use `bifrost init` + `.env` for persistent configuration
3. **Typer Quirk**: Global options must come **before** subcommands
4. **Bug in Error Message**: Suggests `--ssh-key` without showing correct flag placement
5. **All Commands Support**: Push, exec, deploy, run, jobs, logs, download, upload all use the same SSH resolution
