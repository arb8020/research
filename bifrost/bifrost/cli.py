"""Bifrost CLI - remote GPU execution and job management"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from bifrost.client import BifrostClient
from bifrost.types import JobStatus
from shared.config import create_env_template, discover_ssh_keys, get_ssh_key_path
from shared.logging_config import setup_logging

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
    # Remove 'ssh' prefix if present
    conn_str = conn_str.strip()
    if conn_str.startswith("ssh "):
        # Extract from ssh command
        # ssh -p PORT user@host or ssh user@host
        match = re.match(r"ssh\s+(?:-p\s+(\d+)\s+)?([^@]+)@(\S+)", conn_str)
        if match:
            port, user, host = match.groups()
            return user, host, int(port) if port else 22

    # Standard format: user@host:port or user@host
    match = re.match(r"^([^@]+)@([^:]+)(?::(\d+))?$", conn_str)
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
        logger.info("found keys at:")
        for key in found_keys:
            logger.info(f"  {key}")
        logger.info("")
        logger.info(f"or use: bifrost --ssh-key {found_keys[0]} <command>")
    else:
        logger.info("no ssh keys found in ~/.ssh/")
        logger.info("generate one: ssh-keygen -t ed25519")
        logger.info("")
    logger.info("set SSH_KEY_PATH in .env (run: bifrost init)")

    raise typer.Exit(1)


def parse_env_vars(env_list: List[str]) -> Dict[str, str]:
    """Parse environment variables from KEY=VALUE format

    Raises:
        typer.Exit: If format is invalid (caught by CLI commands)
    """
    env_dict = {}
    for item in env_list:
        if "=" not in item:
            logger.error(f"✗ Invalid env format: {item}")
            logger.info("expected format: KEY=VALUE")
            logger.info("example: --env API_KEY=abc123 --env DEBUG=true")
            raise typer.Exit(1)
        key, value = item.split("=", 1)
        env_dict[key] = value
    return env_dict


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


@app.command()
def init():
    """Create .env template for SSH configuration"""
    try:
        create_env_template("bifrost")
        logger.info("created .env with ssh configuration template")
        logger.info("")
        logger.info("edit .env with your ssh key path:")
        logger.info("  SSH_KEY_PATH=~/.ssh/id_ed25519")
        logger.info("")
        logger.info("then run: bifrost push <ssh-connection>")
    except FileExistsError:
        logger.error("✗ .env already exists")
        logger.info("edit manually or delete to recreate")
        raise typer.Exit(1)


@app.command()
def push(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(..., help="SSH connection (user@host:port)"),
    bootstrap: Optional[List[str]] = typer.Option(
        None,
        "--bootstrap",
        help="Bootstrap command (can specify multiple, joined with &&)",
    ),
    bootstrap_script: Optional[str] = typer.Option(
        None, "--bootstrap-script", help="Path to bootstrap script to upload and execute"
    ),
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

    logger.info("deploying code...")
    workspace_path = client.push(bootstrap_cmd=bootstrap_cmd)

    logger.info(f"code deployed to {workspace_path}")


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

    logger.info(f"executing: {command}")
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
            logger.info("command completed successfully")


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

    logger.info("deploying code and executing command...")
    result = client.deploy(command, bootstrap_cmd=bootstrap_cmd, env=env_dict)

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
            logger.error("✗ Command failed")
            raise typer.Exit(result.exit_code)
        else:
            logger.info("deploy and execution completed successfully")


@app.command()
def run(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    command: str = typer.Argument(...),
    bootstrap: Optional[List[str]] = typer.Option(None, "--bootstrap"),
    bootstrap_script: Optional[str] = typer.Option(None, "--bootstrap-script"),
    env: Optional[List[str]] = typer.Option(None, "--env"),
    name: Optional[str] = typer.Option(None, "--name", help="Human-readable job name"),
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

    logger.info("starting detached job...")
    job_info = client.run_detached(
        command=command, bootstrap_cmd=bootstrap_cmd, env=env_dict, session_name=name
    )

    if ctx.obj["json"]:
        print(
            json.dumps(
                {
                    "job_id": job_info.job_id,
                    "status": job_info.status.value,
                    "command": job_info.command,
                },
                indent=2,
            )
        )
    else:
        logger.info(f"job {job_info.job_id} started")
        logger.info(
            f"monitor: bifrost logs {ssh_connection} {job_info.job_id} --follow"
        )


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
    jobs_list = client.get_all_jobs()

    if ctx.obj["json"]:
        print(
            json.dumps(
                [
                    {
                        "job_id": j.job_id,
                        "status": j.status.value,
                        "command": j.command,
                        "start_time": (
                            j.start_time.isoformat() if j.start_time else None
                        ),
                    }
                    for j in jobs_list
                ],
                indent=2,
            )
        )
    else:
        if not jobs_list:
            logger.info("No jobs found")
            return

        table = Table(title=f"Jobs on {ssh_connection}")
        table.add_column("Job ID", style="cyan")
        table.add_column("Status")
        table.add_column("Command", max_width=40)
        table.add_column("Runtime")

        for job in jobs_list:
            status_style = "green" if job.status == JobStatus.COMPLETED else "yellow"
            runtime = (
                f"{int(job.runtime_seconds)}s" if job.runtime_seconds else "N/A"
            )

            table.add_row(
                job.job_id,
                f"[{status_style}]{job.status.value}[/{status_style}]",
                (
                    job.command[:40] + "..."
                    if len(job.command) > 40
                    else job.command
                ),
                runtime,
            )

        console.print(table)


@app.command()
def logs(
    ctx: typer.Context,
    ssh_connection: str = typer.Argument(...),
    job_id: str = typer.Argument(...),
    follow: bool = typer.Option(
        False, "-f", "--follow", help="Follow logs in real-time (like tail -f)"
    ),
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
        logger.info(f"following logs for {job_id} (Ctrl+C to exit)...")
        try:
            for line in client.follow_job_logs(job_id):
                print(line)
        except KeyboardInterrupt:
            logger.info("stopped following logs")
    else:
        logs_content = client.get_logs(job_id, lines=lines)
        print(logs_content)


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

    logger.info(f"downloading {remote_path}...")
    result = client.download_files(remote_path, local_path, recursive=recursive)

    if result.success:
        logger.info(
            f"downloaded {result.files_copied} files ({result.total_bytes} bytes)"
        )
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

    logger.info(f"uploading {local_path}...")
    result = client.upload_files(local_path, remote_path, recursive=recursive)

    if result.success:
        logger.info(
            f"uploaded {result.files_copied} files ({result.total_bytes} bytes)"
        )
    else:
        logger.error(f"✗ Upload failed: {result.error_message}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
