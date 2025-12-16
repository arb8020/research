"""
Example showing Phase 3 API - Enhanced job management with sessions
"""

from bifrost import BifrostClient

# Phase 3: Enhanced job management
client = BifrostClient(ssh_connection="root@1.2.3.4:22", ssh_key_path="~/.ssh/id_ed25519")

# Run detached job with human-readable session name
job = client.run_detached(
    command="python train.py",
    bootstrap_cmd="uv sync --frozen",
    bootstrap_timeout=600,  # 10 minutes for bootstrap
    session_name="training-run-1",  # Human-readable!
)

print(f"Job ID: {job.job_id}")  # training-run-1-20251012-143025-a3f2b8c1
print(f"Main session: {job.tmux_session}")  # bifrost-training-run-1-20251012-143025-a3f2b8c1
print(f"Bootstrap session: {job.bootstrap_session}")  # bifrost-training-run-1-...-bootstrap

# Get session info with attach commands
session_info = client.get_session_info(job.job_id)
print(f"\nAttach to main: {session_info.attach_main}")
print(f"Attach to bootstrap: {session_info.attach_bootstrap}")

# Get separate logs for bootstrap and command
bootstrap_logs = client.get_logs(job.job_id, log_type="bootstrap", lines=50)
print(f"\nBootstrap logs:\n{bootstrap_logs}")

command_logs = client.get_logs(job.job_id, log_type="command", lines=50)
print(f"\nCommand logs:\n{command_logs}")

# List all bifrost sessions on remote
sessions = client.list_sessions()
print(f"\nAll bifrost sessions: {sessions}")

# Wait for completion
result = client.wait_for_completion(job.job_id)
print(f"Job completed with status: {result.status}")
