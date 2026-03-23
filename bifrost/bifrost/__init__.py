"""Bifrost SDK - Python client for remote GPU execution and job management."""

# ruff: noqa: F822

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AsyncBifrostClient",
    "BifrostClient",
    "ExecutionSession",
    "connect",
    "ModalExecutionRequest",
    "ModalSandboxHandle",
    "ObservedProcessHandle",
    "create_modal_sandbox",
    "exec_modal_command",
    "materialize_modal_workspace",
    "run_modal_request",
    "terminate_modal_sandbox",
    "ProcessSpec",
    "ServiceSpec",
    "ProcessHandle",
    "JobInfo",
    "ServiceHandle",
    "ServerInfo",
    "GPUQuery",
    "job_status",
    "job_wait",
    "job_logs",
    "job_stream_logs",
    "job_stream_until_complete",
    "job_exit_code",
    "job_kill",
    "server_is_healthy",
    "server_wait_until_healthy",
    "server_logs",
    "server_stop",
    "server_is_running",
    "acquire_node",
    "InstanceNotFoundError",
    "SSHConnection",
    "WorkspaceMaterializationSpec",
    "PythonProjectMaterialization",
    "EnvironmentVariables",
    "WorkspaceHandle",
    "ChildEvent",
    "ProcessOutputLine",
    "LifecycleEvent",
    "OutputSink",
    "EventStreamRef",
    "create_jsonl_event_stream",
    "read_lifecycle_events",
    "ReadinessProbe",
    "ProcessState",
    "ServiceState",
    "write_file_safe",
    "ensure_dir",
    "path_exists",
    "read_file",
    "remove_file",
]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "AsyncBifrostClient": (".async_client", "AsyncBifrostClient"),
    "BifrostClient": (".client", "BifrostClient"),
    "ExecutionSession": (".session", "ExecutionSession"),
    "connect": (".session", "connect"),
    "ModalExecutionRequest": (".modal_backend", "ModalExecutionRequest"),
    "ModalSandboxHandle": (".modal_backend", "ModalSandboxHandle"),
    "create_modal_sandbox": (".modal_backend", "create_modal_sandbox"),
    "exec_modal_command": (".modal_backend", "exec_modal_command"),
    "materialize_modal_workspace": (".modal_backend", "materialize_modal_workspace"),
    "run_modal_request": (".modal_backend", "run_modal_request"),
    "terminate_modal_sandbox": (".modal_backend", "terminate_modal_sandbox"),
    "job_exit_code": (".job", "job_exit_code"),
    "job_kill": (".job", "job_kill"),
    "job_logs": (".job", "job_logs"),
    "job_status": (".job", "job_status"),
    "job_stream_logs": (".job", "job_stream_logs"),
    "job_stream_until_complete": (".job", "job_stream_until_complete"),
    "job_wait": (".job", "job_wait"),
    "GPUQuery": (".provision", "GPUQuery"),
    "InstanceNotFoundError": (".provision", "InstanceNotFoundError"),
    "acquire_node": (".provision", "acquire_node"),
    "ensure_dir": (".remote_fs", "ensure_dir"),
    "path_exists": (".remote_fs", "path_exists"),
    "read_file": (".remote_fs", "read_file"),
    "remove_file": (".remote_fs", "remove_file"),
    "write_file_safe": (".remote_fs", "write_file_safe"),
    "server_is_healthy": (".server", "server_is_healthy"),
    "server_is_running": (".server", "server_is_running"),
    "server_logs": (".server", "server_logs"),
    "server_stop": (".server", "server_stop"),
    "server_wait_until_healthy": (".server", "server_wait_until_healthy"),
    "ChildEvent": (".types", "ChildEvent"),
    "EnvironmentVariables": (".types", "EnvironmentVariables"),
    "EventStreamRef": (".types", "EventStreamRef"),
    "JobInfo": (".types", "JobInfo"),
    "LifecycleEvent": (".types", "LifecycleEvent"),
    "ObservedProcessHandle": (".types", "ObservedProcessHandle"),
    "OutputSink": (".types", "OutputSink"),
    "ProcessOutputLine": (".types", "ProcessOutputLine"),
    "ProcessSpec": (".types", "ProcessSpec"),
    "ProcessHandle": (".types", "ProcessHandle"),
    "ProcessState": (".types", "ProcessState"),
    "PythonProjectMaterialization": (".types", "PythonProjectMaterialization"),
    "ReadinessProbe": (".types", "ReadinessProbe"),
    "create_jsonl_event_stream": (".types", "create_jsonl_event_stream"),
    "read_lifecycle_events": (".types", "read_lifecycle_events"),
    "ServiceSpec": (".types", "ServiceSpec"),
    "ServerInfo": (".types", "ServerInfo"),
    "ServiceHandle": (".types", "ServiceHandle"),
    "ServiceState": (".types", "ServiceState"),
    "SSHConnection": (".types", "SSHConnection"),
    "WorkspaceMaterializationSpec": (".types", "WorkspaceMaterializationSpec"),
    "WorkspaceHandle": (".types", "WorkspaceHandle"),
}


def __getattr__(name: str) -> object:
    if name not in _SYMBOLS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _SYMBOLS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
