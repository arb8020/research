class WorkspaceInfraError(RuntimeError):
    """Terminal infrastructure failure for a workspace-backed environment."""

    def __init__(self, message: str, *, kind: str = "workspace_infra_error") -> None:
        super().__init__(message)
        self.kind = kind
