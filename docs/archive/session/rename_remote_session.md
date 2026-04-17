# Handoff: Rename RemoteSession → RemoteConnection

## Branch / worktree

```
cd ~/research/rollouts
git worktree add ../rollouts-rename-remote-session main
```

Base commit: `88b71fa`

---

## Goal

`RemoteSession` in `resources.py` is a live connection handle to an execution substrate
(SSH, Modal, Docker). "Session" is the wrong word — it collides with the agent
conversation session concept (`SessionStore`, `session_id`) which is becoming
increasingly central. Rename to `RemoteConnection` throughout.

This is a pure mechanical rename. No logic changes.

---

## Read first

```
~/research/docs/code_style/cheatsheet.md
rollouts/rollouts/environments/resources.py        (definitions)
```

---

## Renames

```
RemoteSession                →  RemoteConnection
InspectableRemoteSession     →  InspectableRemoteConnection
SessionBackedWorkspaceHandle →  ConnectionBackedWorkspaceHandle
SessionExecSpec              →  ExecSpec
```

## Files to update

```
rollouts/rollouts/environments/resources.py
rollouts/rollouts/environments/bifrost_workspace_resource.py
rollouts/rollouts/environments/modal_sandbox_resource.py
rollouts/rollouts/environments/docker_workspace_resource.py
rollouts/rollouts/environments/local_workspace_resource.py
rollouts/rollouts/eval/remote_runtime.py
rollouts/rollouts/eval/external_attempts.py
```

Also check `core/__init__.py` and `core/types.py` for any re-exports.

## Verify

```
cd ~/research/rollouts && .venv/bin/python -m pytest tests/unit/ -x -q
```

`SessionExecSpec` has a `session_id` field on it — that field refers to the agent
conversation session, not the connection. Keep the field name `session_id`, just rename
the containing type to `ExecSpec`.
