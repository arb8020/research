from __future__ import annotations

from types import SimpleNamespace

import pytest

from argus import run as argus_run
from bifrost.types import ExecResult


def test_read_remote_manifest_uses_exec_result_stdout() -> None:
    manifest_json = '{"resolved_image_ref":"ghcr.io/example/image:latest","source_ref":"ghcr.io/example/image:latest"}'
    fake_bifrost = SimpleNamespace(
        exec=lambda command: ExecResult(stdout=manifest_json, stderr="", exit_code=0)
    )

    manifest = argus_run._read_remote_manifest(fake_bifrost)

    assert manifest is not None
    assert manifest.resolved_image_ref == "ghcr.io/example/image:latest"


def test_read_remote_manifest_raises_on_exec_failure() -> None:
    fake_bifrost = SimpleNamespace(
        exec=lambda command: ExecResult(stdout="", stderr="permission denied", exit_code=1)
    )

    with pytest.raises(RuntimeError, match="Remote image manifest probe failed"):
        argus_run._read_remote_manifest(fake_bifrost)
