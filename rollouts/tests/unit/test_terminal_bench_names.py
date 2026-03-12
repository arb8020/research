from pathlib import Path

from rollouts.environments.terminal_bench import _docker_resource_names


def test_docker_resource_names_include_logging_dir_hash() -> None:
    first = _docker_resource_names("Fix Permissions", Path("/tmp/tb/run-a"))
    second = _docker_resource_names("Fix Permissions", Path("/tmp/tb/run-b"))

    assert first != second
    assert first[0].startswith("tb-fix-permissions-")
    assert first[1].endswith("-image")
