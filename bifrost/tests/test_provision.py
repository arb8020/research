from types import SimpleNamespace

from bifrost.provision import _ssh_not_ready_message


def test_ssh_not_ready_message_explains_runpod_proxy_is_unsupported() -> None:
    instance = SimpleNamespace(
        provider="runpod",
        public_ip="ssh.runpod.io",
        ssh_port=22,
        ssh_username="pod-host-id",
    )

    message = _ssh_not_ready_message(instance, 900, node_label="runpod:abc123")

    assert "ssh.runpod.io" in message
    assert "not a supported transport" in message
    assert "direct SSH" in message


def test_ssh_not_ready_message_stays_generic_for_non_proxy_instances() -> None:
    instance = SimpleNamespace(
        provider="runpod",
        public_ip=None,
        ssh_port=None,
        ssh_username=None,
    )

    message = _ssh_not_ready_message(instance, 900, node_label="runpod:abc123")

    assert "ssh.runpod.io" not in message
    assert "starting up" in message
