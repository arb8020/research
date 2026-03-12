from __future__ import annotations

import pytest

from broker.api import _resolve_persistent_volume
from broker.providers.runpod import _apply_persistent_volume
from broker.types import PersistentVolumeAttachment, ProvisionRequest


def test_provision_request_normalizes_legacy_runpod_volume_fields() -> None:
    request = ProvisionRequest(
        provider="runpod",
        network_volume_id="vol-123",
        datacenter_id="dc-1",
    )

    assert request.persistent_volume is not None
    assert request.persistent_volume.volume_id == "vol-123"
    assert request.persistent_volume.location_hint == "dc-1"
    assert request.persistent_volume.mount_path == "/workspace"


def test_resolve_persistent_volume_prefers_generic_attachment() -> None:
    attachment = PersistentVolumeAttachment(
        volume_id="vol-123",
        mount_path="/cache",
        location_hint="iad-1",
    )

    resolved = _resolve_persistent_volume(
        persistent_volume=attachment,
        persistent_volume_id="vol-123",
        persistent_volume_mount_path="/cache",
        persistent_volume_location="iad-1",
        network_volume_id=None,
        datacenter_id=None,
    )

    assert resolved is attachment


def test_resolve_persistent_volume_rejects_conflicting_aliases() -> None:
    attachment = PersistentVolumeAttachment(
        volume_id="vol-123",
        mount_path="/cache",
        location_hint="iad-1",
    )

    with pytest.raises(AssertionError, match="location_hint"):
        _resolve_persistent_volume(
            persistent_volume=attachment,
            persistent_volume_id=None,
            persistent_volume_mount_path=None,
            persistent_volume_location="phx-1",
            network_volume_id=None,
            datacenter_id=None,
        )


def test_runpod_translation_uses_generic_persistent_volume() -> None:
    pod_input: dict[str, object] = {}
    request = ProvisionRequest(
        provider="runpod",
        persistent_volume=PersistentVolumeAttachment(
            volume_id="vol-123",
            mount_path="/workspace",
            location_hint="dc-1",
        ),
    )

    _apply_persistent_volume(pod_input, request)

    assert pod_input["networkVolumeId"] == "vol-123"
    assert pod_input["dataCenterId"] == "dc-1"
    assert pod_input["volumeMountPath"] == "/workspace"


def test_runpod_translation_rejects_conflicting_mount_paths() -> None:
    pod_input: dict[str, object] = {"volumeMountPath": "/other"}
    request = ProvisionRequest(
        provider="runpod",
        persistent_volume=PersistentVolumeAttachment(
            volume_id="vol-123",
            mount_path="/workspace",
            location_hint="dc-1",
        ),
    )

    with pytest.raises(AssertionError, match="volumeMountPath"):
        _apply_persistent_volume(pod_input, request)
