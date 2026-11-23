"""Unit tests for Windows Core Audio device discovery and stable GUID enumeration."""

import pytest

from hth.audio.devices import (
    FakeDeviceProvider,
    WindowsMMDeviceProvider,
    get_device_inventory,
    list_audio_devices,
    set_audio_device_provider,
)
from hth.domain.audio_contracts import (
    EndpointFormFactor,
)


@pytest.fixture(autouse=True)
def setup_fake_devices() -> None:
    """Ensure tests run against a deterministic fake device inventory."""
    provider = FakeDeviceProvider()
    set_audio_device_provider(provider)


def test_device_inventory_contains_stable_guids() -> None:
    inv = get_device_inventory()
    assert len(inv.capture_devices) >= 2
    assert len(inv.render_devices) >= 2

    # Verify GUIDs are stable strings, not transient integer indices
    for dev in inv.capture_devices:
        assert isinstance(dev.id, str)
        assert dev.id.startswith("{") and dev.id.endswith("}")
        assert dev.friendly_name

    for dev in inv.render_devices:
        assert isinstance(dev.id, str)
        assert dev.id.startswith("{") and dev.id.endswith("}")
        assert dev.friendly_name


def test_vb_cable_endpoint_detection() -> None:
    inv = get_device_inventory()
    assert inv.vb_cable_installed is True

    # Detect CABLE Output on capture side
    cable_out = next((d for d in inv.capture_devices if d.is_vb_cable_output), None)
    assert cable_out is not None
    assert "CABLE Output" in cable_out.friendly_name
    assert cable_out.is_physical is False

    # Detect CABLE Input on render side
    cable_in = next((d for d in inv.render_devices if d.is_vb_cable_input), None)
    assert cable_in is not None
    assert "CABLE Input" in cable_in.friendly_name
    assert cable_in.is_physical is False


def test_physical_device_classification() -> None:
    inv = get_device_inventory()
    assert inv.physical_microphone_detected is True
    assert inv.physical_headphones_detected is True

    phys_mic = next((d for d in inv.capture_devices if d.is_physical), None)
    assert phys_mic is not None
    assert phys_mic.form_factor == EndpointFormFactor.MICROPHONE

    phys_hp = next((d for d in inv.render_devices if d.is_physical), None)
    assert phys_hp is not None
    assert phys_hp.form_factor == EndpointFormFactor.HEADPHONES


def test_default_roles_detected() -> None:
    inv = get_device_inventory()
    assert inv.default_console_capture_id is not None
    assert inv.default_communications_capture_id is not None
    assert inv.default_console_render_id is not None
    assert inv.default_communications_render_id is not None

    def_cap = next(d for d in inv.capture_devices if d.id == inv.default_console_capture_id)
    assert def_cap.is_default_console is True


def test_legacy_list_audio_devices_backward_compatibility() -> None:
    devs = list_audio_devices()
    assert len(devs) >= 4
    for d in devs:
        assert d.id
        assert d.friendly_name
        assert d.default_sample_rate > 0


def test_windows_provider_initialization() -> None:
    # Windows provider should instantiate cleanly without raising
    wp = WindowsMMDeviceProvider()
    assert wp is not None
