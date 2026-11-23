"""Unit tests for AudioRouter loop detection, validation, and fail-closed state handling."""

import pytest

from hth.audio.devices import FakeDeviceProvider, set_audio_device_provider
from hth.audio.router import AudioRouter
from hth.domain.audio_contracts import AudioRoutingConfig


@pytest.fixture
def router() -> AudioRouter:
    set_audio_device_provider(FakeDeviceProvider())
    return AudioRouter()


def test_valid_routing_passes(router: AudioRouter) -> None:
    # Physical mic -> CABLE Input, Headphones -> Monitor
    cfg = AudioRoutingConfig(
        input_device_id="{0.0.1.00000000}.{realtek-mic-guid}",
        output_device_id="{0.0.0.00000000}.{vbcable-in-guid}",
        monitor_device_id="{0.0.0.00000000}.{realtek-hp-guid}",
        monitor_enabled=True,
    )
    result = router.validate_routing(cfg)
    assert result.valid is True
    assert len(result.errors) == 0
    assert len(result.checklist) >= 4

    apply_res = router.apply_routing(cfg)
    assert apply_res.valid is True

    status = router.get_status()
    assert status.active_config == cfg
    assert status.feedback_loop_detected is False


def test_feedback_loop_input_is_cable_output_blocked(router: AudioRouter) -> None:
    # Attempting to capture from CABLE Output creates infinite loop
    cfg = AudioRoutingConfig(
        input_device_id="{0.0.1.00000000}.{vbcable-out-guid}",
        output_device_id="{0.0.0.00000000}.{vbcable-in-guid}",
    )
    result = router.validate_routing(cfg)
    assert result.valid is False
    assert any("FEEDBACK LOOP BLOCKED" in err for err in result.errors)

    apply_res = router.apply_routing(cfg)
    assert apply_res.valid is False
    assert router.get_status().feedback_loop_detected is True


def test_feedback_loop_monitor_is_cable_input_blocked(router: AudioRouter) -> None:
    # Attempting to monitor back into CABLE Input creates echo loop
    cfg = AudioRoutingConfig(
        input_device_id="{0.0.1.00000000}.{realtek-mic-guid}",
        output_device_id="{0.0.0.00000000}.{vbcable-in-guid}",
        monitor_device_id="{0.0.0.00000000}.{vbcable-in-guid}",
        monitor_enabled=True,
    )
    result = router.validate_routing(cfg)
    assert result.valid is False
    assert any("FEEDBACK LOOP BLOCKED" in err for err in result.errors)


def test_conflict_same_output_and_monitor_blocked(router: AudioRouter) -> None:
    cfg = AudioRoutingConfig(
        input_device_id="{0.0.1.00000000}.{realtek-mic-guid}",
        output_device_id="{0.0.0.00000000}.{realtek-hp-guid}",
        monitor_device_id="{0.0.0.00000000}.{realtek-hp-guid}",
        monitor_enabled=True,
    )
    result = router.validate_routing(cfg)
    assert result.valid is False
    assert any("ROUTING CONFLICT" in err for err in result.errors)


def test_nonexistent_device_blocked(router: AudioRouter) -> None:
    cfg = AudioRoutingConfig(
        input_device_id="{nonexistent-id}",
        output_device_id="{0.0.0.00000000}.{vbcable-in-guid}",
    )
    result = router.validate_routing(cfg)
    assert result.valid is False
    assert any("not found" in err for err in result.errors)


def test_panic_stop_mutes_output(router: AudioRouter) -> None:
    assert router.get_status().is_muted is False
    router.trigger_panic_stop()
    assert router.get_status().is_muted is True


def test_passthrough_toggle_is_deliberate(router: AudioRouter) -> None:
    assert router.get_status().is_passthrough is False
    state = router.toggle_passthrough()
    assert state is True
    assert router.get_status().is_passthrough is True
    state2 = router.toggle_passthrough(False)
    assert state2 is False
    assert router.get_status().is_passthrough is False


def test_device_unplug_handles_fail_closed(router: AudioRouter) -> None:
    cfg = AudioRoutingConfig(
        input_device_id="{0.0.1.00000000}.{realtek-mic-guid}",
        output_device_id="{0.0.0.00000000}.{vbcable-in-guid}",
    )
    router.apply_routing(cfg)
    router.set_running_state(True)

    # Simulate unplugging physical mic
    router.handle_device_unplug("{0.0.1.00000000}.{realtek-mic-guid}")

    status = router.get_status()
    assert status.is_muted is True
    assert status.device_error is not None
    assert "unplugged" in status.device_error
