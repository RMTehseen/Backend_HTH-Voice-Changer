"""Audio routing coordinator, loop detection, and safe endpoint validation service.

Phase 4 requirements:
- Make engine output to CABLE Input.
- Calling app input is CABLE Output.
- Monitoring must go to physical headphones, never back into the cable.
- Detect accidental feedback loops and block unsafe routing.
- Do not change Windows defaults automatically; provide a guided checklist.
- Fail-closed on device disconnect with recoverable warnings, never silent rerouting.
"""

from __future__ import annotations

import threading
from typing import Any

from hth.audio.devices import get_device_inventory
from hth.core.logging import get_logger
from hth.domain.audio_contracts import (
    AudioEndpointInfo,
    AudioRoutingConfig,
    AudioRoutingStatus,
    EndpointFormFactor,
    RoutingValidationResult,
)

logger = get_logger("hth.audio.router")


class AudioRouter:
    """Manages audio endpoint assignments, feedback loop prevention, and routing safety."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_config: AudioRoutingConfig | None = None
        self._is_running: bool = False
        self._is_muted: bool = False
        self._is_passthrough: bool = False
        self._device_error: str | None = None
        self._feedback_loop_detected: bool = False
        self._pipeline: Any = None

    def attach_pipeline(self, pipeline: Any) -> None:
        """Attach active AudioPipeline to synchronize panic mute and state."""
        with self._lock:
            self._pipeline = pipeline
            logger.info("Attached AudioPipeline to AudioRouter")

    def detach_pipeline(self) -> None:
        """Detach AudioPipeline."""
        with self._lock:
            self._pipeline = None

    def validate_routing(self, config: AudioRoutingConfig) -> RoutingValidationResult:
        """Validate proposed routing configuration against feedback loops and conflicts."""
        inventory = get_device_inventory()

        capture_by_id = {d.id: d for d in inventory.capture_devices}
        render_by_id = {d.id: d for d in inventory.render_devices}

        errors: list[str] = []
        warnings: list[str] = []
        checklist: list[str] = []

        # 1. Existence Checks
        in_dev = capture_by_id.get(config.input_device_id)
        if not in_dev:
            errors.append(f"Selected input device '{config.input_device_id}' was not found.")

        out_dev = render_by_id.get(config.output_device_id)
        if not out_dev:
            errors.append(f"Selected output device '{config.output_device_id}' was not found.")

        mon_dev: AudioEndpointInfo | None = None
        if config.monitor_enabled and config.monitor_device_id:
            mon_dev = render_by_id.get(config.monitor_device_id)
            if not mon_dev:
                errors.append(
                    f"Selected monitoring device '{config.monitor_device_id}' was not found."
                )

        # 2. Hard Blocker: Accidental Feedback Loop Detection
        if in_dev:
            if in_dev.is_vb_cable_output:
                errors.append(
                    "CRITICAL FEEDBACK LOOP BLOCKED: Input device cannot be 'CABLE Output'. "
                    "Virtual cable output as input causes infinite feedback into the engine."
                )
            if in_dev.is_vb_cable_input:
                errors.append("CRITICAL LOOP BLOCKED: Input device cannot be 'CABLE Input'.")

        if mon_dev:
            if mon_dev.is_vb_cable_input:
                errors.append(
                    "CRITICAL FEEDBACK LOOP BLOCKED: Monitoring device cannot be 'CABLE Input'. "
                    "Routing monitor back into virtual cable causes call echo and howling."
                )
            if mon_dev.is_vb_cable_output:
                errors.append("CRITICAL LOOP BLOCKED: Monitoring device cannot be 'CABLE Output'.")

        if config.output_device_id and config.monitor_device_id:
            if config.output_device_id == config.monitor_device_id and config.monitor_enabled:
                errors.append(
                    "ROUTING CONFLICT: Output device and monitor cannot be the same endpoint."
                )

        if config.input_device_id and config.output_device_id:
            if config.input_device_id == config.output_device_id:
                errors.append(
                    "HARDWARE CONFLICT: Input device and output cannot share the same endpoint ID."
                )

        # 3. Best Practice Warnings
        if out_dev and not out_dev.is_vb_cable_input:
            warnings.append(
                f"Notice: Output device is '{out_dev.friendly_name}', not 'CABLE Input'. "
                "Voice conversion will play to physical speakers instead of calling apps."
            )

        if in_dev and not in_dev.is_physical:
            warnings.append(
                f"Notice: Input '{in_dev.friendly_name}' is not physical microphone hardware."
            )

        if mon_dev and mon_dev.form_factor == EndpointFormFactor.SPEAKERS:
            warnings.append(
                "Acoustic feedback warning: Monitoring is routed to open Speakers. "
                "Headphones are strongly recommended to prevent microphone bleed."
            )

        if not inventory.vb_cable_installed:
            warnings.append(
                "VB-CABLE was not detected on this machine. "
                "Install VB-CABLE Driver Pack from https://vb-audio.com/Cable/."
            )

        # 4. Standard Guided Checklist (Windows Defaults must never be changed automatically)
        checklist.extend(
            [
                "1. In Windows Sound Settings, verify default output remains your Headphones.",
                "2. In Windows Sound Settings, verify default input remains your physical Mic.",
                "3. In your calling app (WhatsApp/Teams), set Mic to: 'CABLE Output'.",
                "4. In your calling app, set Speakers to Headphones (NEVER to CABLE Input).",
            ]
        )

        valid = len(errors) == 0
        return RoutingValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            checklist=checklist,
        )

    def apply_routing(self, config: AudioRoutingConfig) -> RoutingValidationResult:
        """Validate and apply a safe audio routing configuration."""
        res = self.validate_routing(config)
        if not res.valid:
            logger.error("Audio routing validation failed", errors=res.errors)
            with self._lock:
                self._feedback_loop_detected = True
            return res

        with self._lock:
            self._active_config = config
            self._feedback_loop_detected = False
            self._device_error = None
            logger.info(
                "Applied safe audio routing configuration",
                input_id=config.input_device_id,
                output_id=config.output_device_id,
                monitor_id=config.monitor_device_id,
                monitor_enabled=config.monitor_enabled,
            )

        return res

    def get_status(self) -> AudioRoutingStatus:
        """Get live status of audio router and endpoints."""
        inventory = get_device_inventory()
        capture_by_id = {d.id: d for d in inventory.capture_devices}
        render_by_id = {d.id: d for d in inventory.render_devices}

        with self._lock:
            cfg = self._active_config
            in_dev = capture_by_id.get(cfg.input_device_id) if cfg else None
            out_dev = render_by_id.get(cfg.output_device_id) if cfg else None
            mon_id = cfg.monitor_device_id if cfg else None
            mon_dev = render_by_id.get(mon_id) if (cfg and mon_id) else None

            # Detect if an active device was unplugged
            device_error = self._device_error
            if self._is_running and cfg:
                if cfg.input_device_id not in capture_by_id:
                    device_error = f"Input device '{cfg.input_device_id}' was disconnected."
                    self._is_muted = True
                elif cfg.output_device_id not in render_by_id:
                    device_error = f"Output device '{cfg.output_device_id}' was disconnected."
                    self._is_muted = True

            return AudioRoutingStatus(
                is_running=self._is_running,
                is_muted=self._is_muted,
                is_passthrough=self._is_passthrough,
                active_config=cfg,
                input_device=in_dev,
                output_device=out_dev,
                monitor_device=mon_dev,
                vb_cable_detected=inventory.vb_cable_installed,
                feedback_loop_detected=self._feedback_loop_detected,
                device_error=device_error,
            )

    def trigger_panic_stop(self) -> None:
        """Instantly mute outgoing audio stream. Thread-safe panic switch."""
        with self._lock:
            self._is_muted = True
            pipe = self._pipeline
            logger.warning("AUDIO PANIC STOP TRIGGERED — Virtual mic stream muted immediately")

        if pipe is not None:
            try:
                pipe.panic_mute()
            except Exception as exc:
                logger.warning("Failed to propagate panic mute to attached pipeline: %s", exc)

    def toggle_passthrough(self, enabled: bool | None = None) -> bool:
        """Toggle deliberate raw-voice passthrough. Never auto-engaged."""
        with self._lock:
            if enabled is None:
                self._is_passthrough = not self._is_passthrough
            else:
                self._is_passthrough = enabled
            logger.info("Passthrough state updated", is_passthrough=self._is_passthrough)
            return self._is_passthrough

    def set_running_state(self, running: bool) -> None:
        """Set whether live audio processing is actively running."""
        with self._lock:
            self._is_running = running
            if not running:
                self._device_error = None

    def handle_device_unplug(self, device_id: str) -> None:
        """Fail-closed response when a device unplug is signaled."""
        pipe = None
        with self._lock:
            if self._active_config:
                cfg = self._active_config
                if device_id in (cfg.input_device_id, cfg.output_device_id):
                    self._is_muted = True
                    self._device_error = (
                        f"Active audio device '{device_id}' was unplugged. Stream muted."
                    )
                    pipe = self._pipeline
                    logger.error(
                        "Device disconnected during active stream — Fail-closed mute engaged",
                        device_id=device_id,
                    )

        if pipe is not None:
            try:
                pipe.panic_mute()
            except Exception as exc:
                logger.warning("Failed to propagate device unplug mute to pipeline: %s", exc)


# Global singleton instance
_audio_router = AudioRouter()


def get_audio_router() -> AudioRouter:
    return _audio_router
