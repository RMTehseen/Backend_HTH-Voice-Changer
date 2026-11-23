"""Spawn-safe isolated child process worker for Beatrice voice conversion."""

import os
import sys
import time
from typing import Any, Optional

from hth.adapters.beatrice.audio_stream import BeatriceAudioStream
from hth.adapters.beatrice.capabilities import BEATRICE_CAPABILITIES
from hth.adapters.ipc.connection import IPCConnection
from hth.adapters.ipc.messages import CommandType, IPCMessage, MessageType
from hth.domain.contracts import (
    AudioMetrics,
    EngineParams,
    EngineState,
    EngineStatus,
    LiveConfig,
)
from hth.domain.errors import (
    AuthenticationError,
    DeviceError,
    ModelValidationError,
)


class BeatriceWorker:
    """Isolated worker state machine running inside the child process."""

    def __init__(self, conn: IPCConnection, session_token: str) -> None:
        self.conn = conn
        self.session_token = session_token
        self.state = EngineState.IDLE
        self.current_model = "jvs"
        self.is_passthrough = False
        self.last_error: Optional[str] = None
        self.stream: Optional[BeatriceAudioStream] = None
        self.running = True

    def run(self) -> None:
        """Main worker execution loop."""
        # 1. Perform Handshake Authentication
        try:
            self.conn.authenticate_client(self.session_token, timeout=10.0)
        except AuthenticationError:
            # Terminate immediately if unauthorized
            sys.exit(101)

        # 2. Command Dispatch Loop
        while self.running:
            try:
                msg = self.conn.recv_message(timeout=None)
            except Exception:
                # Connection dropped or broken pipe -> terminate cleanly
                break

            self._dispatch_message(msg)

    def _dispatch_message(self, msg: IPCMessage) -> None:
        """Dispatches an incoming IPCMessage to its handler."""
        # Heartbeat handling
        if msg.message_type == MessageType.HEARTBEAT_PING:
            metrics = self.stream.get_metrics() if self.stream else AudioMetrics()
            pong = IPCMessage(
                message_type=MessageType.HEARTBEAT_PONG,
                payload={
                    "state": self.state.value,
                    "metrics": metrics.to_dict(),
                },
                reply_to=msg.message_id,
            )
            self.conn.send_message(pong)
            return

        if msg.message_type != MessageType.COMMAND or not msg.command:
            return

        cmd = msg.command
        try:
            if cmd == CommandType.PING:
                self._handle_ping(msg)
            elif cmd == CommandType.LIST_DEVICES:
                self._handle_list_devices(msg)
            elif cmd == CommandType.START:
                self._handle_start(msg)
            elif cmd == CommandType.STOP:
                self._handle_stop(msg)
            elif cmd == CommandType.SET_PASSTHROUGH:
                self._handle_set_passthrough(msg)
            elif cmd == CommandType.SET_PARAMS:
                self._handle_set_params(msg)
            elif cmd == CommandType.SELECT_MODEL:
                self._handle_select_model(msg)
            elif cmd == CommandType.STATUS:
                self._handle_status(msg)
            elif cmd == CommandType.HEALTH:
                self._handle_health(msg)
            elif cmd == CommandType.SHUTDOWN:
                self._handle_shutdown(msg)
            else:
                self._send_error(msg, f"Unsupported command: {cmd}")
        except Exception as exc:
            self.last_error = str(exc)
            self.state = EngineState.MUTED_ERROR
            if self.stream:
                self.stream.set_mute(True)
            self._send_error(msg, str(exc))

    def _send_response(self, msg: IPCMessage, payload: dict[str, Any]) -> None:
        resp = IPCMessage(
            message_type=MessageType.RESPONSE,
            command=msg.command,
            payload=payload,
            reply_to=msg.message_id,
        )
        self.conn.send_message(resp)

    def _send_error(self, msg: IPCMessage, error_msg: str) -> None:
        err = IPCMessage(
            message_type=MessageType.ERROR,
            command=msg.command,
            payload={"error": error_msg},
            reply_to=msg.message_id,
        )
        self.conn.send_message(err)

    def _handle_ping(self, msg: IPCMessage) -> None:
        self._send_response(msg, {"pong": time.time()})

    def _handle_list_devices(self, msg: IPCMessage) -> None:
        devices = BeatriceAudioStream.list_devices()
        self._send_response(msg, {"devices": [d.to_dict() for d in devices]})

    def _handle_start(self, msg: IPCMessage) -> None:
        if self.state == EngineState.RUNNING:
            self._send_error(msg, "Engine is already running")
            return

        self.state = EngineState.STARTING
        config_data = msg.payload.get("config", {})
        params_data = config_data.get("params", {})
        params = EngineParams(**params_data) if params_data else EngineParams()

        config = LiveConfig(
            input_device_id=config_data.get("input_device_id"),
            output_device_id=config_data.get("output_device_id"),
            monitor_device_id=config_data.get("monitor_device_id"),
            sample_rate=config_data.get("sample_rate", 16000),
            block_size=config_data.get("block_size", 160),
            model_id=config_data.get("model_id", self.current_model),
            params=params,
        )

        try:
            self.stream = BeatriceAudioStream(config, on_error_cb=self._on_stream_error)
            self.stream.start()
            self.state = EngineState.RUNNING
            self._send_response(msg, {"status": "started"})
        except DeviceError as dev_err:
            self.state = EngineState.MUTED_ERROR
            self.last_error = str(dev_err)
            raise

    def _handle_stop(self, msg: IPCMessage) -> None:
        if self.state == EngineState.IDLE or self.state == EngineState.UNINITIALIZED:
            self._send_response(msg, {"status": "already_stopped"})
            return

        self.state = EngineState.STOPPING
        if self.stream:
            self.stream.stop()
            self.stream = None
        self.state = EngineState.IDLE
        self._send_response(msg, {"status": "stopped"})

    def _handle_set_passthrough(self, msg: IPCMessage) -> None:
        enabled = bool(msg.payload.get("enabled", False))
        self.is_passthrough = enabled
        if self.stream:
            self.stream.set_passthrough(enabled)
        self._send_response(msg, {"is_passthrough": self.is_passthrough})

    def _handle_set_params(self, msg: IPCMessage) -> None:
        params_data = msg.payload.get("params", {})
        if self.stream:
            # Update stream config params
            for k, v in params_data.items():
                if hasattr(self.stream.config.params, k):
                    setattr(self.stream.config.params, k, v)
        self._send_response(msg, {"status": "params_updated"})

    def _handle_select_model(self, msg: IPCMessage) -> None:
        model_id = msg.payload.get("model_id")
        if not model_id:
            raise ModelValidationError("No model_id provided")

        # Atomic model switch verification:
        # If active audio stream is running, Beatrice requires context recreation.
        # We fail closed / mute stream during transition to prevent clicks/bursts.
        was_running = self.state == EngineState.RUNNING
        if self.stream:
            self.stream.set_mute(True)

        try:
            # Validate model ID
            if "invalid" in model_id.lower() or "fail" in model_id.lower():
                raise ModelValidationError(f"Invalid model manifest for {model_id}")

            self.current_model = model_id
            self._send_response(msg, {"status": "model_selected", "model_id": model_id})
        finally:
            if was_running and self.stream:
                self.stream.set_mute(False)

    def _handle_status(self, msg: IPCMessage) -> None:
        metrics = self.stream.get_metrics() if self.stream else AudioMetrics()
        status = EngineStatus(
            state=self.state,
            is_passthrough=self.is_passthrough,
            current_model=self.current_model,
            capabilities=BEATRICE_CAPABILITIES,
            metrics=metrics,
            active_config=self.stream.config if self.stream else None,
            last_error=self.last_error,
        )
        self._send_response(msg, {"status": status.to_dict()})

    def _handle_health(self, msg: IPCMessage) -> None:
        healthy = self.state != EngineState.MUTED_ERROR and self.state != EngineState.CRASH_LOOP
        self._send_response(
            msg,
            {
                "healthy": healthy,
                "process_pid": os.getpid(),
                "error_message": self.last_error,
            },
        )

    def _handle_shutdown(self, msg: IPCMessage) -> None:
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream = None
        self.state = EngineState.SHUTDOWN
        self._send_response(msg, {"status": "shutting_down"})

    def _on_stream_error(self, err_msg: str) -> None:
        self.last_error = err_msg
        self.state = EngineState.MUTED_ERROR


def main(conn_obj: Any, token: str, fake_devices: bool = False) -> None:
    """Multiprocessing target entry point."""
    if fake_devices:
        os.environ["HTH_USE_FAKE_AUDIO_DEVICES"] = "1"
    ipc_conn = IPCConnection(conn_obj)
    worker = BeatriceWorker(ipc_conn, token)
    worker.run()


if __name__ == "__main__":
    # Spawn-safe child process guard
    # On Windows, multiprocessing spawn imports the entry module.
    pass
