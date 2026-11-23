"""Length-prefixed binary framing and JSON codec for IPC communication."""

import json
import struct

from hth.adapters.ipc.messages import IPCMessage
from hth.domain.errors import ProtocolViolationError

# Protocol Constants
PROTOCOL_VERSION: int = 1
HEADER_FORMAT: str = ">IB"  # 4-byte payload length (big-endian), 1-byte version
HEADER_SIZE: int = struct.calcsize(HEADER_FORMAT)
MAX_PAYLOAD_SIZE: int = 65536  # 64 KB strictly bounded maximum frame size


class IPCProtocol:
    """Encodes and decodes framed IPC messages with bounded validation."""

    @staticmethod
    def encode(message: IPCMessage) -> bytes:
        """Serializes an IPCMessage into length-prefixed bytes."""
        try:
            payload_json = json.dumps(message.to_dict()).encode("utf-8")
        except Exception as exc:
            raise ProtocolViolationError(f"Failed to serialize message: {exc}") from exc

        payload_len = len(payload_json)
        if payload_len > MAX_PAYLOAD_SIZE:
            raise ProtocolViolationError(
                f"Payload size {payload_len} exceeds bounded limit of {MAX_PAYLOAD_SIZE} bytes"
            )

        header = struct.pack(HEADER_FORMAT, payload_len, PROTOCOL_VERSION)
        return header + payload_json

    @staticmethod
    def decode(data: bytes) -> tuple[IPCMessage, int]:
        """Decodes one IPCMessage from buffer.

        Returns (message, consumed_bytes).
        Raises ProtocolViolationError on invalid headers, version mismatch, or exceeding bounds.
        """
        if len(data) < HEADER_SIZE:
            raise ProtocolViolationError("Insufficient bytes for protocol header")

        payload_len, version = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])

        if version != PROTOCOL_VERSION:
            raise ProtocolViolationError(
                f"Unsupported protocol version {version} (expected {PROTOCOL_VERSION})"
            )

        if payload_len > MAX_PAYLOAD_SIZE:
            raise ProtocolViolationError(
                f"Payload length {payload_len} exceeds maximum allowed {MAX_PAYLOAD_SIZE} bytes"
            )

        total_frame_len = HEADER_SIZE + payload_len
        if len(data) < total_frame_len:
            raise ProtocolViolationError("Frame truncated: payload data incomplete")

        payload_raw = data[HEADER_SIZE:total_frame_len]
        try:
            payload_dict = json.loads(payload_raw.decode("utf-8"))
            if not isinstance(payload_dict, dict):
                raise ProtocolViolationError("Decoded JSON root must be an object")
            message = IPCMessage.from_dict(payload_dict)
        except Exception as exc:
            if isinstance(exc, ProtocolViolationError):
                raise
            raise ProtocolViolationError(f"Corrupt frame JSON: {exc}") from exc

        return message, total_frame_len
