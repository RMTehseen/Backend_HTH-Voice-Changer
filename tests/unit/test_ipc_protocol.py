"""Unit tests for the IPC framing protocol, serialization bounds, and authentication."""

import struct

import pytest

from hth.adapters.ipc.connection import IPCConnection
from hth.adapters.ipc.messages import CommandType, IPCMessage, MessageType
from hth.adapters.ipc.protocol import (
    HEADER_FORMAT,
    MAX_PAYLOAD_SIZE,
    PROTOCOL_VERSION,
    IPCProtocol,
)
from hth.domain.errors import AuthenticationError, ProtocolViolationError


def test_ipc_encode_decode_roundtrip() -> None:
    """Verifies that an IPCMessage encodes and decodes losslessly."""
    msg = IPCMessage(
        message_type=MessageType.COMMAND,
        command=CommandType.START,
        payload={"sample_rate": 16000, "model": "jvs"},
        token="test-token-123",
    )
    encoded = IPCProtocol.encode(msg)
    decoded, consumed = IPCProtocol.decode(encoded)

    assert consumed == len(encoded)
    assert decoded.message_id == msg.message_id
    assert decoded.message_type == MessageType.COMMAND
    assert decoded.command == CommandType.START
    assert decoded.payload == {"sample_rate": 16000, "model": "jvs"}
    assert decoded.token == "test-token-123"


def test_ipc_bounded_payload_size_enforced() -> None:
    """Verifies that payloads exceeding MAX_PAYLOAD_SIZE (64KB) are rejected."""
    oversized_data = "x" * (MAX_PAYLOAD_SIZE + 100)
    msg = IPCMessage(
        message_type=MessageType.COMMAND,
        command=CommandType.SET_PARAMS,
        payload={"data": oversized_data},
    )

    with pytest.raises(ProtocolViolationError, match="exceeds bounded limit"):
        IPCProtocol.encode(msg)


def test_ipc_decode_incomplete_header() -> None:
    """Verifies rejection when fewer than HEADER_SIZE bytes are passed."""
    with pytest.raises(ProtocolViolationError, match="Insufficient bytes"):
        IPCProtocol.decode(b"\x00\x00")


def test_ipc_decode_version_mismatch() -> None:
    """Verifies rejection when protocol version does not match."""
    header = struct.pack(HEADER_FORMAT, 10, 99)  # version 99 != PROTOCOL_VERSION
    payload = b'{"msg": 1}'
    with pytest.raises(ProtocolViolationError, match="Unsupported protocol version"):
        IPCProtocol.decode(header + payload)


def test_ipc_decode_corrupt_json() -> None:
    """Verifies that corrupted JSON payload raises ProtocolViolationError."""
    payload = b"not-valid-json-content"
    header = struct.pack(HEADER_FORMAT, len(payload), PROTOCOL_VERSION)
    with pytest.raises(ProtocolViolationError, match="Corrupt frame JSON"):
        IPCProtocol.decode(header + payload)


def test_ipc_decode_payload_bounds_spoofing() -> None:
    """Verifies that header declaring length > MAX_PAYLOAD_SIZE is rejected."""
    header = struct.pack(HEADER_FORMAT, MAX_PAYLOAD_SIZE + 500, PROTOCOL_VERSION)
    with pytest.raises(ProtocolViolationError, match="exceeds maximum allowed"):
        IPCProtocol.decode(header + b"\x00" * 10)


def test_ipc_authentication_success(
    mock_pipe_pair: tuple[IPCConnection, IPCConnection], test_token: str
) -> None:
    """Verifies successful authentication handshake between client and server."""
    server_conn, client_conn = mock_pipe_pair

    # Run handshake sequentially
    import threading

    server_error = []

    def server_thread() -> None:
        try:
            server_conn.authenticate_server(test_token, timeout=2.0)
        except Exception as exc:
            server_error.append(exc)

    t = threading.Thread(target=server_thread)
    t.start()

    client_conn.authenticate_client(test_token, timeout=2.0)
    t.join(timeout=2.0)

    assert not server_error
    assert server_conn.is_authenticated
    assert client_conn.is_authenticated


def test_ipc_authentication_failure(
    mock_pipe_pair: tuple[IPCConnection, IPCConnection], test_token: str
) -> None:
    """Verifies authentication failure when invalid token is presented."""
    server_conn, client_conn = mock_pipe_pair

    import threading

    server_error = []

    def server_thread() -> None:
        try:
            server_conn.authenticate_server("valid-secret-token", timeout=2.0)
        except Exception as exc:
            server_error.append(exc)

    t = threading.Thread(target=server_thread)
    t.start()

    with pytest.raises(AuthenticationError):
        client_conn.authenticate_client("wrong-token-abc", timeout=2.0)

    t.join(timeout=2.0)
    assert len(server_error) == 1
    assert isinstance(server_error[0], AuthenticationError)
