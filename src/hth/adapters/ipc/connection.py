"""Connection transport wrapper for IPC communications with authentication."""

import asyncio
import time
from multiprocessing.connection import Connection
from typing import Optional

from hth.adapters.ipc.messages import IPCMessage, MessageType
from hth.adapters.ipc.protocol import IPCProtocol
from hth.domain.errors import AuthenticationError, ProtocolViolationError, TimeoutError


class IPCConnection:
    """Manages full-duplex message transmission over a Pipe connection with authentication."""

    def __init__(self, conn: Connection, session_token: Optional[str] = None) -> None:
        self._conn = conn
        self._session_token = session_token
        self._is_authenticated = False
        self._buffer = bytearray()

    @property
    def is_authenticated(self) -> bool:
        return self._is_authenticated

    def authenticate_server(self, expected_token: str, timeout: float = 5.0) -> bool:
        """Called by supervisor to verify the child authenticates with expected token."""
        msg = self.recv_message(timeout=timeout)
        if msg.message_type != MessageType.HANDSHAKE_REQ:
            raise AuthenticationError(
                f"Expected HANDSHAKE_REQ, got {msg.message_type}",
            )
        token_provided = msg.token or msg.payload.get("token")
        if token_provided != expected_token:
            self.send_message(
                IPCMessage(
                    message_type=MessageType.ERROR,
                    payload={"error": "Invalid session token"},
                    reply_to=msg.message_id,
                )
            )
            raise AuthenticationError("Invalid session token presented in handshake")

        self._is_authenticated = True
        self.send_message(
            IPCMessage(
                message_type=MessageType.HANDSHAKE_ACK,
                payload={"status": "authenticated"},
                reply_to=msg.message_id,
            )
        )
        return True

    def authenticate_client(self, token: str, timeout: float = 2.0) -> bool:
        """Called by child worker to perform handshake authentication with supervisor."""
        req = IPCMessage(
            message_type=MessageType.HANDSHAKE_REQ,
            token=token,
            payload={"token": token},
        )
        self.send_message(req)
        resp = self.recv_message(timeout=timeout)
        if resp.message_type != MessageType.HANDSHAKE_ACK:
            raise AuthenticationError(
                f"Handshake failed: {resp.payload.get('error', 'Unknown failure')}"
            )
        self._is_authenticated = True
        return True

    def send_message(self, message: IPCMessage) -> None:
        """Encodes and sends an IPCMessage over the transport."""
        frame = IPCProtocol.encode(message)
        try:
            self._conn.send_bytes(frame)
        except Exception as exc:
            raise ProtocolViolationError(f"Failed to transmit frame: {exc}") from exc

    def recv_message(self, timeout: Optional[float] = None) -> IPCMessage:
        """Receives and decodes an IPCMessage with an optional timeout in seconds."""
        deadline = time.time() + timeout if timeout is not None else None

        while True:
            # Check if existing buffer has a complete frame
            if len(self._buffer) >= 5:
                try:
                    msg, consumed = IPCProtocol.decode(bytes(self._buffer))
                    del self._buffer[:consumed]
                    return msg
                except ProtocolViolationError as exc:
                    if (
                        "incomplete" not in str(exc).lower()
                        and "insufficient" not in str(exc).lower()
                    ):
                        # Unrecoverable frame error; purge buffer to avoid poison loop
                        self._buffer.clear()
                        raise

            # Calculate remaining timeout
            remaining = None
            if deadline is not None:
                remaining = max(0.0, deadline - time.time())
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"Timed out waiting for message after {timeout} seconds",
                        timeout_seconds=timeout,
                    )

            # Wait for data on pipe
            try:
                has_data = self._conn.poll(remaining)
            except Exception as exc:
                raise ProtocolViolationError(f"Poll failed: {exc}") from exc

            if not has_data:
                raise TimeoutError(
                    f"Timed out waiting for message after {timeout} seconds",
                    timeout_seconds=timeout,
                )

            try:
                chunk = self._conn.recv_bytes()
                self._buffer.extend(chunk)
            except EOFError as exc:
                raise ProtocolViolationError("Connection closed by peer (EOF)") from exc
            except Exception as exc:
                raise ProtocolViolationError(f"Read error: {exc}") from exc

    async def async_send_message(self, message: IPCMessage) -> None:
        """Asynchronously send an IPC message."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.send_message, message)

    async def async_recv_message(self, timeout: Optional[float] = None) -> IPCMessage:
        """Asynchronously receive an IPC message with timeout."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.recv_message, timeout)

    def close(self) -> None:
        """Close the underlying connection."""
        try:
            self._conn.close()
        except Exception:
            pass
