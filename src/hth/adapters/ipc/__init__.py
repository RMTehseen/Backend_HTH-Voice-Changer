"""Inter-process communication framework for engine isolation."""

from hth.adapters.ipc.connection import IPCConnection
from hth.adapters.ipc.messages import CommandType, IPCMessage, MessageType
from hth.adapters.ipc.protocol import MAX_PAYLOAD_SIZE, PROTOCOL_VERSION, IPCProtocol

__all__ = [
    "IPCMessage",
    "MessageType",
    "CommandType",
    "IPCProtocol",
    "IPCConnection",
    "MAX_PAYLOAD_SIZE",
    "PROTOCOL_VERSION",
]
