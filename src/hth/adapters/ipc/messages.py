"""Typed message definitions for HTH inter-process communication."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageType(str, Enum):
    HANDSHAKE_REQ = "HANDSHAKE_REQ"
    HANDSHAKE_ACK = "HANDSHAKE_ACK"
    COMMAND = "COMMAND"
    RESPONSE = "RESPONSE"
    HEARTBEAT_PING = "HEARTBEAT_PING"
    HEARTBEAT_PONG = "HEARTBEAT_PONG"
    EVENT = "EVENT"
    ERROR = "ERROR"


class CommandType(str, Enum):
    PING = "ping"
    LIST_DEVICES = "list_devices"
    START = "start"
    STOP = "stop"
    SET_PASSTHROUGH = "set_passthrough"
    SET_PARAMS = "set_params"
    SELECT_MODEL = "select_model"
    STATUS = "status"
    HEALTH = "health"
    SHUTDOWN = "shutdown"


@dataclass
class IPCMessage:
    """Envelope for all IPC messages between supervisor and child worker."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.COMMAND
    command: Optional[CommandType] = None
    payload: dict[str, Any] = field(default_factory=dict)
    token: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "command": self.command.value if self.command else None,
            "payload": self.payload,
            "token": self.token,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IPCMessage":
        msg_type_str = data.get("message_type")
        cmd_str = data.get("command")
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(msg_type_str) if msg_type_str else MessageType.COMMAND,
            command=CommandType(cmd_str) if cmd_str else None,
            payload=data.get("payload") or {},
            token=data.get("token"),
            timestamp=float(data.get("timestamp", time.time())),
            reply_to=data.get("reply_to"),
        )
