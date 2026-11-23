"""Domain exception hierarchy for HTH audio engine and adapters."""

from typing import Any, Optional


class EngineError(Exception):
    """Base exception for all audio engine and adapter errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class EngineCrashError(EngineError):
    """Raised when the engine child process crashes or terminates unexpectedly."""

    def __init__(
        self,
        message: str = "Engine child process crashed",
        exit_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if exit_code is not None:
            details["exit_code"] = exit_code
        super().__init__(message, details)
        self.exit_code = exit_code


class DeviceError(EngineError):
    """Raised when an audio input or output device is unavailable or disconnected."""

    def __init__(
        self,
        message: str = "Audio device error",
        device_id: Optional[int | str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if device_id is not None:
            details["device_id"] = device_id
        super().__init__(message, details)
        self.device_id = device_id


class ModelValidationError(EngineError):
    """Raised when a voice model fails validation checks (hash, magic bytes, licensing)."""

    def __init__(
        self,
        message: str = "Model validation failed",
        model_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if model_id is not None:
            details["model_id"] = model_id
        super().__init__(message, details)
        self.model_id = model_id


class ProtocolViolationError(EngineError):
    """Raised when an IPC message exceeds bounded size, is corrupt, or violates protocol."""


class AuthenticationError(EngineError):
    """Raised when an IPC connection fails authentication or presents an invalid token."""


class CommandError(EngineError):
    """Raised when the engine child process explicitly returns an error for a command."""

    def __init__(
        self,
        message: str,
        command: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if command is not None:
            details["command"] = command
        super().__init__(message, details)
        self.command = command


class InvalidStateError(EngineError):
    """Raised when an operation is requested that is illegal for the current engine state."""

    def __init__(
        self,
        message: str,
        current_state: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if current_state is not None:
            details["current_state"] = current_state
        super().__init__(message, details)
        self.current_state = current_state


class TimeoutError(EngineError):
    """Raised when an IPC command or heartbeat response exceeds its deadline."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds


class IncompatibleModelError(ModelValidationError):
    """Raised when a voice model format is incompatible with the requested engine adapter."""

    def __init__(
        self,
        message: str = "Voice model format is incompatible with this engine adapter",
        model_id: Optional[str] = None,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if expected_format:
            details["expected_format"] = expected_format
        if actual_format:
            details["actual_format"] = actual_format
        super().__init__(message, model_id=model_id, details=details)
        self.expected_format = expected_format
        self.actual_format = actual_format


class AudioFormatError(EngineError):
    """Raised when an input or output audio format is unsupported, corrupted, or missing a required codec."""

    def __init__(
        self,
        message: str = "Unsupported or invalid audio format",
        codec: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if codec:
            details["codec"] = codec
        super().__init__(message, details)
        self.codec = codec


class ResourceConflictError(InvalidStateError):
    """Raised when offline processing and live streaming compete for shared CPU/engine resources."""

    def __init__(
        self,
        message: str = "Resource conflict between live engine and offline conversion",
        competing_task: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if competing_task:
            details["competing_task"] = competing_task
        super().__init__(message, details=details)
        self.competing_task = competing_task


class JobCancelledError(EngineError):
    """Raised when an offline conversion job is cooperatively cancelled by the user."""

    def __init__(
        self,
        message: str = "Offline conversion job was cancelled",
        job_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if job_id:
            details["job_id"] = job_id
        super().__init__(message, details)
        self.job_id = job_id


class RoutingConflictError(EngineError):
    """Raised when an audio routing configuration creates an accidental feedback loop or hardware conflict."""

    def __init__(
        self,
        message: str = "Audio routing conflict or feedback loop detected",
        errors: Optional[list[str]] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        details = details or {}
        if errors:
            details["errors"] = errors
        super().__init__(message, details)
        self.errors = errors or []


class LicensingPolicyViolation(EngineError):
    """Raised when an engine adapter or voice model violates licensing or commercial distribution rules."""

    def __init__(
        self,
        message: str = "Licensing policy violation detected",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details or {})


class AudioProcessingError(EngineError):
    """Raised when an internal audio DSP processing or library error occurs."""

    def __init__(
        self,
        message: str = "Audio processing or DSP error",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details or {})
