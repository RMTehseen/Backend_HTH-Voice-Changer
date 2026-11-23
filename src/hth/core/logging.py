"""Structured logging configuration using structlog with privacy & redaction enforcement."""

import logging
import os
import re
import sys
import traceback
from logging.handlers import RotatingFileHandler
from typing import Any, cast

import structlog

from hth.core.settings import get_settings

# Sensitive keys that must NEVER be written to logs (Rule S-10)
_REDACTED_KEYS = {
    "token",
    "password",
    "secret",
    "audio_data",
    "session_token",
    "x_hth_token",
    "authorization",
    "samples",
    "indata",
    "outdata",
    "raw_buffer",
    "audio_bytes",
    "bearer_token",
    "ticket",
    "ws_ticket",
    "bootstrap_ticket",
}

_USER_PROFILE = os.environ.get("USERPROFILE", os.path.expanduser("~"))

# Pattern for redacting sensitive query credentials (such as ?ticket= or &ticket=) from logged URLs/messages
_QUERY_REDACT_REGEX = re.compile(
    r"([?&](?:ticket|token)=)[^&\s\"\'<>]+",
    re.IGNORECASE,
)


def redact_query_params(val: str) -> str:
    """Redacts sensitive query parameter values (e.g. ?ticket=... or ?token=...) from URLs and log strings."""
    if not val or not isinstance(val, str):
        return val
    return _QUERY_REDACT_REGEX.sub(r"\g<1>[REDACTED]", val)


class TicketRedactionFilter(logging.Filter):
    """Ensures raw query parameter credentials like ?ticket= are never written to access, application, or error logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Wrap getMessage() to redact sensitive query params after formatting
            orig_get_message = record.getMessage

            def _clean_get_message() -> str:
                return redact_query_params(orig_get_message())

            record.getMessage = _clean_get_message

            # Redact string args in-place preserving tuple/dict structure so Uvicorn unpacks cleanly
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact_query_params(a) if isinstance(a, str) else a for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: redact_query_params(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }

            if record.exc_info and not record.exc_text:
                formatted_tb = "".join(traceback.format_exception(*record.exc_info))
                record.exc_text = redact_query_params(formatted_tb)
            elif record.exc_text:
                record.exc_text = redact_query_params(record.exc_text)
        except Exception:
            pass
        return True


def _sanitize_value(val: Any) -> Any:
    """Recursively sanitize paths and strip sensitive query parameters/content from values."""
    if isinstance(val, str):
        if _USER_PROFILE and _USER_PROFILE in val:
            val = val.replace(_USER_PROFILE, "[USERPROFILE]")
        return redact_query_params(val)
    elif isinstance(val, dict):
        return {k: _sanitize_value(v) for k, v in val.items() if k.lower() not in _REDACTED_KEYS}
    elif isinstance(val, list):
        return [_sanitize_value(item) for item in val]
    return val


def _redact_sensitive(logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop sensitive keys and sanitize paths before rendering (Rule S-10)."""
    clean_dict: dict[str, Any] = {}
    for key, value in event_dict.items():
        if key.lower() in _REDACTED_KEYS:
            continue
        clean_dict[key] = _sanitize_value(value)
    return clean_dict


def configure_logging() -> None:
    """Configure structlog and standard logging. Call exactly once at lifespan startup."""
    settings = get_settings()
    is_debug = settings.log_level.upper() in ("DEBUG",) or settings.reload

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        _redact_sensitive,
    ]

    console_renderer: Any = (
        structlog.dev.ConsoleRenderer() if is_debug else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            console_renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    redaction_filter = TicketRedactionFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(redaction_filter)

    handlers: list[logging.Handler] = [console_handler]

    # File logging with rotation in %LOCALAPPDATA%\HTH\logs
    log_dir = settings.log_dir
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "hth.log")
            file_formatter = structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
                foreign_pre_chain=shared_processors,
            )
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(redaction_filter)
            handlers.append(file_handler)
        except OSError:
            # If filesystem is read-only or restricted, fall back to console only
            pass

    root = logging.getLogger()
    # Avoid duplicate handlers on hot-reload
    root.handlers = handlers
    root.setLevel(settings.log_level.upper())
    root.addFilter(redaction_filter)

    # Attach redaction filter specifically to uvicorn access and error loggers
    for uvicorn_logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(uvicorn_logger_name).addFilter(redaction_filter)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for the given name."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
