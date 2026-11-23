"""Runtime settings read/write endpoint.

session_token is NEVER exposed via the API (rule S-02).
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from hth.core.settings import get_settings

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_ADAPTERS = {"fake"}  # expand in later phases


class SettingsUpdate(BaseModel):
    log_level: str | None = None
    engine_adapter: str | None = None

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str | None) -> str | None:
        if v is not None and v.upper() not in _VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {_VALID_LOG_LEVELS}")
        return v.upper() if v else v

    @field_validator("engine_adapter")
    @classmethod
    def validate_adapter(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_ADAPTERS:
            raise ValueError(f"engine_adapter must be one of {_VALID_ADAPTERS}")
        return v


def _safe_settings_dict() -> dict[str, Any]:
    settings = get_settings()
    data = settings.model_dump()
    data["session_token"] = "[REDACTED]"  # never expose the token
    return data


@router.get("/")
async def get_current_settings() -> dict[str, Any]:
    return _safe_settings_dict()


@router.put("/")
async def update_settings(update: SettingsUpdate) -> dict[str, Any]:
    """Update in-memory settings. Changes are not persisted across restarts."""
    settings = get_settings()
    if update.log_level is not None:
        object.__setattr__(settings, "log_level", update.log_level)
    if update.engine_adapter is not None:
        object.__setattr__(settings, "engine_adapter", update.engine_adapter)
    return _safe_settings_dict()
