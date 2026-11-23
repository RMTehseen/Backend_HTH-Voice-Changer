"""API Routers package."""

from hth.api.routers import (
    auth,
    devices,
    engine,
    events,
    health,
    models,
    noise,
    settings_router,
    system,
    voice_notes,
)

__all__ = [
    "auth",
    "health",
    "system",
    "engine",
    "devices",
    "events",
    "settings_router",
    "models",
    "voice_notes",
    "noise",
]
