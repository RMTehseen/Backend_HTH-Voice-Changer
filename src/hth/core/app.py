"""FastAPI application factory.

Usage with uvicorn:
    uvicorn hth.core.app:create_app --factory --host 127.0.0.1 --port 8765
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hth.api.routers import (
    auth,
    devices,
    engine,
    events,
    health,
    meanvc2,
    models,
    noise,
    settings_router,
    system,
    voice_notes,
)
from hth.core.lifespan import lifespan
from hth.core.security import ALLOWED_ORIGINS, SecurityMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Each call returns a fresh instance — safe for use in tests.
    """
    app = FastAPI(
        lifespan=lifespan,
        title="HTH Real Voice Changer API",
        version="1.0.0",
        description=(
            "Local-only voice conversion backend. "
            "Binds to 127.0.0.1 only. No audio data leaves the machine."
        ),
    )

    # Security middleware (validates Host, Origin, and Session Token)
    app.add_middleware(SecurityMiddleware)

    # CORS — added AFTER SecurityMiddleware so that CORSMiddleware is outermost.
    # This guarantees that all responses (including 401/403 security rejections)
    # contain proper CORS headers for permitted local-first and Tauri origins
    # even when connecting to a remote Colab GPU tunnel.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(ALLOWED_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(auth.router)
    app.include_router(health.router)
    app.include_router(system.router)
    app.include_router(engine.router)
    app.include_router(devices.router)
    app.include_router(events.router)
    app.include_router(settings_router.router)
    app.include_router(models.router)
    app.include_router(voice_notes.router)
    app.include_router(noise.router)
    app.include_router(meanvc2.router, prefix="/api/v1/meanvc2")
    app.include_router(meanvc2.router, prefix="/api/meanvc2")

    return app
