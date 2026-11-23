"""System information and lifecycle endpoints."""

import asyncio
import os
import platform
import signal
import sys
from typing import Any

import psutil
from fastapi import APIRouter, BackgroundTasks

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/")
@router.get("/info")
async def system_info() -> dict[str, Any]:
    mem = psutil.virtual_memory()
    # psutil.cpu_percent(interval=0.1) blocks briefly — acceptable for an on-demand endpoint
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_mb": mem.used / 1024 / 1024,
        "ram_total_mb": mem.total / 1024 / 1024,
        "ram_percent": mem.percent,
    }


def _trigger_shutdown() -> None:
    """Send termination signal to trigger graceful lifespan shutdown."""
    pid = os.getpid()
    if sys.platform == "win32":
        # On Windows, SIGTERM triggers python shutdown handling
        os.kill(pid, signal.SIGTERM)
    else:
        os.kill(pid, signal.SIGTERM)


@router.post("/shutdown")
async def shutdown(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Authenticated endpoint to trigger graceful backend shutdown."""

    # Run shutdown in background task after response is sent
    async def _delayed_shutdown() -> None:
        await asyncio.sleep(0.5)
        _trigger_shutdown()

    background_tasks.add_task(_delayed_shutdown)
    return {"status": "shutting_down", "message": "HTH backend shutdown initiated"}
