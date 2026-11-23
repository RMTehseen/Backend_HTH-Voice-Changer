"""Health liveness endpoint — exempt from token auth."""

import datetime
from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
@router.get("/api/v1/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
