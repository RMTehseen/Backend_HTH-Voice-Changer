"""Engine control endpoints — start, stop, status, passthrough."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from hth.api.dependencies import get_engine_service
from hth.domain.api_contracts import EngineInfo, LiveConfig, LiveStatus
from hth.services.engine_service import EngineService

router = APIRouter(prefix="/api/v1/engine", tags=["engine"])


class PassthroughRequest(BaseModel):
    enabled: bool


@router.get("/status", response_model=LiveStatus)
async def engine_status(svc: EngineService = Depends(get_engine_service)) -> LiveStatus:
    return await svc.get_status()


@router.get("/capabilities", response_model=EngineInfo)
async def engine_capabilities(svc: EngineService = Depends(get_engine_service)) -> EngineInfo:
    return await svc.get_health()


@router.post("/start", response_model=LiveStatus)
async def engine_start(
    config: LiveConfig,
    svc: EngineService = Depends(get_engine_service),
) -> LiveStatus:
    try:
        return await svc.start_conversion(config)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stop")
async def engine_stop(svc: EngineService = Depends(get_engine_service)) -> dict[str, Any]:
    try:
        await svc.stop_conversion()
        return {"ok": True}
    except ValueError as exc:
        if "IDLE" in str(exc).upper():
            return {"ok": True, "state": "IDLE"}
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/passthrough")
async def engine_passthrough(
    req: PassthroughRequest,
    svc: EngineService = Depends(get_engine_service),
) -> dict[str, Any]:
    try:
        await svc.enable_passthrough(req.enabled)
        return {"ok": True, "passthrough": req.enabled}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/panic-mute")
async def engine_panic_mute(svc: EngineService = Depends(get_engine_service)) -> dict[str, Any]:
    try:
        await svc.panic_mute()
        return {"ok": True, "muted": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/unmute", response_model=LiveStatus)
async def engine_unmute(svc: EngineService = Depends(get_engine_service)) -> LiveStatus:
    try:
        return await svc.un_mute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class SelectModelRequest(BaseModel):
    model_id: str


class EngineParamsRequest(BaseModel):
    params: dict[str, Any]


@router.post("/params")
async def update_engine_params(
    req: EngineParamsRequest,
    svc: EngineService = Depends(get_engine_service),
) -> dict[str, Any]:
    try:
        await svc.configure_runtime_params(req.params)
        return {"ok": True, "params": req.params}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/select-model")
async def engine_select_model(
    req: SelectModelRequest,
    svc: EngineService = Depends(get_engine_service),
) -> dict[str, Any]:
    try:
        await svc.select_model(req.model_id)
        return {"ok": True, "model_id": req.model_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
