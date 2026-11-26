"""MeanVC2 neural voice conversion API routes."""
from __future__ import annotations

import asyncio
import hmac
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from hth.api.dependencies import get_engine_service
from hth.core.logging import get_logger
from hth.core.security import ticket_store
from hth.core.settings import get_settings
from hth.domain.api_contracts import EngineState
from hth.services.engine_service import EngineService

logger = get_logger("hth.api.meanvc2")
router = APIRouter(tags=["meanvc2"])


def _get_meanvc2_adapter(engine_svc: EngineService = Depends(get_engine_service)) -> Any:
    """Get MeanVC2 adapter from engine service."""
    adapter = getattr(engine_svc, "_engine", None)
    if hasattr(adapter, "_inner"):
        adapter = adapter._inner
    if not hasattr(adapter, "create_voice_profile"):
        # If not current active engine, create an inspector instance
        from hth.adapters.meanvc2.adapter import MeanVC2VoiceAdapter
        from hth.adapters.meanvc2.config import MeanVC2Config
        return MeanVC2VoiceAdapter(MeanVC2Config())
    return adapter


class ProfileCreateRequest(BaseModel):
    display_name: str
    reference_wav_path: str
    consent_declared: bool
    consent_statement: str = ""
    language_hint: str = ""
    avatar_id: str = "female_eva"
    category: str = "Female"
    badge: str = "AI"


class ProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    avatar_id: Optional[str] = None
    category: Optional[str] = None
    badge: Optional[str] = None
    pitch_shift: Optional[float] = None
    timbre_ratio: Optional[float] = None
    volume_gain: Optional[float] = None
    sound_effect: Optional[str] = None
    favorite: Optional[bool] = None
    description: Optional[str] = None


class StreamStartRequest(BaseModel):
    input_device_id: Optional[str] = None
    output_device_id: Optional[str] = None
    monitor_device_id: Optional[str] = None
    enable_audio_streams: bool = True


class StreamMuteRequest(BaseModel):
    muted: bool


class MonitorControlRequest(BaseModel):
    enabled: bool
    volume: Optional[float] = None
    device_id: Optional[str] = None


class VolumeControlRequest(BaseModel):
    volume: float


class ProfileSelectRequest(BaseModel):
    profile_id: str


class ModelVariantRequest(BaseModel):
    variant: str  # "40ms" or "120ms"


def _detect_cpu_name() -> str:
    """Detect host CPU model name cleanly across Windows and Linux/Colab."""
    import platform

    if os.name != "nt":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        if name:
                            return name
        except OSError:
            pass
    proc = platform.processor() or platform.machine() or "CPU"
    return proc.strip()


@router.get("/status")
async def get_meanvc2_status(adapter: Any = Depends(_get_meanvc2_adapter)) -> dict[str, Any]:
    """Get MeanVC2 asset/model readiness status."""
    asset_status = adapter.get_asset_status()
    profiles = [p.to_dict() for p in adapter.list_voice_profiles()]
    worker_proc = getattr(adapter, "_worker_proc", None)
    worker_loaded = getattr(adapter, "_worker_loaded", None)
    target_set = getattr(adapter, "_worker_target_set", None)
    bridge = getattr(adapter, "_audio_bridge", None)

    # Dynamic hardware detection (supports local CPU fallback and Cloud GPU / CUDA)
    is_cuda = False
    cpu_model = _detect_cpu_name()
    is_cloud_mode = os.environ.get("HTH_CLOUD_MODE", "false").lower() in ("true", "1")
    device_name = f"Cloud CPU ({cpu_model})" if is_cloud_mode else f"{cpu_model} (CPU)"
    measured_rtf = getattr(adapter, "_last_rtf", 2.39) or (1.0 if is_cloud_mode else 2.39)
    try:
        import torch
        if torch.cuda.is_available():
            is_cuda = True
            device_name = f"NVIDIA {torch.cuda.get_device_name(0)}"
            measured_rtf = getattr(adapter, "_last_rtf", 0.0) or 0.08
    except Exception:
        pass

    return {
        "asset_status": asset_status,
        "profiles": profiles,
        "active_profile_id": getattr(adapter, "_active_profile_id", None),
        "model_variant": adapter._cfg.model_variant,
        "worker_running": bool(worker_proc and worker_proc.returncode is None),
        "worker_loaded": bool(worker_loaded and worker_loaded.is_set()),
        "target_set": bool(target_set and target_set.is_set()),
        "engine_state": adapter._state.value if hasattr(adapter._state, "value") else str(adapter._state),
        "last_error": getattr(adapter, "_last_error", None),
        "overruns": getattr(adapter, "_overruns", 0),
        "is_muted": getattr(adapter, "_is_muted", False),
        "monitor_enabled": getattr(bridge, "monitor_enabled", False) if bridge else False,
        "monitor_volume": getattr(bridge, "monitor_volume", 1.0) if bridge else 1.0,
        "output_volume": getattr(bridge, "output_volume", 1.0) if bridge else 1.0,
        "hardware_info": {
            "device": "cuda" if is_cuda else "cpu",
            "cpu_model": cpu_model,
            "device_name": device_name,
            "is_cuda_ready": is_cuda,
            "realtime_viable": is_cuda,
            "measured_rtf": measured_rtf,
            "recommendation": (
                f"{device_name} Active — Sub-40ms zero-latency real-time voice conversion running."
                if is_cuda
                else f"Running on {device_name} (RTF ~{measured_rtf:.2f}). NVIDIA GPU with CUDA required for sub-40ms zero-delay live calls."
            ),
        },
    }


@router.post("/profiles")
async def create_voice_profile(
    req: ProfileCreateRequest,
    adapter: Any = Depends(_get_meanvc2_adapter),
) -> dict[str, Any]:
    """Create a voice profile from an existing WAV file path."""
    try:
        profile = await adapter.create_voice_profile(
            display_name=req.display_name,
            reference_wav_path=req.reference_wav_path,
            consent_declared=req.consent_declared,
            consent_statement=req.consent_statement,
            language_hint=req.language_hint,
            avatar_id=req.avatar_id,
            category=req.category,
            badge=req.badge,
        )
        return profile.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/profiles/upload")
async def upload_voice_profile(
    display_name: str = Form(...),
    consent_declared: bool = Form(...),
    consent_statement: str = Form(""),
    language_hint: str = Form(""),
    avatar_id: str = Form("female_eva"),
    category: str = Form("Female"),
    badge: str = Form("AI"),
    file: UploadFile = File(...),
    adapter: Any = Depends(_get_meanvc2_adapter),
) -> dict[str, Any]:
    """Upload a WAV file and create a voice profile."""
    local_app_data = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    upload_dir = Path(local_app_data) / "HTH" / "MeanVC2Profiles" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "ref.wav").suffix or ".wav"
    tmp_path = upload_dir / f"ref_{os.urandom(8).hex()}{suffix}"
    try:
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50 MB limit
            raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
        with open(tmp_path, "wb") as f:
            f.write(content)

        profile = await adapter.create_voice_profile(
            display_name=display_name,
            reference_wav_path=str(tmp_path),
            consent_declared=consent_declared,
            consent_statement=consent_statement,
            language_hint=language_hint,
            avatar_id=avatar_id,
            category=category,
            badge=badge,
        )
        return profile.to_dict()
    except ValueError as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/profiles")
async def list_voice_profiles(adapter: Any = Depends(_get_meanvc2_adapter)) -> list[dict[str, Any]]:
    """List all saved voice profiles."""
    profiles = adapter.list_voice_profiles()
    return [p.to_dict() for p in profiles]


@router.put("/profiles/{profile_id}")
async def update_voice_profile(
    profile_id: str,
    req: ProfileUpdateRequest,
    adapter: Any = Depends(_get_meanvc2_adapter),
) -> dict[str, Any]:
    """Update tuning parameters or visual metadata of a profile."""
    updates = req.model_dump(exclude_unset=True)
    updated = adapter.update_voice_profile(profile_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Profile not found")
    return updated.to_dict()


@router.post("/monitor")
async def toggle_headphone_monitor(
    req: MonitorControlRequest,
    adapter: Any = Depends(_get_meanvc2_adapter),
) -> dict[str, Any]:
    """Toggle or configure real-time headphone monitoring."""
    if req.device_id is not None and hasattr(adapter, "set_monitor_device"):
        adapter.set_monitor_device(req.device_id)
    adapter.set_monitor_enabled(req.enabled)
    if req.volume is not None:
        adapter.set_monitor_volume(req.volume)
    bridge = getattr(adapter, "_audio_bridge", None)
    return {
        "monitor_enabled": getattr(bridge, "monitor_enabled", req.enabled),
        "monitor_volume": getattr(bridge, "monitor_volume", 1.0),
        "monitor_device_id": getattr(bridge, "monitor_device_id", req.device_id),
    }


@router.post("/volume")
async def set_master_volume(
    req: VolumeControlRequest,
    adapter: Any = Depends(_get_meanvc2_adapter),
) -> dict[str, Any]:
    """Set master virtual microphone output volume gain."""
    adapter.set_output_volume(req.volume)
    bridge = getattr(adapter, "_audio_bridge", None)
    return {"output_volume": getattr(bridge, "output_volume", req.volume)}


@router.post("/stream/start")
async def start_stream(
    req: StreamStartRequest,
    adapter: Any = Depends(_get_meanvc2_adapter),
) -> dict[str, Any]:
    """Start MeanVC2 live audio streaming with WASAPI mic capture and VB-CABLE output."""
    from hth.domain.api_contracts import LiveConfig
    if not adapter._active_profile_id:
        profiles = adapter.list_voice_profiles()
        if profiles:
            await adapter.select_voice_profile(profiles[0].profile_id)
    if not adapter._cfg.all_assets_ready():
        missing = [k for k, v in adapter._cfg.check_assets().items() if not v]
        raise HTTPException(
            status_code=400,
            detail=f"MeanVC2 assets missing: {', '.join(missing)}. Download weights first.",
        )

    cfg = LiveConfig(
        input_device_id=req.input_device_id,
        output_device_id=req.output_device_id,
        monitor_device_id=req.monitor_device_id,
        enable_audio_streams=req.enable_audio_streams,
    )
    try:
        if hasattr(adapter, "start_stream"):
            await adapter.start_stream(cfg)
        else:
            await adapter.start(cfg)
        return {"status": "started", "engine_state": "RUNNING"}
    except RuntimeError as e:
        logger.warning("MeanVC2 stream start rejected", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error("Failed to start MeanVC2 stream", error=str(e), traceback=tb)
        raise HTTPException(status_code=500, detail=str(e) or repr(e)) from e


@router.post("/stream/stop")
async def stop_stream(
    adapter: Any = Depends(_get_meanvc2_adapter),
) -> dict[str, Any]:
    """Stop MeanVC2 live audio streaming."""
    try:
        await adapter.stop()
        return {"status": "stopped", "engine_state": "IDLE"}
    except Exception as e:
        logger.error("Failed to stop MeanVC2 stream", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stream/mute")
async def toggle_stream_mute(
    req: StreamMuteRequest,
    adapter: Any = Depends(_get_meanvc2_adapter),
) -> dict[str, Any]:
    """Engage or disengage fail-closed stream mute."""
    if req.muted:
        await adapter.panic_mute()
    else:
        await adapter.unmute()
    return {"is_muted": getattr(adapter, "_is_muted", False)}


@router.delete("/profiles/{profile_id}")
async def delete_voice_profile(profile_id: str, adapter: Any = Depends(_get_meanvc2_adapter)) -> dict[str, str]:
    """Delete a voice profile."""
    success = adapter.delete_voice_profile(profile_id)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"deleted": profile_id}


@router.post("/profiles/{profile_id}/select")
async def select_voice_profile(profile_id: str, adapter: Any = Depends(_get_meanvc2_adapter)) -> dict[str, str]:
    """Select a voice profile as the active target speaker."""
    try:
        await adapter.select_voice_profile(profile_id)
        return {"selected": profile_id}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/initialize")
async def initialize_adapter(adapter: Any = Depends(_get_meanvc2_adapter)) -> dict[str, str]:
    """Initialize the MeanVC2 worker subprocess."""
    try:
        await adapter.initialize()
        return {"status": "initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/download-assets")
async def download_assets(adapter: Any = Depends(_get_meanvc2_adapter)) -> dict[str, Any]:
    """Trigger HF asset download and upstream MeanVC2 code clone (background task)."""
    from hth.adapters.meanvc2.asset_manager import MeanVC2AssetManager

    mgr = MeanVC2AssetManager(adapter._cfg.model_dir)
    target_source_dir = adapter._cfg.source_dir or str(Path.cwd() / "MeanVC2-main")
    adapter._cfg.source_dir = target_source_dir
    variant = getattr(adapter._cfg, "model_variant", "40ms")

    async def _download() -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, mgr.ensure_source_code, target_source_dir)
        await loop.run_in_executor(None, mgr.download_hf_assets, variant)
        await loop.run_in_executor(None, mgr.download_wavlm)

    asyncio.create_task(_download())
    return {
        "ok": True,
        "status": "download_started",
        "message": "Model weights and MeanVC2 source download started in background.",
        "model_dir": str(adapter._cfg.model_dir),
        "source_dir": target_source_dir,
    }


@router.get("/assets")
async def get_asset_status(adapter: Any = Depends(_get_meanvc2_adapter)) -> dict[str, Any]:
    """Get detailed asset download/existence status."""
    from hth.adapters.meanvc2.asset_manager import MeanVC2AssetManager
    mgr = MeanVC2AssetManager(adapter._cfg.model_dir)
    statuses = mgr.status()
    return {
        "assets": [
            {
                "name": s.name,
                "local_path": s.local_path,
                "exists": s.exists,
                "size_bytes": s.size_bytes,
                "source_url": s.source_url,
            }
            for s in statuses
        ],
        "model_dir": str(adapter._cfg.model_dir),
    }


@router.post("/variant")
async def set_model_variant(req: ModelVariantRequest, adapter: Any = Depends(_get_meanvc2_adapter)) -> dict[str, str]:
    """Switch between 40ms (low latency) and 120ms (high quality) model variants."""
    if req.variant not in ("40ms", "120ms"):
        raise HTTPException(status_code=400, detail="variant must be '40ms' or '120ms'")
    adapter._cfg.model_variant = req.variant
    if hasattr(adapter, "_worker_loaded"):
        adapter._worker_loaded.clear()
    if hasattr(adapter, "_worker_target_set"):
        adapter._worker_target_set.clear()
    return {"variant": req.variant, "note": "Restart engine to apply new variant"}


_MAX_AUDIO_WS_FRAME_BYTES = 262144  # 256 KB guard per binary PCM chunk


@router.websocket("/ws/audio")
async def websocket_audio_stream(
    websocket: WebSocket,
    ticket: str = "",
) -> None:
    """Real-time bidirectional 16kHz float32 PCM WebSocket for remote/Colab GPU audio conversion.

    Authenticated via single-use ?ticket= (from POST /api/v1/auth/ws-ticket) or hth_session cookie.
    Fail-closed: on engine error or mute, stops emitting converted audio cleanly.
    """
    settings = get_settings()

    if ticket:
        if not ticket_store.redeem_ws_ticket(ticket):
            await websocket.close(code=1008)
            return
    else:
        cookie_token = websocket.cookies.get("hth_session", "")
        if not cookie_token or not hmac.compare_digest(
            cookie_token.encode("utf-8"), settings.session_token.encode("utf-8")
        ):
            await websocket.close(code=1008)
            return

    engine_svc = get_engine_service()
    adapter = getattr(engine_svc, "_engine", None)
    if hasattr(adapter, "_inner"):
        adapter = adapter._inner
    if not hasattr(adapter, "subscribe_audio_out") or not hasattr(adapter, "push_remote_audio"):
        await websocket.close(code=1011)
        return

    await websocket.accept()
    out_queue = adapter.subscribe_audio_out()

    async def _send_loop() -> None:
        try:
            while True:
                pcm_bytes = await out_queue.get()
                if adapter._state == EngineState.ERROR:
                    break
                await websocket.send_bytes(pcm_bytes)
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception:
            pass

    sender_task = asyncio.create_task(_send_loop())
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if not data or len(data) > _MAX_AUDIO_WS_FRAME_BYTES or (len(data) % 4) != 0:
                continue
            if adapter._state == EngineState.ERROR:
                break
            samples = np.frombuffer(data, dtype=np.float32).copy()
            await adapter.push_remote_audio(samples)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as e:
        logger.warning("Remote audio WebSocket error", error=str(e))
    finally:
        sender_task.cancel()
        adapter.unsubscribe_audio_out(out_queue)
