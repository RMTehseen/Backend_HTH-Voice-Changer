"""Voice Note Conversion API Router (Phase 9).

Endpoints:
- GET /api/v1/voice-notes/live-guide — Live routing setup guide for WhatsApp Desktop
- GET /api/v1/voice-notes/capabilities — Local ffmpeg and codec capability probing
- POST /api/v1/voice-notes/convert — Submit an audio file for offline conversion
- GET /api/v1/voice-notes/jobs/{job_id} — Query progress and status
- DELETE /api/v1/voice-notes/jobs/{job_id} — Cancel job and clean temporary files
- GET /api/v1/voice-notes/jobs/{job_id}/source-audio — Stream source audio for A/B preview
- GET /api/v1/voice-notes/jobs/{job_id}/converted-audio — Stream converted audio for A/B preview
- GET /api/v1/voice-notes/jobs/{job_id}/download — Download exported audio
- POST /api/v1/voice-notes/cleanup — Trigger 24h retention sweep
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from hth.api.dependencies import get_voice_note_service
from hth.core.logging import get_logger
from hth.domain.api_contracts import (
    AudioContainerFormat,
    LiveRouteGuideResponse,
    RetentionCleanupResponse,
    SupportedModelFormat,
    VoiceNoteCapabilities,
    VoiceNoteJobCreate,
    VoiceNoteJobStatus,
)
from hth.domain.errors import (
    AudioFormatError,
    IncompatibleModelError,
    ResourceConflictError,
)
from hth.services.voice_note_service import VoiceNoteService

logger = get_logger("hth.api.routers.voice_notes")

router = APIRouter(prefix="/api/v1/voice-notes", tags=["voice-notes"])

MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}


@router.get("/live-guide", response_model=LiveRouteGuideResponse)
async def get_live_route_guide(
    service: VoiceNoteService = Depends(get_voice_note_service),
) -> LiveRouteGuideResponse:
    """Retrieve WhatsApp Desktop live virtual-mic routing instructions and device status."""
    return await service.get_live_guide()


@router.get("/capabilities", response_model=VoiceNoteCapabilities)
async def get_voice_note_capabilities(
    service: VoiceNoteService = Depends(get_voice_note_service),
) -> VoiceNoteCapabilities:
    """Probe local ffmpeg installation, available encoders, and audio constraints."""
    return await service.get_capabilities()


@router.post("/convert", response_model=VoiceNoteJobStatus, status_code=status.HTTP_202_ACCEPTED)
async def convert_voice_note(
    file: UploadFile = File(
        ..., description="Audio file to convert (.wav, .mp3, .ogg, .flac, .m4a)"
    ),
    target_format: str = Form(
        "wav", description="Target container format (wav, mp3, ogg, flac, m4a)"
    ),
    model_id: str = Form("default", description="Model ID to apply"),
    model_format: str = Form("fake_model", description="Model format type"),
    pitch_shift: float = Form(0.0, description="Semitone pitch shift"),
    formant_shift: float = Form(0.0, description="Formant shift"),
    service: VoiceNoteService = Depends(get_voice_note_service),
) -> VoiceNoteJobStatus:
    """Submit an audio file for chunked offline conversion."""
    try:
        container_fmt = AudioContainerFormat(target_format.lower().strip())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported target format: '{target_format}'. Allowed: {[f.value for f in AudioContainerFormat]}",
        )

    try:
        mdl_fmt = SupportedModelFormat(model_format.lower().strip())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model format: '{model_format}'. Allowed: {[f.value for f in SupportedModelFormat]}",
        )

    job_params = VoiceNoteJobCreate(
        target_format=container_fmt,
        model_id=model_id,
        model_format=mdl_fmt,
        pitch_shift=pitch_shift,
        formant_shift=formant_shift,
    )

    try:
        file_bytes = await file.read()
        filename = file.filename or "recording.wav"
        return await service.submit_job(
            file_bytes=file_bytes,
            filename=filename,
            job_params=job_params,
        )
    except ResourceConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.message,
        )
    except IncompatibleModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.message,
        )
    except AudioFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        )
    except Exception as exc:
        logger.error("Failed to submit conversion job", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal conversion error: {exc}",
        )


@router.get("/jobs/{job_id}", response_model=VoiceNoteJobStatus)
async def get_job_status(
    job_id: str,
    service: VoiceNoteService = Depends(get_voice_note_service),
) -> VoiceNoteJobStatus:
    """Query conversion progress and status."""
    try:
        return service.get_job_status(job_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )


@router.delete("/jobs/{job_id}", response_model=VoiceNoteJobStatus)
async def cancel_job(
    job_id: str,
    service: VoiceNoteService = Depends(get_voice_note_service),
) -> VoiceNoteJobStatus:
    """Cancel an active conversion job and purge intermediate files."""
    try:
        return await service.cancel_job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )


@router.get("/jobs/{job_id}/source-audio")
async def get_source_audio(
    job_id: str,
    service: VoiceNoteService = Depends(get_voice_note_service),
) -> FileResponse:
    """Stream original source audio for A/B preview."""
    try:
        path = service.get_source_audio_path(job_id)
        media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(
            path=str(path),
            media_type=media_type,
            filename=path.name,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source audio file for job '{job_id}' not found.",
        )


@router.get("/jobs/{job_id}/converted-audio")
async def get_converted_audio(
    job_id: str,
    service: VoiceNoteService = Depends(get_voice_note_service),
) -> FileResponse:
    """Stream converted audio for A/B preview."""
    try:
        path = service.get_converted_audio_path(job_id)
        media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(
            path=str(path),
            media_type=media_type,
            filename=path.name,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Converted audio for job '{job_id}' not found or still processing.",
        )


@router.get("/jobs/{job_id}/download")
async def download_converted_file(
    job_id: str,
    service: VoiceNoteService = Depends(get_voice_note_service),
) -> FileResponse:
    """Download converted audio file as attachment."""
    try:
        path = service.get_converted_audio_path(job_id)
        media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(
            path=str(path),
            media_type=media_type,
            filename=f"converted_{job_id}{path.suffix.lower()}",
            headers={
                "Content-Disposition": f'attachment; filename="converted_{job_id}{path.suffix.lower()}"'
            },
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Converted audio for job '{job_id}' not found.",
        )


@router.post("/cleanup", response_model=RetentionCleanupResponse)
async def trigger_retention_cleanup(
    retention_hours: int = 24,
    service: VoiceNoteService = Depends(get_voice_note_service),
) -> RetentionCleanupResponse:
    """Trigger manual sweep of old voice note files according to retention policy."""
    return service.cleanup_retention(retention_hours=retention_hours)
