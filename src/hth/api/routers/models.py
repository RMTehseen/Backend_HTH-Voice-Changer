"""Voice model management, manifests, licensing, and consent endpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse

from hth.api.dependencies import get_model_service, get_switching_service
from hth.domain.model_contracts import (
    ConsentAttestationRequest,
    ConsentRecord,
    ModelAuditEntry,
    ModelRecord,
    ModelSummary,
    ModelSwitchResult,
)
from hth.services.archive_validator import MAX_COMPRESSED_BYTES, ArchiveSecurityError
from hth.services.license_service import LicensingPolicyViolation
from hth.services.model_service import ModelLifecycleError, ModelNotFoundError, ModelService
from hth.services.switching_service import ModelSwitchError, SwitchingService

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=list[ModelSummary])
async def list_models(
    model_service: ModelService = Depends(get_model_service),
) -> list[ModelSummary]:
    """List all registered voice models with their licensing and consent statuses."""
    return await model_service.list_models()


@router.get("/audit/events", response_model=list[ModelAuditEntry])
async def list_audit_events(
    model_id: str | None = None,
    model_service: ModelService = Depends(get_model_service),
) -> list[ModelAuditEntry]:
    """Retrieve immutable audit log events for model lifecycle actions."""
    return await model_service.get_audit_trail(model_id=model_id)


@router.get("/{model_id}", response_model=ModelRecord)
async def get_model_details(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
) -> ModelRecord:
    """Retrieve comprehensive manifest and asset inventory for a specific model."""
    try:
        return await model_service.get_model(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/import/upload", response_model=ModelRecord, status_code=status.HTTP_201_CREATED)
async def upload_model_archive(
    file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service),
) -> ModelRecord:
    """Upload and validate a voice model archive (.zip, .tar.gz).

    Enforces magic bytes verification, Zip Slip protection, and archive bomb limits.
    Never loads untrusted pickle or .pth bytecode in the API process.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must have a filename."
        )

    # Write uploaded chunks to a temporary file
    temp_dir = tempfile.mkdtemp(prefix="hth_upload_")
    temp_archive = Path(temp_dir) / file.filename

    total_bytes = 0
    try:
        with open(temp_archive, "wb") as out_f:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_COMPRESSED_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Uploaded file exceeds maximum size limit of {MAX_COMPRESSED_BYTES // (1024 * 1024)} MB",
                    )
                out_f.write(chunk)

        record = await model_service.import_archive(temp_archive, original_filename=file.filename)
        return record

    except ArchiveSecurityError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ModelLifecycleError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except LicensingPolicyViolation as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    finally:
        # Clean up uploaded archive file and temp dir
        if temp_archive.exists():
            temp_archive.unlink(missing_ok=True)
        if Path(temp_dir).exists():
            os.rmdir(temp_dir)


@router.post("/{model_id}/consent", response_model=ConsentRecord)
async def submit_consent_attestation(
    model_id: str,
    request: ConsentAttestationRequest,
    model_service: ModelService = Depends(get_model_service),
) -> ConsentRecord:
    """Submit mandatory legal and likeness consent attestation for a registered model."""
    try:
        return await model_service.record_consent(model_id, request)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except LicensingPolicyViolation as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@router.post("/{model_id}/activate", response_model=ModelSwitchResult)
async def activate_model(
    model_id: str,
    switching_service: SwitchingService = Depends(get_switching_service),
) -> ModelSwitchResult:
    """Explicitly switch the active voice conversion model.

    If audio streaming is currently active:
    - Automatically stops and mutes the stream.
    - Swaps model weights and warms up inference.
    - Resumes conversion cleanly.
    - If failure occurs, audio output remains muted (fail-closed, no raw mic passthrough).
    """
    try:
        return await switching_service.switch_model(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except LicensingPolicyViolation as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ModelSwitchError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
async def delete_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
) -> Response:
    """Unregister model and completely delete application-managed disk assets."""
    try:
        await model_service.delete_model(model_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ModelLifecycleError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("/{model_id}/export")
async def export_model_package(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
) -> FileResponse:
    """Export model files as a signed zip archive."""
    try:
        temp_export = Path(tempfile.gettempdir()) / f"{model_id}_export.zip"
        zip_path = await model_service.export_model(model_id, temp_export)
        return FileResponse(
            str(zip_path),
            media_type="application/zip",
            filename=f"{model_id}.zip",
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ModelLifecycleError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
